import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torchmdnet.models.utils import (
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)


#this util creates a tensor from a vector, which will be the antisymmetric part Aij
def vector_to_skewtensor(vector):
    tensor = torch.cross(*torch.broadcast_tensors(vector[...,None], torch.eye(3,3, device=vector.device)[None,None]))
    return tensor.squeeze(0)

#this util creates a symmetric traceFULL tensor from a vector (by means of the outer product of the vector by itself)
#it contributes to the scalar part Iij and the symmetric part Sij
def vector_to_symtensor(vector):
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    return tensor

#this util decomposes an arbitrary tensor into Iij, Aij, Sij
def decompose_tensor(tensor):
    Iij = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device)
    Aij = 0.5*(tensor-tensor.transpose(-2,-1))
    Sij = 0.5*(tensor+tensor.transpose(-2,-1))-Iij
    return Iij, Aij, Sij

#this util modifies the tensor by adding 3 rotationally invariant functions (fijs) to Iij, Aij, Sij
def new_radial_tensor(tensor, fij):
    new_tensor = ((fij[...,0] - fij[...,2]) * tensor.diagonal(offset=0, dim1=-1, dim2=-2).mean(-1))[...,None,None] * (
        torch.eye(3,3, device=tensor.device))
    new_tensor = new_tensor + 0.5 * (fij[...,1]+fij[...,2])[...,None,None] * tensor
    new_tensor = new_tensor - 0.5 * (fij[...,1]-fij[...,2])[...,None,None] * tensor.transpose(-2,-1)
    return new_tensor

#I tried different implementations of the norm, since torch.norm was unstable, this works well
def tensor_norm(tensor):
    return (tensor**2).sum((-2,-1))


class TensorNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        #number of linears applied to tensors in tensor mlp (every linear represents 3 actual linears, one for Iij,Aij,Sij)
        num_linears_tensor=1,
        #number of linears used in mlps dealing with scalar quantities
        num_linears_scalar=2,
        num_layers=3,
        num_rbf=32,
        rbf_type = 'expnorm',
        cutoff_lower=0,
        cutoff_upper=5,
        max_num_neighbors=32,
        return_vecs=True,
        loop=False,
        trainable_rbf=False,
        max_z = 100,
    ):
        super(TensorNetwork, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_linears_tensor = num_linears_tensor
        self.num_linears_scalar = num_linears_scalar
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.distance = Distance(cutoff_lower, cutoff_upper, max_num_neighbors, return_vecs, loop)
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        #self.act = activation()

        self.tensor_embedding = TensorNeighborEmbedding(hidden_channels, num_rbf, num_linears_tensor, num_linears_scalar, cutoff_lower,
                                                        cutoff_upper, trainable_rbf, max_z).jittable()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TensorMessage(num_rbf, hidden_channels, num_linears_scalar, num_linears_tensor, cutoff_lower, cutoff_upper).jittable())
        #self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(3*hidden_channels, hidden_channels))
        #self.out_norm = nn.LayerNorm(3*hidden_channels)
        self.act = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for i in range(0,self.num_layers):
            self.layers[i].reset_parameters()
        self.layers[-1].reset_parameters()

        #self.out_norm.reset_parameters()

    def forward(self, z, pos, batch):

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)

        for i in range(0,self.num_layers):
            X = self.layers[i](X, edge_index, edge_weight, edge_attr)

        Iij, Aij, Sij = decompose_tensor(X)

        #I concatenate here the three ways to obtain scalars: 0x0->0, 1x1->0, 2x2->0
        x = (torch.cat((tensor_norm(Iij),tensor_norm(Aij),tensor_norm(Sij)),dim=-1))
        #x = tensor_norm(X)

        #x = self.out_norm(x)

        x = self.act(self.layers[-1]((x)))

        return x, None, z, pos, batch


#initialize tensor features for each node, which is a superposition of tensors computed from the edge vectors
class TensorNeighborEmbedding(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        num_linears_tensor,
        num_linears_scalar,
        cutoff_lower,
        cutoff_upper,
        trainable_rbf=True,
        max_z=100,
    ):
        super(TensorNeighborEmbedding, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.num_linears_tensor = num_linears_tensor
        self.num_linears_scalar = num_linears_scalar
        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels)

        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        #self.act = activation()
        self.max_z = max_z
        self.emb = torch.nn.Embedding(max_z, hidden_channels)
        self.emb2 = nn.Linear(2*hidden_channels,hidden_channels)

        self.act = nn.Tanh()

        #self.act = activation()
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            linear = nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True)
            self.linears_scalar.append(linear)
        self.reset_parameters()



    def reset_parameters(self):
        self.emb.reset_parameters()
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()

    def forward(self, z, edge_index, edge_weight, edge_vec, edge_attr):

        Z = self.emb(z)

        C = self.cutoff(edge_weight)

        W1 = (self.distance_proj1(edge_attr) * C.view(-1,1))

        W2 = (self.distance_proj2(edge_attr) * C.view(-1,1))

        edge_vec = edge_vec / edge_weight.unsqueeze(-1)

        #here I will create tensors for every edge from the edge_vectors using utils
        #first a contribution to the skewsymmetric part of the tensor (weighted by scalar functions from the edge rbfs)
        edge_tensor = W1[...,None,None] * vector_to_skewtensor(edge_vec)[:,None,:,:]
        #second a symmetric traceFULL contribution, which could be further decomposed into scalar and symmetric traceless contributions
        edge_tensor = edge_tensor + W2[...,None,None] * vector_to_symtensor(edge_vec)[:,None,:,:]

        # propagate_type: (Z: Tensor, edge_tensor: Tensor)
        X = self.propagate(edge_index, Z=Z, edge_tensor=edge_tensor, size=None)

        #first I decompose the tensor so I can apply different Linears to different components, the idea is to be able to do:
        # Wi Iij + Wa Aij + Ws Sij
        Iij, Aij, Sij = decompose_tensor(X)

        #I concatenate here the three ways to obtain scalars from current tensor representation: 0x0->0, 1x1->0, 2x2->0
        norm = torch.cat((tensor_norm(Iij),tensor_norm(Aij),tensor_norm(Sij)),dim=-1)

        # Wi Iij + Wa Aij + Ws Sij
        Iij = self.linears_tensor[0](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Aij = self.linears_tensor[1](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Sij = self.linears_tensor[2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute

        #create new rotationally invariant functions (scalars) from previous concatenation of scalars with a MLP
        for j in range(self.num_linears_scalar):
            norm = self.act(self.linears_scalar[j](norm))

        # X = Wi Iij + Wa Aij + Ws Sij
        X = Iij + Aij + Sij

        #I use the new rotationally invariant functions (scalars) to modify the tensor using new_radial_tensor, which multiplies three
        #different scalar functions to the the different components of the tensor
        X = new_radial_tensor(X,norm.reshape(norm.shape[0],self.hidden_channels,3))

        return X


    def message(self, Z_i, Z_j, edge_tensor):

        Zij = self.emb2(torch.cat((Z_i,Z_j),dim=-1))
        msg = Zij[...,None,None]*edge_tensor

        return msg


#message passing step, combining curent node features with neighbors message
class TensorMessage(MessagePassing):
    def __init__(
        self,
        num_rbf,
        hidden_channels,
        num_linears_scalar,
        num_linears_tensor,
        cutoff_lower,
        cutoff_upper,
    ):
        super(TensorMessage, self).__init__(aggr="add", node_dim=0)

        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.num_linears_scalar = num_linears_scalar
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(num_rbf, 3*hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            self.linears.append(nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))

        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            linear = nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True)
            self.linears_scalar.append(linear)

        self.act = nn.Tanh()


    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()


    def forward(self, X, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        #here I create rotationally invariant functions from the RBF expansion of neighbors distances
        #these will be used to modify the message by using the new_radial_tensor util
        for i in range(self.num_linears_scalar):
            edge_attr = self.act(self.linears[i](edge_attr))
        edge_attr = (edge_attr * C.view(-1,1)).reshape(edge_attr.shape[0], self.hidden_channels, 3)

        Iij, Aij, Sij = decompose_tensor(X)

        Iij = self.linears[-4](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Aij = self.linears[-5](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Sij = self.linears[-6](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute

        Y = Iij + Aij + Sij

        # propagate_type: (Y: Tensor, edge_attr: Tensor)
        msg = self.propagate(edge_index, Y=Y, edge_attr=edge_attr)

        #interaction of current node features with message (could be changed, but the idea is to be matrix products or polynomials of matrix products)
        Y = torch.matmul(msg, torch.matmul(Y, msg))

        Iij, Aij, Sij = decompose_tensor(Y)

        #I apply linears
        Iij = self.linears[-1](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Aij = self.linears[-2](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Sij = self.linears[-3](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        #I update the node representation
        #previously: X = X + I + A + S (unstable)
        Y = X + Iij + Aij + Sij

        ####
        #normalization block, not using it makes the net unstable
        Y = Y / (tensor_norm(Y) + 1)[...,None,None]
        #

        Iij, Aij, Sij = decompose_tensor(Y)
        norm = torch.cat((tensor_norm(Iij),tensor_norm(Aij),tensor_norm(Sij)),dim=-1)

        Iij = self.linears_tensor[0](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Aij = self.linears_tensor[1](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Sij = self.linears_tensor[2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for j in range(self.num_linears_scalar):
            norm = self.act(self.linears_scalar[j](norm))

        #for j in range(self.num_linears_scalar-1):
            #norm = self.act(self.linears_scalar[j](norm))
        #norm = self.linears_scalar[self.num_linears_scalar-1](norm)

        dX = Iij + Aij + Sij

        dX = new_radial_tensor(dX,norm.reshape(norm.shape[0],self.hidden_channels,3))

        #polynomial non-linearity
        dX = dX - 0.5*torch.matrix_power(dX,2)

        X = X + dX

        return X


    def message(self, Y_j, edge_attr):

        #here what I do is multiply three different scalar functions produced from edge attributes (rbfs) to the three different components
        #of the tensor representation of the neighbor
        msg = new_radial_tensor(Y_j, edge_attr)

        return msg
