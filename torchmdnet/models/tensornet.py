import torch
import numpy as np
from typing import Optional, Tuple
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torchmdnet.models.utils import (
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)


#this util creates a tensor from a vector, which will be the antisymmetric part Aij
def vector_to_skewtensor(vector):
    tensor = torch.cross(*torch.broadcast_tensors(vector[...,None], torch.eye(3,3, device=vector.device)[None,None])) ### FIXED: repeat, torch.cross apparently doesn't support broadcasting, that's why I manually added it there
    return tensor.squeeze(0)

#this util creates a symmetric traceFULL tensor from a vector (by means of the outer product of the vector by itself)
#it contributes to the scalar part Iij and the symmetric part Sij
def vector_to_symtensor(vector):
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2)) ### ADDED: final unsqueeze to add the dimension I removed previously
    Iij = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device)
    Sij = 0.5*(tensor+tensor.transpose(-2,-1))-Iij
    tensor = Sij
    return tensor

#this util decomposes an arbitrary tensor into Iij, Aij, Sij
def decompose_tensor(tensor):
    Iij = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device) ### FIXED: repeat
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

def tensor_norm(tensor):
    return (tensor**2).sum((-2,-1))

def tensor_norm1(tensor):
    return (0.5*((tensor.diagonal(offset=0, dim1=-1, dim2=-2)).sum(-1))**2 - 0.5*(
        torch.matrix_power(tensor,2).diagonal(offset=0, dim1=-1, dim2=-2)).sum(-1))**2


class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor,) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max).type(torch.get_default_dtype())

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})" 
 
 
 
 
class TensorNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_linears_tensor=2,
        num_linears_scalar=2,
        num_layers=2,
        num_rbf=32,
        rbf_type = 'expnorm',
        cutoff_lower=0,
        cutoff_upper=5,
        max_num_neighbors=64,
        return_vecs=True,
        loop=True,
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
        self.distance = Distance(cutoff_lower, cutoff_upper, max_num_neighbors, return_vecs, loop=True)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff_lower, cutoff_upper, num_rbf, trainable_rbf)
        #self.distance_expansion = BesselBasis(cutoff_upper, num_rbf, trainable_rbf)
        self.tensor_embedding = TensorNeighborEmbedding(hidden_channels, num_rbf, num_linears_tensor, num_linears_scalar, cutoff_lower, cutoff_upper, trainable_rbf, max_z).jittable()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TensorMessage(num_rbf, hidden_channels, num_linears_scalar, num_linears_tensor, cutoff_lower, cutoff_upper).jittable())
        #self.layers.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        #self.layers.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        #self.layers.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.layers.append(nn.Linear(3*hidden_channels, hidden_channels))
        self.out_norm = nn.LayerNorm(3*hidden_channels)
        self.act = nn.SiLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for i in range(0,self.num_layers):
            self.layers[i].reset_parameters()
        #nn.init.xavier_uniform_(self.layers[-1].weight)
        #self.layers[-1].bias.data.fill_(0)

    def forward(self, z, pos, batch, q: Optional[Tensor] = None, s: Optional[Tensor] = None):
        
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        
        X, Z = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)
        
        for i in range(0,self.num_layers):
            X = self.layers[i](X, edge_index, edge_weight, edge_attr)
            #X = X/(tensor_norm2(X) + 1)[...,None,None]
            
        I, A, S = decompose_tensor(X)
        #I = self.layers[-4](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #A = self.layers[-3](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #S = self.layers[-2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        #A = A.reshape(A.shape[0], A.shape[1], 9)
        #v = torch.stack((A[:,:,5],-A[:,:,2],A[:,:,1]),dim=-1)
        #print(v.shape)
        #M = torch.matmul(I,S)
        #M = I+S
        #print(M.shape)
        #v = v.unsqueeze(-1)
        
        #s = torch.matmul(v.transpose(-1,-2), torch.matmul(M, v)).squeeze().squeeze()
        
              
        #I concatenate here the three ways to obtain scalars
        s = torch.cat((tensor_norm(I),tensor_norm(A),tensor_norm(S)),dim=-1)
        
        x = self.out_norm(s)
        
        x = self.act(self.layers[-1]((x)))        
        
        return x, None, z, pos, batch
            

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
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels)
        
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        #self.cutoff = PolynomialCutoff(cutoff_upper, p=6)
        #self.act = activation()
        self.max_z = max_z
        self.emb = torch.nn.Embedding(max_z, hidden_channels)
        self.emb2 = nn.Linear(2*hidden_channels,hidden_channels)
        
        self.act = nn.SiLU()
        
        #self.act = activation()
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(hidden_channels, 2*hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            linear = nn.Linear(2*hidden_channels, 3*hidden_channels, bias=True)
            self.linears_scalar.append(linear)
        self.reset_parameters()

        self.init_norm = nn.LayerNorm(hidden_channels)
        

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
        
        W1 = (self.distance_proj1(edge_attr)) * C.view(-1,1)
        W2 = (self.distance_proj2(edge_attr)) * C.view(-1,1)
        W3 = (self.distance_proj3(edge_attr)) * C.view(-1,1)
        
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        
        edge_tensor = W1[...,None,None] * (vector_to_skewtensor(edge_vec)[:,None,:,:])
        edge_tensor = edge_tensor + W2[...,None,None] * vector_to_symtensor(edge_vec)[:,None,:,:]
        edge_tensor = edge_tensor + W3[...,None,None] * torch.eye(3,3, device=edge_vec.device)[None,None,:,:]
        
        # propagate_type: (Z: Tensor, edge_tensor: Tensor)
        X = self.propagate(edge_index, Z=Z, edge_tensor=edge_tensor, size=None)
        
        Iij, Aij, Sij = decompose_tensor(X)
        
        #norm = torch.cat((tensor_norm(Iij),tensor_norm(Aij),tensor_norm(Sij)),dim=-1)
        norm = tensor_norm(X)
        #X = X / (norm + 1)[...,None,None]
        
        norm = self.init_norm(norm)
        
          
        Iij = self.linears_tensor[0](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Aij = self.linears_tensor[1](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Sij = self.linears_tensor[2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
        for j in range(self.num_linears_scalar):
            norm = self.act(self.linears_scalar[j](norm))
        #norm = self.linears_scalar[self.num_linears_scalar-1](norm)
        
        #Iij = Iij / (tensor_norm(Iij)+1)[...,None,None]
        #Aij = Aij / (tensor_norm(Aij)+1)[...,None,None]
        #Sij = Sij / (tensor_norm(Sij)+1)[...,None,None]
            
        X = Iij + Aij + Sij
        
        X = new_radial_tensor(X,norm.reshape(norm.shape[0],self.hidden_channels,3))
        
        return X, edge_tensor
      

    def message(self, Z_i, Z_j, edge_tensor):
        
        Zij = self.emb2(torch.cat((Z_i,Z_j),dim=-1))
        msg = Zij[...,None,None]*(edge_tensor)
        
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
        #self.cutoff = PolynomialCutoff(cutoff_upper, p=6)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(num_rbf, hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            self.linears.append(nn.Linear(hidden_channels, 2*hidden_channels, bias=True))
            self.linears.append(nn.Linear(2*hidden_channels, 3*hidden_channels, bias=True))
        #self.linears.append(nn.Linear(2*hidden_channels, 3*hidden_channels, bias=True))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        #self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        #self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        #self.linears_scalar = nn.ModuleList()
        #self.linears_scalar.append(nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True))
        #for _ in range(num_linears_scalar-1):
            #linear = nn.Linear(3*hidden_channels, 3*hidden_channels, bias=True)
            #self.linears_scalar.append(linear)
        
        
        #self.w = nn.Parameter(torch.ones(hidden_channels))
        #self.w1 = nn.Parameter(torch.ones(hidden_channels))
        #self.w2 = nn.Parameter(torch.ones(hidden_channels))
        
        
        #self.lay_norm1 = nn.LayerNorm(hidden_channels)
        #self.lay_norm2 = nn.LayerNorm(hidden_channels)
       
        self.act = nn.SiLU()
        
        
    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        #for linear in self.linears_scalar:
            #linear.reset_parameters()
            
    
    def forward(self, X, edge_index, edge_weight, edge_attr):
        
        C = self.cutoff(edge_weight)
        
        for i in range(self.num_linears_scalar+1):
            edge_attr = self.act(self.linears[i](edge_attr))
        edge_attr = (edge_attr * C.view(-1,1)).reshape(edge_attr.shape[0], self.hidden_channels, 3)
        
        
        X = X / (tensor_norm(X)+1)[...,None,None]
        
        Iij, Aij, Sij = decompose_tensor(X)
        
        Iij = self.linears[-4](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Aij = self.linears[-3](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Sij = self.linears[-2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        
        Y = Iij + Aij + Sij
        
        # propagate_type: (Y: Tensor, edge_attr: Tensor)
        msg = self.propagate(edge_index, Y=Y, edge_attr=edge_attr, size=None)
       
        A = torch.matmul(msg,Y)
        B = torch.matmul(Y,msg)
        Iij, Aij, Sij = decompose_tensor(A+B)
        
        Y = Iij + Aij + Sij
        
        #new
        norm1 = tensor_norm(Y)
        Y = Y / (norm1 + 1)[...,None,None]
        
        Y = self.linears[-1](Y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        
        #usual
        #norm1 = tensor_norm(Y)
        #Y = Y / (norm1 + 1)[...,None,None]
        
        #Y = X + Y        
    
        Iij, Aij, Sij = decompose_tensor(Y)
        
        Iij = self.linears_tensor[0](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Aij = self.linears_tensor[1](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Sij = self.linears_tensor[2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
            
        
        dX = Iij + Aij + Sij
        
        #norm = tensor_norm(dX)
        #norm1 = tensor_norm(dX)
        #for j in range(self.num_linears_scalar):
            #norm = self.act(self.linears_scalar[j](norm))
            
        #dX = new_radial_tensor(dX,norm.reshape(norm.shape[0],self.hidden_channels,3))
        
        #dX = self.linears[-1](dX.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        #norm1 = tensor_norm(dX)
        #dX = dX / (norm1+1)[...,None,None]
        
        dX = dX + torch.matmul(dX,dX)
        
        X = X + dX
        
        return X
       
       
    def message(self, Y_j, edge_attr):
        
        msg = new_radial_tensor(Y_j, edge_attr)
        
        return msg