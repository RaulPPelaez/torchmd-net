import torch
import numpy as np
from typing import Optional, Tuple
from torch import Tensor, nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torchmdnet.models.utils import (
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)

enable_nvtx = False

def tensor_range_push(name):
    if enable_nvtx:
        torch.cuda.nvtx.range_push(name)

def tensor_range_pop():
    if enable_nvtx:
        torch.cuda.nvtx.range_pop()

#this util creates a tensor from a vector, which will be the antisymmetric part Aij
@torch.compile
def vector_to_skewtensor(vector):
    tensor = torch.cross(*torch.broadcast_tensors(vector[...,None], torch.eye(3,3, device=vector.device)[None,None]))
    return tensor.squeeze(0)

#this util creates a symmetric traceFULL tensor from a vector (by means of the outer product of the vector by itself)
#it contributes to the scalar part Iij and the symmetric part Si
@torch.compile
def vector_to_symtensor(vector):
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device)
    S = 0.5 * (tensor + tensor.transpose(-2,-1)) - I
    return S

#chatgpt suggestion
#def vector_to_symtensor(vector):
    #tensor = torch.ger(vector, vector)
    #I = torch.eye(3, 3, device=vector.device).unsqueeze(0)
    #S = 0.5 * (tensor + tensor.transpose(0, 1)) - torch.sum(I * tensor, dim=0, keepdim=True)
    #return S


#this util decomposes an arbitrary tensor into Iij, Aij, Si
@torch.compile
def decompose_tensor(tensor):
    Iij = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device)
    Aij = 0.5*(tensor-tensor.transpose(-2,-1))
    Sij = 0.5*(tensor+tensor.transpose(-2,-1))-Iij
    return Iij, Aij, Sij


def new_radial_tensor(tensor1, tensor2, tensor3, fij1, fij2, fij3):
    I = (fij1)[...,None,None] * tensor1
    A = (fij2)[...,None,None] * tensor2
    S = (fij3)[...,None,None] * tensor3
    return I, A, S

@torch.compile
def tensor_norm(tensor):
    return (tensor**2).sum((-2,-1))

from torch_cluster import radius_graph

class MyDistance(nn.Module):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(MyDistance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos: Tensor , batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = radius_graph(
            pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors + 1,
        )
        # make sure we didn't miss any neighbors due to max_num_neighbors
        # assert not (
        #     torch.unique(edge_index[0], return_counts=True)[1] > self.max_num_neighbors
        # ).any(), (
        #     "The neighbor search missed some atoms due to max_num_neighbors being too low. "
        #     "Please increase this parameter to include the maximum number of atoms within the cutoff."
        # )

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        mask: Optional[torch.Tensor] = None
        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        if self.loop and mask is not None:
            # keep self loops even though they might be below the lower cutoff
            lower_mask = lower_mask | ~mask
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        edge_vec = edge_vec[lower_mask]
        return edge_index, edge_weight, edge_vec

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

        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)

        for i in range(0,self.num_layers):
            X = self.layers[i](X, edge_index, edge_weight, edge_attr)

        I, A, S = decompose_tensor(X)

        x = torch.cat((tensor_norm(I),tensor_norm(A),tensor_norm(S)),dim=-1)

        x = self.out_norm(x)

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
        self.max_z = max_z
        self.emb = torch.compile(torch.nn.Embedding(max_z, hidden_channels))
        self.emb2 = nn.Linear(2*hidden_channels,hidden_channels)

        self.act = nn.SiLU()
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(torch.compile(nn.Linear(hidden_channels, hidden_channels, bias=False)))
        self.linears_tensor.append(torch.compile(nn.Linear(hidden_channels, hidden_channels, bias=False)))
        self.linears_tensor.append(torch.compile(nn.Linear(hidden_channels, hidden_channels, bias=False)))
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

        I, A, S = new_radial_tensor(torch.eye(3,3, device=edge_vec.device)[None,None,:,:],
                                    vector_to_skewtensor(edge_vec)[...,None,:,:],
                                    vector_to_symtensor(edge_vec)[...,None,:,:],
                                    W1, W2, W3)

        # propagate_type: (Z: Tensor, I: Tensor, A: Tensor, S: Tensor)
        I, A, S = self.propagate(edge_index, Z=Z, I=I, A=A, S=S, size=None)

        norm = tensor_norm(I + A + S)

        norm = self.init_norm(norm)

        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for j in range(self.num_linears_scalar):
            norm = self.act(self.linears_scalar[j](norm))

        norm = norm.reshape(norm.shape[0],self.hidden_channels,3)

        I, A, S = new_radial_tensor(I, A, S, norm[...,0], norm[...,1], norm[...,2])

        X = I + A + S

        return X


    def message(self, Z_i, Z_j, I, A, S):
        zij = torch.cat((Z_i,Z_j),dim=-1)
        Zij = self.emb2(zij)
        I = Zij[...,None,None]*I
        A = Zij[...,None,None]*A
        S = Zij[...,None,None]*S

        return I, A, S

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        I, A, S = features
        I = scatter(I, index, dim=self.node_dim, dim_size=dim_size)
        A = scatter(A, index, dim=self.node_dim, dim_size=dim_size)
        S = scatter(S, index, dim=self.node_dim, dim_size=dim_size)

        return I, A, S

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return inputs




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
        #self.linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))

        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))

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

        I, A, S = decompose_tensor(X)

        I = self.linears[-3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears[-2](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears[-1](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        Y = I + A + S

        # propagate_type: (I: Tensor, A: Tensor, S: Tensor, edge_attr: Tensor)
        Im, Am, Sm = self.propagate(edge_index, I=I, A=A, S=S, edge_attr=edge_attr, size=None)

        msg = Im + Am + Sm

        A = torch.matmul(msg,Y)
        B = torch.matmul(Y,msg)

        I, A, S = decompose_tensor(A+B)

        #Y = I + A + S

        norm = tensor_norm(I + A + S)

        I = I / (norm + 1)[...,None,None]
        A = A / (norm + 1)[...,None,None]
        S = S / (norm + 1)[...,None,None]

        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        dX = I + A + S

        dX = dX + torch.matmul(dX,dX)

        X = X + dX

        return X


    def message(self, I_j, A_j, S_j, edge_attr):

        I, A, S = new_radial_tensor(I_j, A_j, S_j, edge_attr[...,0], edge_attr[...,1], edge_attr[...,2])

        return I, A, S


    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        I, A, S = features
        I = scatter(I, index, dim=self.node_dim, dim_size=dim_size)
        A = scatter(A, index, dim=self.node_dim, dim_size=dim_size)
        S = scatter(S, index, dim=self.node_dim, dim_size=dim_size)

        return I, A, S

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return inputs
