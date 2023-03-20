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
@torch.compile
def vector_to_skewtensor(vector):
    N=vector.shape[0]
    eye = torch.eye(3, device=vector.device).unsqueeze(0).repeat(N, 1, 1)
    expVec = vector.unsqueeze(1).repeat(1, 3, 1).transpose_(1,2)
    tensor = torch.cross(expVec, eye)
    return tensor

#this util creates a symmetric traceFULL tensor from a vector (by means of the outer product of the vector by itself)
#it contributes to the scalar part Iij and the symmetric part Si
@torch.compile
def vector_to_symtensor(vector):
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2)) ### ADDED: final unsqueeze to add the dimension I removed previously
    Iij = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device)
    Sij = 0.5*(tensor+tensor.transpose(-2,-1))-Iij
    tensor = Sij
    return tensor


#this util decomposes an arbitrary tensor into Iij, Aij, Si
@torch.compile
def decompose_tensor(tensor):
    Iij = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[...,None,None] * torch.eye(3,3, device=tensor.device) ### FIXED: repeat
    Aij = 0.5*(tensor-tensor.transpose(-2,-1))
    Sij = 0.5*(tensor+tensor.transpose(-2,-1))-Iij
    return Iij, Aij, Sij

#this util modifies the tensor by adding 3 rotationally invariant functions (fijs) to Iij, Aij, Sij
@torch.compile
def new_radial_tensor(tensor, fij):
    new_tensor = ((fij[...,0] - fij[...,2]) * tensor.diagonal(offset=0, dim1=-1, dim2=-2).mean(-1))[...,None,None] * (
        torch.eye(3,3, device=tensor.device))
    new_tensor = new_tensor + 0.5 * (fij[...,1]+fij[...,2])[...,None,None] * tensor
    new_tensor = new_tensor - 0.5 * (fij[...,1]-fij[...,2])[...,None,None] * tensor.transpose(-2,-1)
    return new_tensor

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
        cutoff_lower=0.0,
        cutoff_upper=5.0,
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
        self.distance = MyDistance(cutoff_lower, cutoff_upper, max_num_neighbors, return_vecs, loop=True)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff_lower, cutoff_upper, num_rbf, trainable_rbf)
        self.tensor_embedding = TensorNeighborEmbedding(hidden_channels, num_rbf, num_linears_tensor, num_linears_scalar, cutoff_lower, cutoff_upper, trainable_rbf, max_z).jittable()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TensorMessage(num_rbf, hidden_channels, num_linears_scalar, num_linears_tensor, cutoff_lower, cutoff_upper).jittable())
        self.extra_layer = nn.Linear(3*hidden_channels, hidden_channels)
        self.out_norm = nn.LayerNorm(3*hidden_channels)
        self.act = nn.SiLU()

    def reset_parameters(self):
        pass

    def forward(self, z: Tensor, pos: Tensor, batch: Tensor, q: Optional[Tensor], s: Optional[Tensor]):
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)
        for i,layer in enumerate(self.layers):
            X = layer(X, edge_index, edge_weight, edge_attr)
        I, A, S = decompose_tensor(X)
        #I concatenate here the three ways to obtain scalars
        s = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.out_norm(s)
        x = self.act(self.extra_layer(x))
        return x, None, z, pos, batch

def normalize_vec(edge_index, edge_vec):
    mask = edge_index[0] != edge_index[1]
    edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
    return edge_vec

@torch.compile
def IAS(linears_tensor, Iij, Aij, Sij):
    Iij = linears_tensor[0](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    Aij = linears_tensor[1](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    Sij = linears_tensor[2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return Iij, Aij, Sij

@torch.compile
def create_edge_tensor(cutoff, distance_proj, edge_weight, edge_attr, edge_vec):
    C = cutoff(edge_weight)
    W1 = (distance_proj[0](edge_attr)) * C.view(-1,1)
    W2 = (distance_proj[1](edge_attr)) * C.view(-1,1)
    W3 = (distance_proj[2](edge_attr)) * C.view(-1,1)
    edge_tensor = W1[...,None,None] * (vector_to_skewtensor(edge_vec)[:,None,:,:])
    edge_tensor = edge_tensor + W2[...,None,None] * vector_to_symtensor(edge_vec)[:,None,:,:]
    edge_tensor = edge_tensor + W3[...,None,None] * torch.eye(3,3, device=edge_vec.device)[None,None,:,:]
    return edge_tensor

class TensorNeighborEmbedding(MessagePassing):
    def __init__(
        self,
        hidden_channels = 128,
        num_rbf = 32,
        num_linears_tensor = 2,
        num_linears_scalar = 2,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        trainable_rbf=False,
        max_z=128,
    ):
        super(TensorNeighborEmbedding, self).__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_linears_tensor = num_linears_tensor
        self.num_linears_scalar = num_linears_scalar
        self.distance_proj = nn.ModuleList()
        self.distance_proj.append(nn.Linear(num_rbf, hidden_channels))
        self.distance_proj.append(nn.Linear(num_rbf, hidden_channels))
        self.distance_proj.append(nn.Linear(num_rbf, hidden_channels))
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = torch.compile(torch.nn.Embedding(max_z, hidden_channels))
        self.emb2 = nn.Linear(2*hidden_channels,hidden_channels)
        self.act = nn.SiLU()
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(hidden_channels, 2*hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            linear = nn.Linear(2*hidden_channels, 3*hidden_channels, bias=True)
            self.linears_scalar.append(linear)
        self.init_norm = torch.compile(nn.LayerNorm(hidden_channels))

    @torch.compile
    def forward(self, z: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_vec: Tensor, edge_attr: Tensor) -> Tensor:
        Z = self.emb(z)
        edge_vec = normalize_vec(edge_index, edge_vec)
        edge_tensor = create_edge_tensor(self.cutoff, self.distance_proj, edge_weight, edge_attr, edge_vec)
        # propagate_type: (Z: Tensor, edge_tensor: Tensor)
        X = self.propagate(edge_index, Z=Z, edge_tensor=edge_tensor, size=None)
        Iij, Aij, Sij = decompose_tensor(X)
        norm = tensor_norm(X)
        norm = self.init_norm(norm)
        Iij, Aij, Sij = IAS(self.linears_tensor, Iij, Aij, Sij)
        norm = self.act(self.linears_scalar[0](norm))
        norm = self.act(self.linears_scalar[1](norm))
        X = Iij + Aij + Sij
        X = new_radial_tensor(X,norm.reshape(norm.shape[0],self.hidden_channels,3))
        return X

    @torch.compile
    def message(self, Z_i: Tensor, Z_j: Tensor, edge_tensor: Tensor):
        Zij = self.emb2(torch.cat((Z_i,Z_j),dim=-1))
        msg = Zij[...,None,None]*(edge_tensor)
        return msg

#message passing step, combining curent node features with neighbors message
class TensorMessage(MessagePassing):
    def __init__(
        self,
        num_rbf = 32,
        hidden_channels = 128,
        num_linears_scalar = 2,
        num_linears_tensor = 2,
        cutoff_lower= 0.0,
        cutoff_upper = 5.0,
    ):
        super(TensorMessage, self).__init__(aggr="add", node_dim=0)
        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.num_linears_scalar = num_linears_scalar
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(num_rbf, hidden_channels, bias=True))
        for _ in range(num_linears_scalar-1):
            self.linears.append(nn.Linear(hidden_channels, 2*hidden_channels, bias=True))
            self.linears.append(nn.Linear(2*hidden_channels, 3*hidden_channels, bias=True))
        self.other_linears = nn.ModuleList()
        self.other_linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.other_linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.other_linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.other_linears.append(nn.Linear(hidden_channels,hidden_channels, bias=False))
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.linears_tensor.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.act = nn.SiLU()

    @torch.compile
    def forward(self, X: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        C = self.cutoff(edge_weight)
        for i,linear in enumerate(self.linears):
            edge_attr = self.act(linear(edge_attr))
        edge_attr = (edge_attr * C.view(-1,1)).reshape(edge_attr.shape[0], self.hidden_channels, 3)
        X = X / (tensor_norm(X)+1)[...,None,None]
        Iij, Aij, Sij = decompose_tensor(X)
        Iij = self.other_linears[-4](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Aij = self.other_linears[-3](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Sij = self.other_linears[-2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Y = Iij + Aij + Sij
        # propagate_type: (Y: Tensor, edge_attr: Tensor)
        msg = self.propagate(edge_index, Y=Y, edge_attr=edge_attr, size=None)
        A = torch.matmul(msg,Y)
        B = torch.matmul(Y,msg)
        Iij, Aij, Sij = decompose_tensor(A+B)
        Y = Iij + Aij + Sij
        norm1 = tensor_norm(Y)
        Y = Y / (norm1 + 1)[...,None,None]
        Y = self.other_linears[-1](Y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Iij, Aij, Sij = decompose_tensor(Y)
        Iij = self.linears_tensor[0](Iij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Aij = self.linears_tensor[1](Aij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        Sij = self.linears_tensor[2](Sij.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) ### FIXED: replace transpose and reshape by permute
        dX = Iij + Aij + Sij
        dX = dX + torch.matmul(dX,dX)
        X = X + dX
        return X
    @torch.compile
    def message(self, Y_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = new_radial_tensor(Y_j, edge_attr)
        return msg
