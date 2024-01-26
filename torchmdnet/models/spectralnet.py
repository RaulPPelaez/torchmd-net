import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torchmdnet.models.utils import (
    CosineCutoff,
    rbf_class_mapping,
    act_class_mapping,
)

class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, activation="silu"):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(out_channels, out_channels))
        self.activation = act_class_mapping[activation]

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

class AtomicEncoder(nn.Module):
    def __init__(self, max_z, hidden_channels, num_layers, activation="silu"):
        super(AtomicEncoder, self).__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.mlp = MLP(hidden_channels, hidden_channels, num_layers, activation=activation)

    def forward(self, atomic_numbers):
        x = self.mlp(self.embedding(atomic_numbers))
        return x


def delta(pos, features, box_size, grid_size):
    """Computes the weight of an atom at pos on the grid."""
class SpectralNet(nn.Module):

    def __init__(
            self,
            hidden_channels=128,
            num_layers=2,
            num_rbf=32,
            rbf_type="expnorm",
            activation="silu",
            max_z=128,
            dtype=torch.float32,
            box_vecs=None,
    ):
        super(SpectralNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        num_encoder_layers = num_layers
        self.encoder = AtomicEncoder(max_z, hidden_channels, num_encoder_layers, activation=activation)
        assert box_vecs is not None

        self.box_size = torch.tensor((box_vecs[0,0], box_vecs[1,1], box_vecs[2,2]), dtype=dtype)
        self.grid_size = [128, 128, 128]
        pass

    def reset_parameters(self):
        pass

    def spread(self, pos, atomic_features, batch):
        """Transforms atomic features into a feature field, using a Gaussian to spread the features."""
        nbatch = (batch.max() + 1).item()
        nx, ny, nz = self.grid_size
        field_data = torch.zeros((nbatch, self.hidden_channels, nz, ny, nx), dtype=pos.dtype, device=pos.device)


    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        atomic_features = self.encoder(z) # shape: (natoms, hidden_channels)
        feature_field = spread(pos, atomic_features, batch) # shape: (nbatch, hidden_channels, nz, ny, nx)
        feature_field_fourier = fourierTransform(feature_field) # shape (nbatch, hidden_channels, nz, ny, nx)
        energy_field_fourier = convolveFeatureField(feature_field_fourier) # shape (nbatch, nz, ny, nx)
        energy_field = inverseFourierTransform(energy_field_fourier)  # shape (nbatch, nz, ny, nx)
        energy = interpolate(pos, energy_field) # shape (natoms,)
        return energy, None, z, pos, batch
