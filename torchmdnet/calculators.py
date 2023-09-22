import torch
from torchmdnet.models.model import load_model

# dict of preset transforms
tranforms = {
    "eV/A -> kcal/mol/A": lambda energy, forces: (
        energy * 23.0609,
        forces * 23.0609,
    ),  # eV->kcal/mol, eV/A -> kcal/mol/A
    "Hartree/Bohr -> kcal/mol/A": lambda energy, forces: (
        energy * 627.509,
        forces * 627.509 / 0.529177,
    ),  # Hartree -> kcal/mol, Hartree/Bohr -> kcal/mol/A
    "Hartree/A -> kcal/mol/A": lambda energy, forces: (
        energy * 627.509,
        forces * 627.509,
    ),  # Hartree -> kcal/mol, Hartree/A -> kcal/mol/A
}


class External:
    """
    The External class is used to calculate the energy and forces of an external potential, such as a neural network. The class is initialized with the path to the neural network
    ckpt, the embeddings, the device on which the neural network should be run and the output_transform argument. The output_transform is used to give a function that transform
    the energy and the forces, this could be a preset transform or a custom function. In this way there is no constraint to the units of the neural network, the user can choose
    the units of the simulation and the neural network will be automatically converted to the units of the simulation. The function should take two arguments, the energy and the
    forces, and return the transformed energy and the transformed forces.
    """

    def __init__(self, netfile, embeddings, device="cpu", output_transform=None, use_cuda_graph=False):
        self.model = load_model(netfile, device=device, derivative=True)
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )
        self.model.eval()

        if not output_transform:
            self.output_transformer = lambda energy, forces: (
                energy,
                forces,
            )  # identity
        elif output_transform in tranforms.keys():
            self.output_transformer = tranforms[output_transform]
        else:
            self.output_transformer = eval(output_transform)
        if not torch.cuda.is_available() and use_cuda_graph:
            raise ValueError("CUDA graphs are only available if CUDA is")
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph = None
        self.energy = None
        self.forces = None
        self.pos = None
        self.stream = None

    def _init_cuda_graph(self):
        self.stream = torch.cuda.Stream()
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.energy, self.forces = self.model(self.embeddings, self.pos, self.batch)
            with torch.cuda.graph(self.cuda_graph):
                self.energy, self.forces = self.model(self.embeddings, self.pos, self.batch)
    torch.cuda.make_graphed_callables
    def calculate(self, pos, box):
        pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)
        if self.use_cuda_graph:
            if self.pos is None:
                self.pos = pos.clone()
                self.pos.requires_grad_(False)
            self.pos.copy_(pos)
            self.pos.requires_grad_(pos.requires_grad)
            if self.cuda_graph is None:
                self._init_cuda_graph()
            with torch.cuda.stream(self.stream):
                self.cuda_graph.replay()
        else:
            self.energy, self.forces = self.model(self.embeddings, pos, self.batch)
        return self.output_transformer(
            self.energy.clone().detach(), self.forces.clone().reshape(-1, self.n_atoms, 3).detach()
        )
