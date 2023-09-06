from abc import ABCMeta
import re
from typing import Optional, List, Tuple, Dict, Union
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torch_scatter import scatter
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet.models.utils import dtype_mapping
from torchmdnet import priors
from lightning_utilities.core.rank_zero import rank_zero_warn
import warnings


def create_model(args, prior_model=None, mean=None, std=None):
    """Create a model from the given arguments.
    See :func:`get_args` in scripts/train.py for a description of the arguments.
    Parameters
    ----------
        args (dict): Arguments for the model.
        prior_model (nn.Module, optional): Prior model to use. Defaults to None.
        mean (torch.Tensor, optional): Mean of the training data. Defaults to None.
        std (torch.Tensor, optional): Standard deviation of the training data. Defaults to None.
    Returns
    -------
        nn.Module: An instance of the TorchMD_Net model.
    """
    dtype = dtype_mapping[args["precision"]]
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
        dtype=dtype
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"],
            aggr=args["aggr"],
            neighbor_embedding=args["neighbor_embedding"],
            **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            neighbor_embedding=args["neighbor_embedding"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            neighbor_embedding=args["neighbor_embedding"],
            **shared_args,
        )
    elif args["model"] == "tensornet":
        from torchmdnet.models.tensornet import TensorNet
	# Setting is_equivariant to False to enforce the use of Scalar output module instead of EquivariantScalar
        is_equivariant = False
        representation_model = TensorNet(
	    equivariance_invariance_group=args["equivariance_invariance_group"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = create_prior_models(args)

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"],
        activation=args["activation"],
        reduce_op=args["reduce_op"],
        dtype=dtype,
    )
    head_model = SingleOutputHeadModel(output_model, prior_model, mean, std, dtype=dtype)
    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        head_model,
        derivative=args["derivative"],
        dtype=dtype,
    )
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    # The following are for backward compatibility with models created when atomref was
    # the only supported prior.
    if 'prior_model.initial_atomref' in state_dict:
        state_dict['prior_model.0.initial_atomref'] = state_dict['prior_model.initial_atomref']
        del state_dict['prior_model.initial_atomref']
    if 'prior_model.atomref.weight' in state_dict:
        state_dict['prior_model.0.atomref.weight'] = state_dict['prior_model.atomref.weight']
        del state_dict['prior_model.atomref.weight']
    # Add prefix "head_model." to keys in state_dict to match the new model structure for compatibility
    # with checkpoints created with older versions of TorchMD-Net.
    # List of main keys to look for
    keys_to_update = ["mean", "std", "output_model", "prior_model"]

    # For each main key, if any state_dict key starts with that main key,
    # prepend "head_model." to it
    for main_key in keys_to_update:
        subkeys = [k for k in state_dict if k.startswith(main_key)]
        for subkey in subkeys:
            state_dict["head_model." + subkey] = state_dict[subkey]
            del state_dict[subkey]
    model.load_state_dict(state_dict)
    return model.to(device)


def create_prior_models(args, dataset=None):
    """Parse the prior_model configuration option and create the prior models."""
    prior_models = []
    if args['prior_model']:
        prior_model = args['prior_model']
        prior_names = []
        prior_args = []
        if not isinstance(prior_model, list):
            prior_model = [prior_model]
        for prior in prior_model:
            if isinstance(prior, dict):
                for key, value in prior.items():
                    prior_names.append(key)
                    if value is None:
                        prior_args.append({})
                    else:
                        prior_args.append(value)
            else:
                prior_names.append(prior)
                prior_args.append({})
        if 'prior_args' in args:
            prior_args = args['prior_args']
            if not isinstance(prior_args, list):
                prior_args = [prior_args]
        for name, arg in zip(prior_names, prior_args):
            assert hasattr(priors, name), (
                f"Unknown prior model {name}. "
                f"Available models are {', '.join(priors.__all__)}"
            )
            # initialize the prior model
            prior_models.append(getattr(priors, name)(dataset=dataset, **arg))
    return prior_models


# class TorchMD_Net(nn.Module):
#     """The  TorchMD_Net class  combines a  given representation  model
#     (such as  the equivariant transformer),  an output model  (such as
#     the scalar output  module) and a prior model (such  as the atomref
#     prior), producing a  Module that takes as input a  series of atoms
#     features  and  outputs  a  scalar   value  (i.e  energy  for  each
#     batch/molecule) and,  derivative is True, the  negative of  its derivative
#     with respect to the positions (i.e forces for each atom).

#     """
#     def __init__(
#         self,
#         representation_model,
#         output_model,
#         prior_model=None,
#         mean=None,
#         std=None,
#         derivative=False,
#         dtype=torch.float32,
#     ):
#         super(TorchMD_Net, self).__init__()
#         self.representation_model = representation_model.to(dtype=dtype)
#         self.output_model = output_model.to(dtype=dtype)

#         if not output_model.allow_prior_model and prior_model is not None:
#             prior_model = None
#             rank_zero_warn(
#                 (
#                     "Prior model was given but the output model does "
#                     "not allow prior models. Dropping the prior model."
#                 )
#             )
#         if isinstance(prior_model, priors.base.BasePrior):
#             prior_model = [prior_model]
#         self.prior_model = None if prior_model is None else torch.nn.ModuleList(prior_model).to(dtype=dtype)

#         self.derivative = derivative

#         mean = torch.scalar_tensor(0) if mean is None else mean
#         self.register_buffer("mean", mean.to(dtype=dtype))
#         std = torch.scalar_tensor(1) if std is None else std
#         self.register_buffer("std", std.to(dtype=dtype))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.representation_model.reset_parameters()
#         self.output_model.reset_parameters()
#         if self.prior_model is not None:
#             for prior in self.prior_model:
#                 prior.reset_parameters()

#     def forward(
#         self,
#         z: Tensor,
#         pos: Tensor,
#         batch: Optional[Tensor] = None,
#         q: Optional[Tensor] = None,
#         s: Optional[Tensor] = None,
#         extra_args: Optional[Dict[str, Tensor]] = None
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         """Compute the output of the model.
#         Args:
#             z (Tensor): Atomic numbers of the atoms in the molecule. Shape (N,).
#             pos (Tensor): Atomic positions in the molecule. Shape (N, 3).
#             batch (Tensor, optional): Batch indices for the atoms in the molecule. Shape (N,).
#             q (Tensor, optional): Atomic charges in the molecule. Shape (N,).
#             s (Tensor, optional): Atomic spins in the molecule. Shape (N,).
#             extra_args (Dict[str, Tensor], optional): Extra arguments to pass to the prior model.
#         """

#         assert z.dim() == 1 and z.dtype == torch.long
#         batch = torch.zeros_like(z) if batch is None else batch

#         if self.derivative:
#             pos.requires_grad_(True)

#         # run the potentially wrapped representation model
#         x, v, z, pos, batch = self.representation_model(z, pos, batch, q=q, s=s)

#         # apply the output network
#         x = self.output_model.pre_reduce(x, v, z, pos, batch)

#         # scale by data standard deviation
#         if self.std is not None:
#             x = x * self.std

#         # apply atom-wise prior model
#         if self.prior_model is not None:
#             for prior in self.prior_model:
#                 x = prior.pre_reduce(x, z, pos, batch, extra_args)

#         # aggregate atoms
#         x = self.output_model.reduce(x, batch)

#         # shift by data mean
#         if self.mean is not None:
#             x = x + self.mean

#         # apply output model after reduction
#         y = self.output_model.post_reduce(x)

#         # apply molecular-wise prior model
#         if self.prior_model is not None:
#             for prior in self.prior_model:
#                 y = prior.post_reduce(y, z, pos, batch, extra_args)

#         # compute gradients with respect to coordinates
#         if self.derivative:
#             grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
#             dy = grad(
#                 [y],
#                 [pos],
#                 grad_outputs=grad_outputs,
#                 create_graph=True,
#                 retain_graph=True,
#             )[0]
#             if dy is None:
#                 raise RuntimeError("Autograd returned None for the force prediction.")

#             return y, -dy
#         # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
#         return y, None


class TorchMD_Net(nn.Module):
    """ This class is similar to TorchMD_Net, but it returns a list of tensors instead of a single tensor.
        The derivative of each tensor is returned as well based on a boolean derivative flag for each output.
        The model is constructed by combining a given representation model (such as the equivariant transformer) and a head model, which takes the per-atom features of the representation model and outputs a list of tensors with either per-atom or per-molecule features.

    """

    def __init__(
        self,
        representation_model,
        head_model,
        mean=None,
        std=None,
        derivative=False,
        dtype=torch.float32,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model.to(dtype=dtype)
        self.head_model = head_model.to(dtype=dtype)
        self.derivative = derivative
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.head_model.reset_parameters()

    def forward(self,
                z: Tensor,
                pos: Tensor,
                batch: Optional[Tensor] = None,
                q: Optional[Tensor] = None,
                s: Optional[Tensor] = None,
                extra_args: Optional[Dict[str, Tensor]] = None
                ) -> Union[Tuple[List[Tensor], List[Optional[Tensor]]], Tuple[Tensor, Optional[Tensor]]]:
        """Compute the output of the model.
        Args:
            z (Tensor): Atomic numbers of the atoms. Shape (N,).
            pos (Tensor): Atomic positions. Shape (N, 3).
            batch (Tensor, optional): Batch indices for the atoms (atoms in the same molecule have the same batch index). Shape (N,).
            q (Tensor, optional): Atomic charges in the molecule. Shape (N,).
            s (Tensor, optional): Atomic spins in the molecule. Shape (N,).
            extra_args (Dict[str, Tensor], optional): Extra arguments to pass to the prior model.
        """

        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative if isinstance(self.derivative, bool) else any(self.derivative):
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch, q=q, s=s)

        # apply the output network
        y = self.head_model(x, v, z, pos, batch, extra_args)
        derivative = torch.tensor([self.derivative]*len(y), device="cpu") if isinstance(self.derivative, bool) else torch.tensor(self.derivative)
        # compute gradients with respect to coordinates
        if torch.any(derivative):
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(yi) if derivative[i] else None for i, yi in enumerate(y)]
            dy = grad(
                y,
                [pos]*len(y),
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")

            neg_dys = [-dyi if derivative[i] else None for i, dyi in enumerate(dy)]
        else:
            neg_dys = [None]*len(y)
        # TODO: return only `out` once Union typing works with TorchScript
        if len(y) == 1:
            return y[0], neg_dys[0]
        return y, neg_dys


class SingleOutputHeadModel(nn.Module):
    """
    Takes the output of a representation model and returns a list of tensors with either per-atom or per-molecule features.
    """

    def __init__(self, output_model, prior_model=None, mean=None, std=None, dtype=torch.float32):
        super(SingleOutputHeadModel, self).__init__()

        self.output_model = output_model.to(dtype=dtype)

        if not output_model.allow_prior_model and prior_model is not None:
            prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )
        if isinstance(prior_model, priors.base.BasePrior):
            prior_model = [prior_model]
        self.prior_model = None if prior_model is None else torch.nn.ModuleList(prior_model).to(dtype=dtype)

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean.to(dtype=dtype))
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std.to(dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            for prior in self.prior_model:
                prior.reset_parameters()

    def forward(self, x, v, z, pos, batch, extra_args) -> List[Tensor]:
        """ Compute the output of the model.
        Args:
            x (Tensor): Number atomic features. Shape (N, C).
            v (Tensor): Vector atomic features. Shape (N, 3).
            z (Tensor): Atomic numbers of the atoms. Shape (N,).
            pos (Tensor): Atomic positions. Shape (N, 3).
            batch (Tensor): Batch indices for the atoms. Shape (N,).
        """

        # apply the output network, transforming atom-wise features into another set of per-atom features
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply atom-wise prior model to each output
        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        # aggregate atoms
        x = self.output_model.reduce(x, batch)

        # shift by data mean
        if self.mean is not None:
            x = x + self.mean

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        # apply molecular-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch)
        return [y]


class MultiOutputHeadModel(nn.Module):
    """
    Takes the output of a representation model and returns a list of tensors with per-atom and/or per-molecule features.
    """

    def __init__(self, output_model, prior_model=None, mean=None, std=None, dtype=torch.float32):
        super(MultiOutputHeadModel, self).__init__()

        self.output_model = output_model.to(dtype=dtype)

        if not output_model.allow_prior_model and prior_model is not None:
            prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )
        if isinstance(prior_model, priors.base.BasePrior):
            prior_model = [prior_model]
        self.prior_model = None if prior_model is None else torch.nn.ModuleList(prior_model).to(dtype=dtype)

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean.to(dtype=dtype))
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std.to(dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            for prior in self.prior_model:
                prior.reset_parameters()

    def forward(self, x, v, z, pos, batch, extra_args) -> List[Tensor]:
        """ Compute the output of the model.
        Args:
            x (Tensor): Number atomic features. Shape (N, C).
            v (Tensor): Vector atomic features. Shape (N, 3).
            z (Tensor): Atomic numbers of the atoms. Shape (N,).
            pos (Tensor): Atomic positions. Shape (N, 3).
            batch (Tensor): Batch indices for the atoms. Shape (N,).
        """

        # apply the output network, transforming atom-wise features into another set of per-atom features
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply atom-wise prior model to each output
        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        # aggregate atoms
        x = self.output_model.reduce(x, batch)

        # shift by data mean
        if self.mean is not None:
            x = x + self.mean

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        # apply molecular-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch)
        return [y,y,y]
