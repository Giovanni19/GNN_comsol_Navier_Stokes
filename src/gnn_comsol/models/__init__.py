"""
Model registry: every architecture is built through build_model.

The BSMS model receives its multiscale hierarchy at forward time, so one
model can run on different meshes.
"""

from .gcn import GCNNet
from .virtual_node import GCNVirtualNodeNet
from .bsms_pressure import PressureBSMSGNN
from .meshgraphnet import MeshGraphNetModel
import torch

ARCHITECTURES = {
    "gcn": GCNNet,
    "gcn_virtual_node": GCNVirtualNodeNet,
    "meshgraphnet": MeshGraphNetModel,
    "bsms": PressureBSMSGNN,
}

def initialize_model(
    model,
    initialization="default",
    num_out=None,
):
    """
    Optional initialization applied after model construction.

    default:
        Keep the standard NVIDIA/PyTorch initialization.

    zero_output:
        Keep the standard initialization everywhere and zero only
        the final Linear layer producing the network output.
    """

    if initialization == "default":
        return

    if initialization == "zero_output":

        if num_out is None:
            raise ValueError(
                "zero_output initialization requires num_out."
            )

        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if (
                isinstance(module, torch.nn.Linear)
                and module.out_features == num_out
            )
        ]

        print(
            f"Candidate output layers "
            f"(out_features={num_out}):"
        )

        for name, layer in candidates:
            print(
                f"  {name}: "
                f"{layer.in_features} -> {layer.out_features}"
            )

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one Linear layer with "
                f"out_features={num_out}, but found "
                f"{len(candidates)}: "
                f"{[name for name, _ in candidates]}"
            )

        output_name, output_layer = candidates[0]

        with torch.no_grad():
            output_layer.weight.zero_()

            if output_layer.bias is not None:
                output_layer.bias.zero_()

        print(
            "Initialization: output layer set to zero | "
            f"layer={output_name} | "
            f"shape={tuple(output_layer.weight.shape)}"
        )

        return

    raise ValueError(
        f"Unknown initialization: {initialization!r}. "
        "Expected 'default' or 'zero_output'."
    )
def report_initialization(model, name):
    total = sum(p.numel() for p in model.parameters())

    nonzero = sum(
        torch.count_nonzero(p.detach()).item()
        for p in model.parameters()
    )

    max_abs = max(
        p.detach().abs().max().item()
        for p in model.parameters()
    )

    print(
        f"{name} initialization | "
        f"parameters={total:,} | "
        f"nonzero={nonzero:,} | "
        f"max|parameter|={max_abs:.6e}"
    )
def build_model(
    architecture,
    num_in,
    num_out,
    num_neurons,
    num_layers,
    dropout,
    *,
    pos_dim=2,
    processor_size=10,
    edge_dim=3,
    num_mesh_levels=2,
    num_layers_bistride=2,
    bistride_unet_levels=1,
    num_layers_node_processor=2,
    num_layers_edge_processor=2,
    num_layers_node_encoder=2,
    num_layers_edge_encoder=2,
    num_layers_node_decoder=2,
    aggregation="sum",
):
    """
    Instantiate a network by the name used in the config files.
    """

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}."
        )

    # NVIDIA BiStride MeshGraphNet

    if architecture == "bsms":

        return PressureBSMSGNN(
            num_in=num_in,
            num_out=num_out,
            edge_dim=edge_dim,
            hidden_dim=num_neurons,
            processor_size=processor_size,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            num_mesh_levels=num_mesh_levels,
            num_layers_bistride=num_layers_bistride,
            bistride_unet_levels=bistride_unet_levels,
            pos_dim=pos_dim,
            num_layers_node_encoder=num_layers_node_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
        )

    # NVIDIA MeshGraphNet

    if architecture == "meshgraphnet":

        return MeshGraphNetModel(
            num_in=num_in,
            num_out=num_out,
            edge_dim=edge_dim,
            hidden_dim=num_neurons,
            processor_size=processor_size,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            num_layers_node_encoder=num_layers_node_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
        )

    # Standard models

    return ARCHITECTURES[architecture](
        num_in=num_in,
        num_out=num_out,
        num_neurons=num_neurons,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = [
    "ARCHITECTURES",
    "GCNNet",
    "GCNVirtualNodeNet",
    "MeshGraphNetModel",
    "PressureBSMSGNN",
    "build_model",
]
