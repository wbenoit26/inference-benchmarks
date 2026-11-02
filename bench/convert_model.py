import sys

import torch
from architectures.supervised import SupervisedTimeDomainResNet
from ml4gw.nn.norm import GroupNorm1DGetter

KERNEL_SHAPE = (2, 3072)


def main() -> None:
    """Export the o4c model from .pt to .pt2

    python -m bench.convert_model <aframe.pt> <model_exported.pt2>
    """
    src, dest = sys.argv[1:]
    scripted = torch.jit.load(src, map_location="cpu")
    model = SupervisedTimeDomainResNet(
        num_ifos=2,
        sample_rate=2048,
        kernel_length=1.5,
        layers=[3, 4, 6, 3],
        norm_layer=GroupNorm1DGetter(16),
    )
    model.load_state_dict(scripted.state_dict(), strict=True)
    model.eval()
    exported = torch.export.export(
        model,
        (torch.randn(8, *KERNEL_SHAPE),),
        dynamic_shapes=({0: torch.export.Dim.AUTO},),
    )
    torch.export.save(exported, dest)


if __name__ == "__main__":
    main()
