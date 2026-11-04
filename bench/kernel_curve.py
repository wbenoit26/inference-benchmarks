import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch._inductor import codecache  # noqa: F401

SIZES = [
    8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    1536, 2048, 3072, 4096,
]  # fmt: skip
SHAPE = (2, 3072)
REPS = 5


def seconds_per_kernel(
    model: Callable[[torch.Tensor], torch.Tensor], inputs: list[torch.Tensor]
) -> float:
    """Time per kernel of passing each input through `model`"""
    num_kernels = len(inputs) * inputs[0].shape[0]

    # Warm-up
    model(torch.zeros_like(inputs[0]))
    torch.cuda.synchronize()

    start = time.perf_counter()
    for input in inputs:
        model(input)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_kernels


def fp16(module: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap the NN so that it runs with fp16 autocast"""

    def model(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            return module(x)

    return model


def graphed(
    module: torch.nn.Module, size: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Capture one call of the NN at a fixed size in a CUDA graph.

    See: https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs
    """
    static = torch.zeros(size, *SHAPE, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(static)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = module(static)

    def model(x: torch.Tensor) -> torch.Tensor:
        static.copy_(x)
        graph.replay()
        return out

    return model


def compiled(
    module: torch.nn.Module, size: int, mode: str
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float]:
    """Compile the NN for one fixed size and time the compilation"""
    model = torch.compile(module, mode=mode, dynamic=False)

    # Compiling happens on the first call
    x = torch.randn(size, *SHAPE, device="cuda")
    start = time.perf_counter()
    model(x)
    torch.cuda.synchronize()

    return model, time.perf_counter() - start


def main() -> None:
    """Time every variant at every size and write the curves for this GPU."""
    torch.set_grad_enabled(False)

    props = torch.cuda.get_device_properties(0)
    name = props.name.removeprefix("NVIDIA ").replace(" ", "-")
    l2_mib = props.L2_cache_size / 1024**2
    module = torch.export.load("train/model_exported.pt2").module().cuda()
    fp16_module = fp16(module)
    aoti = torch._inductor.aoti_load_package(f"gpus/{name}/model_aoti.pt2")

    compile_modes = ["default", "reduce-overhead"]
    variants = {
        "export": module,
        "aoti": aoti,
        "export_fp16": fp16_module,
        "export_cudagraph": None,
    }
    variants |= {f"compile_{mode}": None for mode in compile_modes}

    x = torch.randn(REPS * SIZES[-1], *SHAPE, device="cuda")
    curves = {var: {size: {} for size in SIZES} for var in variants}
    compile_seconds = {
        "default": {},
        "reduce-overhead": {},
    }

    for size in SIZES:
        inputs = list(x[: len(x) // size * size].split(size))
        reference = module(inputs[0])

        for mode in compile_modes:
            variants[f"compile_{mode}"], compile_seconds[mode][size] = (
                compiled(module, size, mode)
            )
        variants["export_cudagraph"] = graphed(module, size)

        # Clear what earlier sizes compiled
        torch._dynamo.reset()
        for variant, model in variants.items():
            curves[variant][size]["ms_per_kernel"] = (
                1_000 * seconds_per_kernel(model, inputs)
            )
            comparison = model(inputs[0])
            curves[variant][size]["max_abs_diff"] = float(
                (comparison - reference).abs().max()
            )

    result = {
        "gpu": props.name,
        "l2_mib": l2_mib,
        "curves": curves,
        "compile_seconds": compile_seconds,
    }
    out = Path("gpus") / name / "kernel_curve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
