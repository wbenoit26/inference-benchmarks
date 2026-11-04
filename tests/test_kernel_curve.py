import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from bench.kernel_curve import graphed  # noqa: E402


def test_graphed_model_uses_each_new_input():
    module = torch.nn.Conv1d(2, 4, 3).cuda()
    x = torch.randn(3, 2, 3072, device="cuda")
    with torch.no_grad():
        model = graphed(module, 3)
        model(torch.zeros_like(x))
        torch.testing.assert_close(model(x), module(x))
