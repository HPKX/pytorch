# torch API 181-185 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.rad2deg` | dense Tensor（`out=None`；整数输入会升到默认浮点；`requires_grad` 时先过 Autograd） | `torch.rad2deg` / `Tensor.rad2deg` -> `aten::rad2deg` -> `Autograd{backend}`（若需 grad） -> `CompositeExplicitAutograd(rad2deg)` -> `at::native::rad2deg` -> `aten::empty_like` -> `aten::rad2deg.out` -> `CompositeExplicitAutograd(rad2deg.out)` -> `at::native::rad2deg_out` -> `aten::mul.out` -> TensorIterator backend kernel | `aten::rad2deg`、`aten::empty_like`、`aten::rad2deg.out`、`aten::mul.out` |
| `torch.range` | `out=None`（手写 Python wrapper；先发弃用警告） | `torch.range` manual wrapper -> `aten::range.step` -> `BackendSelect` -> `CompositeExplicitAutograd(range.step)` -> `at::native::range` -> `aten::empty` -> `aten::range.out` -> `CPU/CUDA/MPS` backend kernel（`range_out` / `range_cuda_out` / `range_mps_out`） | `aten::range.step`、`aten::empty`、`aten::range.out` |
| `torch.range` | `out=` 已给定（device / dtype 由 `out` 锁定） | `torch.range` manual wrapper -> `aten::range.out` -> `CPU/CUDA/MPS` backend kernel（`range_out` / `range_cuda_out` / `range_mps_out`） | `aten::range.out` |
| `torch.renorm` | dense Tensor（CPU/CUDA 主路径；`requires_grad` 有专用 backward） | `torch.renorm` / `Tensor.renorm` -> `aten::renorm` -> `Autograd{backend}`（若需 grad） -> CompositeExplicitAutogradNonFunctional wrapper 做 `meta`，随后调用 `aten::renorm.out` -> `TORCH_IMPL_FUNC(renorm_out)`：`aten::linalg_vector_norm` -> 必要时 `aten::empty` 分配 `factor` -> `renorm_scale_factor_stub` -> `aten::mul.out` -> backend kernel | `aten::renorm`、`aten::renorm.out`、`aten::linalg_vector_norm`、`aten::empty`、`aten::mul.out` |
| `torch.rot90` | `k % 4 == 1` 或 `3`（90° / 270°） | `torch.rot90` / `Tensor.rot90` -> `aten::rot90` -> `Autograd{backend}`（若需 grad） -> `CompositeExplicitAutograd(rot90)` -> `at::native::rot90` -> `aten::flip` -> backend flip kernel -> `aten::transpose_`（view） | `aten::rot90`、`aten::flip`、`aten::transpose_` |
| `torch.rot90` | `k % 4 == 2`（180°） | `torch.rot90` / `Tensor.rot90` -> `aten::rot90` -> `Autograd{backend}`（若需 grad） -> `CompositeExplicitAutograd(rot90)` -> `at::native::rot90` -> `aten::flip` -> backend flip kernel | `aten::rot90`、`aten::flip` |
| `torch.rot90` | `k % 4 == 0`（等价不旋转） | `torch.rot90` / `Tensor.rot90` -> `aten::rot90` -> `Autograd{backend}`（若需 grad） -> `CompositeExplicitAutograd(rot90)` -> `at::native::rot90` -> `aten::clone` -> backend copy kernel | `aten::rot90`、`aten::clone` |
| `torch.row_stack` | 1D tensors 常见路径（0D 会改走 `reshape`；梯度主要由内部 view / `cat` 记录） | `torch.row_stack` -> `aten::row_stack` -> `CompositeImplicitAutograd(row_stack)` -> `at::native::row_stack` -> `aten::vstack` -> `CompositeImplicitAutograd(vstack)` -> `at::native::vstack` -> `aten::atleast_2d.Sequence` -> 每个元素走 `aten::unsqueeze`（0D 时为 `aten::reshape`） -> `aten::cat` -> `Autograd{backend}`（若需 grad） -> backend cat kernel | `aten::row_stack`、`aten::vstack`、`aten::atleast_2d.Sequence`、`aten::unsqueeze`、`aten::reshape`、`aten::cat` |

## 备注

- `torch.range` 的前端在 `python_torch_functions_manual.cpp`，不是常规生成 binding；`requires_grad=True` 是结果返回后再 `.set_requires_grad(...)`，不是 dispatcher 上的 autograd 记录。
- `renorm` 的 `structured_delegate: renorm.out` 只是 codegen 指令，不是 runtime 先调一次 `aten::renorm.out`；runtime 是 backend structured wrapper 直接执行 `meta + impl`。
- `row_stack` 是 `vstack` 别名，本身没有单独的 `VariableType` autograd 包装；梯度主要沿内部 `unsqueeze` / `reshape` 和 `aten::cat` 传播。
