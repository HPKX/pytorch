# torch API 11-15 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.atleast_3d` | dim == 0 | `aten::atleast_3d` → CompositeImplicitAutograd → `at::native::atleast_3d` → 真实运行时 hop `aten::reshape({1,1,1})` | `aten::atleast_3d`, `aten::reshape` |
| `torch.atleast_3d` | dim == 1 | `aten::atleast_3d` → CompositeImplicitAutograd → `at::native::atleast_3d` → 真实运行时 hop `aten::unsqueeze(0)` → `aten::unsqueeze(-1)` | `aten::atleast_3d`, `aten::unsqueeze` |
| `torch.atleast_3d` | dim == 2 | `aten::atleast_3d` → CompositeImplicitAutograd → `at::native::atleast_3d` → 真实运行时 hop `aten::unsqueeze(-1)` | `aten::atleast_3d`, `aten::unsqueeze` |
| `torch.atleast_3d` | dim >= 3 | `aten::atleast_3d` → CompositeImplicitAutograd → `at::native::atleast_3d` → 返回 self (alias) | `aten::atleast_3d` |
| `torch.bartlett_window` | 默认 (periodic=true) | `aten::bartlett_window` → CompositeExplicitAutograd → `at::native::bartlett_window(periodic=true)`；`window_length==0/1` 时分别走 `aten::empty` / `aten::ones`，一般情形走 `aten::arange -> aten::mul_ -> aten::narrow -> aten::add_` | `aten::bartlett_window`, `aten::empty`, `aten::ones`, `aten::arange`, `aten::mul_`, `aten::narrow`, `aten::add_` |
| `torch.bartlett_window` | periodic=true, window_length>=2 | `aten::bartlett_window.periodic` → CompositeExplicitAutograd → `at::native::bartlett_window` → `aten::arange` → `aten::mul_` → `aten::narrow` → `aten::mul_(-1)` → `aten::add_(2)` → `aten::narrow` 返回 | `aten::bartlett_window.periodic`, `aten::arange`, `aten::mul_`, `aten::narrow`, `aten::add_` |
| `torch.bartlett_window` | window_length==0 | `aten::bartlett_window` → CompositeExplicitAutograd → `aten::empty({0})` | `aten::bartlett_window`, `aten::empty` |
| `torch.bartlett_window` | window_length==1 | `aten::bartlett_window` → CompositeExplicitAutograd → `aten::ones({1})` | `aten::bartlett_window`, `aten::ones` |
| `torch.bitwise_left_shift` | Tensor+Tensor, CPU | `aten::bitwise_left_shift.Tensor` → CPU key → `wrapper_CPU_bitwise_left_shift` (structured) → `op.meta()` + `op.impl()` → `bitwise_left_shift_out` → `lshift_stub` → `lshift_kernel` | `aten::bitwise_left_shift.Tensor` |
| `torch.bitwise_left_shift` | Tensor+Tensor, CUDA | `aten::bitwise_left_shift.Tensor` → CUDA key → `wrapper_CUDA_bitwise_left_shift` (structured) → `op.meta()` + `op.impl()` → `bitwise_left_shift_out` → `lshift_stub` → `lshift_kernel_cuda` | `aten::bitwise_left_shift.Tensor` |
| `torch.bitwise_left_shift` | Tensor+Scalar | `aten::bitwise_left_shift.Tensor_Scalar` → CompositeExplicitAutograd → `at::native::bitwise_left_shift` → `lshift_stub` | `aten::bitwise_left_shift.Tensor_Scalar` |
| `torch.bitwise_left_shift` | Scalar+Tensor | `aten::bitwise_left_shift.Scalar_Tensor` → CompositeExplicitAutograd → `at::native::bitwise_left_shift` → `lshift_stub` | `aten::bitwise_left_shift.Scalar_Tensor` |
| `torch.bitwise_right_shift` | Tensor+Tensor, CPU | `aten::bitwise_right_shift.Tensor` → CPU key → `wrapper_CPU_bitwise_right_shift` (structured) → `op.meta()` + `op.impl()` → `bitwise_right_shift_out` → `rshift_stub` → `rshift_kernel` | `aten::bitwise_right_shift.Tensor` |
| `torch.bitwise_right_shift` | Tensor+Tensor, CUDA | `aten::bitwise_right_shift.Tensor` → CUDA key → `wrapper_CUDA_bitwise_right_shift` (structured) → `op.meta()` + `op.impl()` → `bitwise_right_shift_out` → `rshift_stub` → `rshift_kernel_cuda` | `aten::bitwise_right_shift.Tensor` |
| `torch.bitwise_right_shift` | Tensor+Scalar | `aten::bitwise_right_shift.Tensor_Scalar` → CompositeExplicitAutograd → `at::native::bitwise_right_shift` → `rshift_stub` | `aten::bitwise_right_shift.Tensor_Scalar` |
| `torch.bitwise_right_shift` | Scalar+Tensor | `aten::bitwise_right_shift.Scalar_Tensor` → CompositeExplicitAutograd → `at::native::bitwise_right_shift` → `rshift_stub` | `aten::bitwise_right_shift.Scalar_Tensor` |
| `torch.blackman_window` | 默认 (periodic=true) | `aten::blackman_window` → CompositeExplicitAutograd → `at::native::blackman_window(periodic=true)`；`window_length==0/1` 时分别走 `aten::empty` / `aten::ones`，一般情形走 `aten::arange -> aten::mul -> aten::mul_ -> aten::cos_ -> aten::narrow` | `aten::blackman_window`, `aten::empty`, `aten::ones`, `aten::arange`, `aten::mul`, `aten::mul_`, `aten::cos_`, `aten::narrow` |
| `torch.blackman_window` | periodic=true, window_length>=2 | `aten::blackman_window.periodic` → CompositeExplicitAutograd → `at::native::blackman_window` → `aten::arange` → `aten::mul_(π/N)` → `aten::mul(4) -> aten::cos_ -> aten::mul_(0.08)` − `aten::mul(2) -> aten::cos_ -> aten::mul_(0.5)` + 0.42 → `aten::narrow` 返回 | `aten::blackman_window.periodic`, `aten::arange`, `aten::mul_`, `aten::mul`, `aten::cos_`, `aten::narrow` |
| `torch.blackman_window` | window_length==0 | `aten::blackman_window` → CompositeExplicitAutograd → `aten::empty({0})` | `aten::blackman_window`, `aten::empty` |
| `torch.blackman_window` | window_length==1 | `aten::blackman_window` → CompositeExplicitAutograd → `aten::ones({1})` | `aten::blackman_window`, `aten::ones` |

**备注:**
- `torch.atleast_3d`: CompositeImplicitAutograd，无 derivatives.yaml 条目，autograd 穿透至 reshape/unsqueeze
- `torch.bartlett_window` / `torch.blackman_window`: CompositeExplicitAutograd 工厂函数，autograd 为 `autogradNotImplementedFallback`（无输入 Tensor）
- `torch.bitwise_left_shift` / `torch.bitwise_right_shift`: Tensor+Tensor 重载为 structured op (structured_delegate → .Tensor_out)；Tensor+Scalar 和 Scalar+Tensor 重载为 CompositeExplicitAutograd。所有重载 autograd 均为 `autogradNotImplementedFallback`（整数位运算不可导）
