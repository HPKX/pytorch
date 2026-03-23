# torch API 41-45 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.fmax` | 浮点 Tensor-Tensor，`requires_grad=True`，普通 CPU/CUDA 路径 | `torch.fmax` -> `aten::fmax` -> `Autograd(VariableType::fmax)` -> `at::redispatch::fmax` -> CPU/CUDA structured wrapper（由 `fmax.out` codegen 生成，但不额外增加 runtime hop） -> `fmax_stub` -> `fmax_kernel` / `fmax_kernel_cuda` | `aten::fmax` |
| `torch.fmod` | Tensor-Tensor，普通 CPU/CUDA/MPS 路径 | `torch.fmod` -> `aten::fmod.Tensor` -> `Autograd(VariableType::fmod_Tensor)` -> `at::redispatch::fmod` -> structured wrapper（由 `fmod.Tensor_out` codegen 生成） -> `fmod_stub` -> `fmod_kernel` / `fmod_kernel_cuda` | `aten::fmod.Tensor` |
| `torch.fmod` | Tensor-Scalar，关注 overload 分流 | `torch.fmod` -> `aten::fmod.Scalar` -> `Autograd(VariableType::fmod_Scalar)` -> `CompositeExplicitAutograd(native::fmod)` -> `aten::fmod.Tensor`（`wrapped_scalar_tensor(other)` 后重入） -> structured wrapper（由 `fmod.Tensor_out` codegen 生成） -> `fmod_stub` -> kernel | `aten::fmod.Scalar`<br>`aten::fmod.Tensor` |
| `torch.gather` | 前向，普通 dense index，CPU/CUDA/MPS | `torch.gather` -> `aten::gather` -> `Autograd(VariableType::gather)` -> `at::redispatch::gather` -> structured wrapper（CPU/CUDA: `TORCH_IMPL_FUNC(gather_out)`；MPS: `gather_out_mps`） -> `gather_stub` -> `gather_cpu_kernel` / `gather_cuda_kernel` | `aten::gather` |
| `torch.gcd` | 整型 Tensor-Tensor，普通 CPU/CUDA 路径 | `torch.gcd` -> `aten::gcd` -> `AutogradNotImplementedFallback` -> CPU/CUDA structured wrapper（由 `gcd.out` codegen 生成） -> `gcd_stub` -> `gcd_kernel` / `gcd_kernel_cuda` | `aten::gcd` |
| `torch.ge` | Tensor-Tensor 比较，结果为 bool/不可微 | `torch.ge` -> `aten::ge.Tensor` -> `Autograd(VariableType::ge_Tensor，结果不可微，直接 redispatch)` -> `at::redispatch::ge` -> CPU/CUDA structured wrapper（由 `ge.Tensor_out` codegen 生成） -> `ge_stub` -> `ge_kernel` / `ge_kernel_cuda` | `aten::ge.Tensor` |

## 备注

- `torch.ge` 的 `Scalar` overload 也存在，但主链与 `Tensor` overload 基本一致，只是入口 schema 变为 `aten::ge.Scalar` / `aten::ge.Scalar_out`。
- `torch.gather` 在部分 expanded-index 优化场景会走 `gather_expanded_index_stub`，表中写的是更常见的普通 `gather_stub` 主路径。
