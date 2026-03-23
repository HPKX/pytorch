# torch API 176-180 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.ormqr` | CPU/CUDA 稠密输入，`left=True`，`transpose=False`，`requires_grad=True` | `torch.ormqr` -> `aten::ormqr` -> `AutogradCPU/CUDA` -> `CPU/CUDA wrapper` -> `at::native::ormqr` -> `aten::empty` -> 原生 `ormqr_out` helper 做 `aten::resize_as_ -> aten::transpose_ -> aten::copy_` 的尺寸/转置/拷贝整理 -> `ormqr_stub` -> CPU LAPACK / CUDA cuSOLVER kernel | `aten::ormqr`, `aten::empty`, `aten::resize_as_`, `aten::transpose_`, `aten::copy_` |
| `torch.pdist` | 2D 浮点输入，`p=2`，前向 | `torch.pdist` -> `aten::pdist` (`CompositeImplicitAutograd`) -> `at::native::pdist` -> `aten::contiguous` -> `aten::_pdist_forward` -> `AutogradCPU/CUDA` -> `CPU/CUDA wrapper` -> `at::native::_pdist_forward` -> `aten::empty` -> `pdist_forward_stub` -> CPU/CUDA kernel | `aten::pdist`, `aten::contiguous`, `aten::_pdist_forward`, `aten::empty` |
| `torch.poisson` | 浮点输入，带/不带 `Generator` | `torch.poisson` -> `aten::poisson` -> `AutogradCPU/CUDA` -> CPU: `at::native::_s_poisson_cpu` -> `aten::zeros` -> `TensorIterator` -> `cpu_serial_kernel(sample_poisson)`；CUDA: `at::native::_s_poisson_cuda` -> `aten::empty` -> `launch_poisson_cuda_kernel` | `aten::poisson`, `aten::zeros`, `aten::empty` |
| `torch.polygamma` | `n=0` 或 `n=1`，`requires_grad=True` | `torch.polygamma` -> `aten::polygamma` -> `AutogradCPU/CUDA` -> 结构化 `CPU/CUDA wrapper`（`meta + impl`）-> `polygamma_stub` -> `n=0` 走 digamma kernel / `n=1` 走 trigamma kernel | `aten::polygamma` |
| `torch.polygamma` | `n>1` 一般情形 | `torch.polygamma` -> `aten::polygamma` -> `AutogradCPU/CUDA` -> 结构化 `CPU/CUDA wrapper`（`meta + impl`）-> `polygamma_stub` -> CPU `cpu_kernel(calc_polygamma)` / CUDA `gpu_kernel` 或 Jiterator kernel | `aten::polygamma` |
| `torch.positive` | 非 `bool` Tensor | `torch.positive` -> `aten::positive` (`CompositeImplicitAutograd`) -> `at::native::positive` -> 直接返回 `self`（alias/identity，无后端 kernel） | `aten::positive` |

## 备注

- `pdist` 的真实后端入口是 `aten::_pdist_forward`；`aten::pdist` 只是 composite 前端包装，并先做 `contiguous()`.
- `polygamma` 是 structured 算子；functional `aten::polygamma` 直接走生成的 `meta + impl` wrapper，不会在运行时再额外经一次 `aten::polygamma.out` dispatcher hop。
- `poisson` 有 Autograd 包装，但导数按 `zeros_like(self)` 处理；`positive` 本质是带类型检查的 identity，`bool` 输入会直接报错。
