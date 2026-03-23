# torch API 186-190 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.select_scatter` | 稠密张量，`self`/`src` 参与反向传播 | `torch.select_scatter` -> `aten::select_scatter` -> `Autograd`(`SelectScatterBackward0`) -> redispatch 同一 `SymInt` overload -> `CompositeExplicitAutogradNonFunctional` -> `native::select_scatter_symint`(`clone_preserve_strides` -> `aten::select.int` -> `aten::copy_`)` | `aten::select_scatter`<br>`aten::select.int`<br>`aten::copy_` |
| `torch.sgn` | 稠密实数/复数；复数且 `requires_grad=True` 最有代表性 | `torch.sgn` -> `aten::sgn` -> `[requires_grad] AutogradCPU/CUDA`(`SgnBackward0`，保存 `self/result`) -> CPU/CUDA structured wrapper(`meta` + `impl`) -> `native::sgn_out` -> `{实数: sign_stub -> sign_kernel_cpu/cuda；复数: sgn_stub -> sgn_kernel_cpu/cuda}` | `aten::sgn` |
| `torch.signbit` | 稠密浮点/整型输入；`bool` 输入有专门分支 | `torch.signbit` -> `aten::signbit` -> `[若输入 requires_grad，Autograd wrapper 仅 redispatch，不建 grad_fn]` -> CPU/CUDA structured wrapper(`meta` + `impl`) -> `native::signbit_out` -> `{bool: aten::fill_.Scalar(false)；其他: signbit_stub -> signbit_kernel_cpu/cuda}` | `aten::signbit`<br>`aten::fill_.Scalar` |
| `torch.slogdet` | 稠密方阵；`requires_grad=True` 时最能体现真实 autograd 入口 | `torch.slogdet` -> `aten::slogdet` -> `CompositeImplicitAutograd`(`native::slogdet`) -> `aten::linalg_slogdet` -> `CompositeImplicitAutograd`(`native::linalg_slogdet`) -> `aten::_linalg_slogdet` -> `[requires_grad] AutogradCPU/CUDA/MPS`(`LinalgSlogdetBackward0`，保存 `A/LU/pivots/sign`) -> CPU/CUDA/MPS structured wrapper(`meta` + `impl`) -> `native::_linalg_slogdet_out` -> `aten::linalg_lu_factor_ex.out` + `aten::sgn` + `aten::prod` + `aten::mul.out` + `aten::abs` + `aten::log_` + `aten::sum.IntList_out` | `aten::slogdet`<br>`aten::linalg_slogdet`<br>`aten::_linalg_slogdet`<br>`aten::linalg_lu_factor_ex.out`<br>`aten::sgn`<br>`aten::prod`<br>`aten::mul.out`<br>`aten::abs`<br>`aten::log_`<br>`aten::sum.IntList_out` |
| `torch.special.bessel_j0` | 稠密浮点张量；即使 `input.requires_grad=True` 结果也标记为不可导 | `torch.special.bessel_j0` -> `aten::special_bessel_j0` -> `[Autograd key 若存在，仅 redispatch，不建 grad_fn]` -> CPU/CUDA/MPS structured wrapper(`meta` + `impl`) -> `native::special_bessel_j0_out` -> `special_bessel_j0_stub` -> `{CPU: bessel_j0_kernel；CUDA: bessel_j0_kernel_cuda；MPS: bessel_j0_kernel_mps}` | `aten::special_bessel_j0` |

## 备注

- `torch.slogdet` 文档上是 `torch.linalg.slogdet` 的别名，但 generated Python binding 的首个 dispatcher 入口仍是 `aten::slogdet`；真正承载求导逻辑的是 `aten::_linalg_slogdet`。
- `sgn`、`signbit`、`special_bessel_j0` 都是 structured 路径；表中列出 `*.out` 是对应的 structured ATen 接口，不表示 dense CPU/CUDA 一定会先额外 dispatch 一次 `aten::*.out`。
