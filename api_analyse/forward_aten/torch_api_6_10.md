# torch API 6-10 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.aminmax`    | no_grad, CPU        | `aten::aminmax` → CPU key → `wrapper_CPU_aminmax` (structured) → `op.meta()` + `op.impl()` → `aminmax_out` → `aminmax_stub` → `aminmax_kernel`        | `aten::aminmax`  |
| `torch.aminmax`    | no_grad, CUDA       | `aten::aminmax` → CUDA key → `wrapper_CUDA_aminmax` (structured) → `op.meta()` + `op.impl()` → `aminmax_out` → `aminmax_stub` → `aminmax_kernel_impl` | `aten::aminmax`  |
| `torch.aminmax`    | requires_grad       | `aten::aminmax` → Autograd key → `autogradNotImplementedFallback` → redispatch → (同 no_grad 路径)                                                       | `aten::aminmax`  |
| `torch.angle`      | no_grad, CPU        | `aten::angle` → CPU key → `wrapper_CPU__angle` → `at::native::angle` → `angle_stub` → `angle_kernel`                                                  | `aten::angle`                         |
| `torch.angle`      | no_grad, CUDA       | `aten::angle` → CUDA key → `wrapper_CUDA__angle` → `at::native::angle` → `angle_stub` → `angle_kernel_cuda`                                           | `aten::angle`                         |
| `torch.angle`      | requires_grad, CPU  | `aten::angle` → AutogradCPU → `AngleBackward0`(save self) → redispatch → CPU key → `wrapper_CPU__angle` → `angle_stub` → `angle_kernel`               | `aten::angle`                         |
| `torch.angle`      | requires_grad, CUDA | `aten::angle` → AutogradCUDA → `AngleBackward0`(save self) → redispatch → CUDA key → `wrapper_CUDA__angle` → `angle_stub` → `angle_kernel_cuda`       | `aten::angle`                         |
| `torch.angle`      | SparseCsr, CPU      | `aten::angle` → SparseCsrCPU key → `wrapper_SparseCsrCPU__angle` → `angle_sparse_csr`                                                                 | `aten::angle`                         |
| `torch.argwhere`   | 任意 backend          | `aten::argwhere` → CompositeImplicitAutograd → `at::native::argwhere` → `self.nonzero()` → `aten::nonzero` → backend key → backend kernel             | `aten::argwhere`, `aten::nonzero`     |
| `torch.atleast_1d` | dim == 0            | `aten::atleast_1d` → CompositeImplicitAutograd → `at::native::atleast_1d` → `self.reshape({1})` → `aten::reshape` → backend                           | `aten::atleast_1d`, `aten::reshape`   |
| `torch.atleast_1d` | dim >= 1            | `aten::atleast_1d` → CompositeImplicitAutograd → `at::native::atleast_1d` → 返回 self (alias)                                                           | `aten::atleast_1d`                    |
| `torch.atleast_2d` | dim == 0            | `aten::atleast_2d` → CompositeImplicitAutograd → `at::native::atleast_2d` → `self.reshape({1,1})` → `aten::reshape` → backend                         | `aten::atleast_2d`, `aten::reshape`   |
| `torch.atleast_2d` | dim == 1            | `aten::atleast_2d` → CompositeImplicitAutograd → `at::native::atleast_2d` → `self.unsqueeze(0)` → `aten::unsqueeze` → backend                         | `aten::atleast_2d`, `aten::unsqueeze` |
| `torch.atleast_2d` | dim >= 2            | `aten::atleast_2d` → CompositeImplicitAutograd → `at::native::atleast_2d` → 返回 self (alias)                                                           | `aten::atleast_2d`                    |
