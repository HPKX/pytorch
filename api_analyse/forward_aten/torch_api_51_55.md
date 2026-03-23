# torch API 51-55 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.hsplit` | int 重载；按 sections 水平切分张量 | `torch.hsplit` → `aten::hsplit.int` → `CompositeImplicitAutograd` → `at::native::hsplit` → `aten::tensor_split.sections`（dim=0 for 1-D, dim=1 otherwise）→ `CompositeImplicitAutograd` → `tensor_split_sections_symint` → 循环调用 `at::slice_symint` → `aten::slice.Tensor` | `aten::hsplit.int`, `aten::tensor_split.sections`, `aten::slice.Tensor` |
| `torch.hsplit` | array 重载；按 indices 水平切分张量 | `torch.hsplit` → `aten::hsplit.array` → `CompositeImplicitAutograd` → `at::native::hsplit` → `aten::tensor_split.indices`（dim=0 for 1-D, dim=1 otherwise）→ `CompositeImplicitAutograd` → `tensor_split_indices_symint` → 循环调用 `at::slice_symint` → `aten::slice.Tensor` | `aten::hsplit.array`, `aten::tensor_split.indices`, `aten::slice.Tensor` |
| `torch.hstack` | 水平拼接张量列表 | `torch.hstack` → `aten::hstack` → `CompositeImplicitAutograd` → `at::native::hstack` → `aten::atleast_1d.Sequence` → `CompositeImplicitAutograd` → `at::native::atleast_1d` → 若结果 dim==1 则 `aten::cat(..., 0)`，否则 `aten::cat(..., 1)` → `CPU/CUDA/MPS` structured `cat` wrapper（`precompute + op.meta() + op.impl()`） | `aten::hstack`, `aten::atleast_1d.Sequence`, `aten::cat` |
| `torch.hstack` | `out=` 变体；水平拼接 | `aten::hstack.out` → `CompositeImplicitAutograd` → `at::native::hstack_out` → `aten::atleast_1d.Sequence` → `CompositeImplicitAutograd` → `at::native::atleast_1d` → 若结果 dim==1 则 `aten::cat.out(..., 0)`，否则 `aten::cat.out(..., 1)` → `CPU/CUDA/MPS` structured `cat.out` wrapper（`precompute + op.meta() + op.impl()`） | `aten::hstack.out`, `aten::atleast_1d.Sequence`, `aten::cat.out` |
| `torch.hypot` | 计算直角三角形斜边长度 | `torch.hypot` → `aten::hypot` → `CPU/CUDA/MPS` structured wrapper（`op.meta() + op.impl()`）→ `TORCH_IMPL_FUNC(hypot_out)` → `hypot_stub` → `hypot_kernel` / `hypot_kernel_cuda` / `hypot_mps_kernel` | `aten::hypot` |
| `torch.hypot` | Autograd 反向传播 | `torch.hypot` → `aten::hypot` → `AutogradCPU/CUDA` 记录梯度 -> redispatch -> `CPU/CUDA/MPS` structured wrapper；反向公式来自 `derivatives.yaml`：`grad_self = grad * self / result`, `grad_other = grad * other / result` | `aten::hypot` |
| `torch.i0` | 第一类零阶修正贝塞尔函数 | `torch.i0` → `aten::i0` → `CPU/CUDA/MPS` structured wrapper（`op.meta() + op.impl()`）→ `TORCH_IMPL_FUNC(i0_out)` → `i0_stub` → `i0_kernel` / `i0_kernel_cuda` / `i0_kernel_mps` | `aten::i0` |
| `torch.i0` | Autograd 反向传播 | `torch.i0` → `aten::i0` → `AutogradCPU/CUDA` 记录梯度 -> redispatch -> `CPU/CUDA/MPS` structured wrapper；反向公式来自 `derivatives.yaml`：`grad_self = grad * aten::special_i1(self)` | `aten::i0`, `aten::special_i1` |
| `torch.igamma` | 正则化下不完全伽玛函数 | `torch.igamma` → `aten::igamma` → `CPU/CUDA/MPS` structured wrapper（`op.meta() + op.impl()`）→ `TORCH_IMPL_FUNC(igamma_out)` → `igamma_stub` → `igamma_kernel` / `igamma_kernel_cuda` / `igamma_mps_kernel` | `aten::igamma` |
| `torch.igamma` | Autograd 反向传播 | `torch.igamma` → `aten::igamma` → `AutogradCPU/CUDA` 记录梯度 -> redispatch -> `CPU/CUDA/MPS` structured wrapper；反向中 `self` 梯度未实现，`other` 梯度公式使用 `exp` / `log` / `lgamma` | `aten::igamma` |

## 备注

1. **torch.hsplit**: CompositeImplicitAutograd，无显式 backend dispatch 段。两个重载分别进入 `aten::tensor_split.sections` / `aten::tensor_split.indices`，梯度继续由内部 `slice` 视图路径承担。

2. **torch.hstack**: CompositeImplicitAutograd。先调用 `aten::atleast_1d.Sequence` 将所有输入提升至至少 1 维，然后根据第一个张量的维度选择 `aten::cat` / `aten::cat.out` 的拼接轴（1-D → dim=0，否则 dim=1）。梯度由内部 `cat` 承担。

3. **torch.hypot**: structured op。YAML 里虽然有 `structured_delegate: hypot.out`，但按 `aten-dispatch-tracer` 规则，这不是运行时 hop；运行时 `aten::hypot` 直接进入 backend structured wrapper，再执行 `op.meta() + op.impl()`，其中 `impl` 落到 `TORCH_IMPL_FUNC(hypot_out)` 并继续经 `hypot_stub` 分发。

4. **torch.i0**: structured op。运行时 `aten::i0` 直接走 backend structured wrapper，而不是先 dispatch 到 `aten::i0.out`；真正的 backend 计算由 `TORCH_IMPL_FUNC(i0_out)` 内的 `i0_stub` 承接。反向传播需要调用 `aten::special_i1`。

5. **torch.igamma**: structured op。运行时 `aten::igamma` 直接走 backend structured wrapper，再在 `TORCH_IMPL_FUNC(igamma_out)` 中调用 `igamma_stub`；不是先 runtime hop 到 `aten::igamma.out`。反向传播中 `self` 的梯度未实现，仅 `other` 参数有梯度公式。`special_gammainc` 是 `igamma` 的别名。
