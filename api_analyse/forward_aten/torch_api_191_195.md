# torch API 191-195 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.special.bessel_j1` | 普通 `CPU/CUDA/MPS` 张量 | `torch.special.bessel_j1` -> `aten::special_bessel_j1` -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_bessel_j1_stub` -> backend kernel | `aten::special_bessel_j1` |
| `torch.special.bessel_j1` | `requires_grad` 输入 | `torch.special.bessel_j1` -> `aten::special_bessel_j1` -> Autograd wrapper（`non_differentiable`，仅 redispatch） -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_bessel_j1_stub` -> backend kernel | `aten::special_bessel_j1` |
| `torch.special.bessel_y0` | 普通 `CPU/CUDA/MPS` 张量 | `torch.special.bessel_y0` -> `aten::special_bessel_y0` -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_bessel_y0_stub` -> backend kernel | `aten::special_bessel_y0` |
| `torch.special.bessel_y0` | `requires_grad` 输入 | `torch.special.bessel_y0` -> `aten::special_bessel_y0` -> Autograd wrapper（`non_differentiable`，仅 redispatch） -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_bessel_y0_stub` -> backend kernel | `aten::special_bessel_y0` |
| `torch.special.bessel_y1` | 普通 `CPU/CUDA/MPS` 张量 | `torch.special.bessel_y1` -> `aten::special_bessel_y1` -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_bessel_y1_stub` -> backend kernel | `aten::special_bessel_y1` |
| `torch.special.bessel_y1` | `requires_grad` 输入 | `torch.special.bessel_y1` -> `aten::special_bessel_y1` -> Autograd wrapper（`non_differentiable`，仅 redispatch） -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_bessel_y1_stub` -> backend kernel | `aten::special_bessel_y1` |
| `torch.special.i0` | 普通 `CPU/CUDA/MPS` 张量 | `torch.special.i0` -> `aten::special_i0` -> `CompositeImplicitAutograd` wrapper -> `at::native::special_i0` -> `aten::i0` -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `i0_stub` -> backend kernel | `aten::special_i0`, `aten::i0` |
| `torch.special.i0` | `requires_grad` 输入 | `torch.special.i0` -> `aten::special_i0` -> `CompositeImplicitAutograd` wrapper -> `at::native::special_i0` -> `aten::i0` -> Autograd wrapper -> redispatch -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `i0_stub` -> backend kernel | `aten::special_i0`, `aten::i0` |
| `torch.special.i1` | 普通 `CPU/CUDA/MPS` 张量 | `torch.special.i1` -> `aten::special_i1` -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_i1_stub` -> backend kernel | `aten::special_i1` |
| `torch.special.i1` | `requires_grad` 输入 | `torch.special.i1` -> `aten::special_i1` -> Autograd wrapper（记录 `grad_fn` 后 redispatch） -> `CPU/CUDA/MPS` structured wrapper -> `op.meta()` -> `op.impl()` -> `special_i1_stub` -> backend kernel | `aten::special_i1` |

## 备注

- `special_i0` 在 `UnaryOps.cpp` 中只是 `i0` 的别名，所以它自己的前向先入 `aten::special_i0`，真正的 backend 计算与 autograd 都由内层 `aten::i0` 承接；其反向公式会用到 `aten::special_i1`。
- `special_bessel_j1`、`special_bessel_y0`、`special_bessel_y1`、`special_i1` 都是 structured unary op；YAML 里的 `structured_delegate: foo.out` 是 codegen 关系，不是运行时先 dispatch 到 `foo.out`。
- `special_bessel_j1` / `special_bessel_y0` / `special_bessel_y1` / `special_i1` 还生成了 `CompositeExplicitAutogradNonFunctional` 默认实现，供缺少专用 backend 时回落；表中只列常见 `CPU/CUDA/MPS` 的实际计算路径。
