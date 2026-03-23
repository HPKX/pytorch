# torch API 56-60 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.igammac` | 普通 Tensor；CPU/CUDA/MPS；`other` 可求导 | generated `torch` binding -> `aten::igammac` -> `Autograd` 包装（若 `requires_grad`，保存 `self/other`）-> CompositeExplicitAutogradNonFunctional wrapper 做 `meta`，随后调用 `aten::igammac.out` -> CPU/CUDA/MPS structured out-wrapper（`meta + impl`）-> `igammac_stub` -> backend `igammac_kernel` | `aten::igammac`, `aten::igammac.out` |
| `torch.inner` | 任一输入是标量 | generated `torch` binding -> `aten::inner` -> `CompositeImplicitAutograd` -> `at::native::inner` -> `aten::mul.Tensor` -> 后续由 `mul` 的 autograd / backend 路径处理 | `aten::inner`, `aten::mul.Tensor` |
| `torch.inner` | 两侧都不是标量 | generated `torch` binding -> `aten::inner` -> `CompositeImplicitAutograd` -> `at::native::inner` -> `aten::tensordot` -> `CompositeImplicitAutograd` -> `at::native::tensordot` -> `aten::permute` / `aten::reshape` -> 通常落到 `aten::mm`；全缩并时可走 `aten::dot` | `aten::inner`, `aten::tensordot`, `aten::permute`, `aten::reshape`, `aten::mm`, `aten::dot` |
| `torch.is_complex` | 任意 Tensor | generated `torch` binding -> `aten::is_complex` -> `CompositeImplicitAutograd` -> `at::native::is_complex` -> 直接查询 Tensor dtype 元数据（无后端 kernel hop） | `aten::is_complex` |
| `torch.is_floating_point` | 任意 Tensor | generated `torch` binding -> `aten::is_floating_point` -> `CompositeImplicitAutograd` -> `at::native::is_floating_point` -> 直接查询 Tensor dtype 元数据（无后端 kernel hop） | `aten::is_floating_point` |
| `torch.is_nonzero` | 1 元素 dense Tensor | generated `torch` binding -> `aten::is_nonzero` -> `CompositeImplicitAutograd` -> `at::native::is_nonzero` -> `aten::item` -> `aten::_local_scalar_dense` -> backend local-scalar kernel -> C++ 标量到 `bool` 判定 | `aten::is_nonzero`, `aten::item`, `aten::_local_scalar_dense` |
| `torch.is_nonzero` | 0 元素或多元素 Tensor | generated `torch` binding -> `aten::is_nonzero` -> `CompositeImplicitAutograd` -> `at::native::is_nonzero` -> `numel` 检查失败直接报错，不再进入 item / local-scalar 路径 | `aten::is_nonzero` |

## 备注

- `igammac` 在 `derivatives.yaml` 里只有 `other` 的梯度公式；`self` 的梯度明确是 `not_implemented("igammac: input")`。
- `inner`、`is_complex`、`is_floating_point`、`is_nonzero` 在 `VariableTypeEverything.cpp` 中都没有专门的前向 autograd 注册；`inner` 的梯度由内部 `mul` / `tensordot` 链承担，后三者返回 `bool`。
