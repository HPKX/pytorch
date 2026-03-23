# torch API 206-210 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.triu_indices` | 按 `device` 创建上三角索引（`CPU/CUDA/MPS`） | Python 绑定 -> `aten::triu_indices` -> `BackendSelect`（按 `dtype/layout/device` 选 key）-> `CPU/CUDA/MPS` -> `native::triu_indices_{cpu,cuda,mps}` | `aten::triu_indices` |
| `torch.true_divide` | dense `Tensor / Tensor` | Python 绑定 -> `aten::true_divide.Tensor` -> `CompositeImplicitAutograd::native::true_divide` -> 内部改调 `aten::div.Tensor` -> `AutogradCPU/CUDA`（若需要梯度）-> 后端 `div` structured wrapper（`meta -> impl`） | `aten::true_divide.Tensor`, `aten::div.Tensor` |
| `torch.true_divide` | dense `Tensor / Scalar` | Python 绑定 -> `aten::true_divide.Scalar` -> `CompositeImplicitAutograd::native::true_divide` -> 内部改调 `aten::div.Scalar` -> `AutogradCPU/CUDA`（若需要梯度）-> `CompositeExplicitAutograd::native::div`（把 `Scalar` 包成 tensor）-> `aten::div.Tensor` -> 后端 `div` structured wrapper（`meta -> impl`） | `aten::true_divide.Scalar`, `aten::div.Scalar`, `aten::div.Tensor` |
| `torch.vander` | 1D 输入；默认 `increasing=False`（`increasing=True` 仅省掉最后一步） | Python 绑定 -> `aten::vander` -> `CompositeImplicitAutograd::native::vander` -> `aten::empty` -> `aten::select.int` + `aten::fill_.Scalar` -> `aten::slice.Tensor` + `aten::unsqueeze` + `aten::copy_` -> `aten::cumprod` -> 默认再 `aten::flip` | `aten::vander`, `aten::empty`, `aten::select.int`, `aten::fill_.Scalar`, `aten::slice.Tensor`, `aten::unsqueeze`, `aten::copy_`, `aten::cumprod`, `aten::flip` |
| `torch.view_as_real` | dense 复数张量，`requires_grad=True` | Python 绑定 -> `aten::view_as_real` -> `AutogradCPU/CUDA` 的 `ViewAsRealBackward0` 包装 -> `CPU/CUDA/MPS/Meta::view_as_real` -> `native::view_as_real` | `aten::view_as_real` |
| `torch.vsplit` | 至少 2 维，按段数切分 | Python 绑定 -> `aten::vsplit.int` -> `CompositeImplicitAutograd::native::vsplit` -> `aten::tensor_split.sections(dim=0)` -> `CompositeImplicitAutograd::tensor_split_sections_symint` -> 多次 `aten::slice.Tensor` | `aten::vsplit.int`, `aten::tensor_split.sections`, `aten::slice.Tensor` |
| `torch.vsplit` | 至少 2 维，按索引切分 | Python 绑定 -> `aten::vsplit.array` -> `CompositeImplicitAutograd::native::vsplit` -> `aten::tensor_split.indices(dim=0)` -> `CompositeImplicitAutograd::tensor_split_indices_symint` -> 多次 `aten::slice.Tensor` | `aten::vsplit.array`, `aten::tensor_split.indices`, `aten::slice.Tensor` |

## 备注

- `torch.true_divide` 自己没有单独的 `derivatives.yaml` 公式；梯度记录落在内部 `aten::div.*`。
- `torch.vander` 走的是 `aten::vander`，不是 `torch.linalg.vander` 对应的 `aten::linalg_vander`。
- 表中 `torch.view_as_real` 聚焦 dense path；稀疏复数张量会改走 `view_as_real_sparse`，再对子 `values()` 递归调用 `aten::view_as_real`。
- `triu_indices` 在 `VariableTypeEverything.cpp` 注册的是 `autogradNotImplementedFallback()`，没有单独 backward 链。
