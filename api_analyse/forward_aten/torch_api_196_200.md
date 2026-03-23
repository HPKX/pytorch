# torch API 196-200 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.special.i1e` | `input.requires_grad=True`，普通 CPU/CUDA/MPS 张量 | `torch.special.i1e` -> `aten::special_i1e` -> `Autograd` -> CPU/CUDA/MPS 后端 wrapper（`meta`+`impl`） -> `special_i1e_stub` -> `i1e` kernel | `aten::special_i1e` |
| `torch.special.zeta` | Tensor-Tensor，关注 `other` 的梯度 | `torch.special.zeta` -> `aten::special_zeta` -> `Autograd` -> CPU/CUDA/MPS 后端 wrapper（`meta`+`impl`） -> `zeta_stub` -> `zeta` kernel | `aten::special_zeta` |
| `torch.special.zeta` | Scalar-Tensor，`other.requires_grad=True` | `torch.special.zeta` -> `aten::special_zeta.self_scalar` -> `Autograd` -> `CompositeExplicitAutograd(native::special_zeta)` -> `aten::special_zeta`（把 `Scalar` 包成 wrapped scalar tensor 后重入 Tensor-Tensor 重载） -> 后端 wrapper（`meta`+`impl`） -> `zeta_stub` | `aten::special_zeta.self_scalar`<br>`aten::special_zeta` |
| `torch.substract` | 名称校验 / 常见误拼 | 仓库中无 `torch.substract` Python 入口，也无对应 `substract` schema；若意图是 `torch.subtract`，链路为：`torch.subtract` -> `aten::subtract.Tensor` / `aten::subtract.Scalar` -> `CompositeImplicitAutograd(native::subtract)` -> `aten::sub.Tensor` / `aten::sub.Scalar` -> 后端 `sub_out` 实现 | 无；候选为 `aten::subtract.Tensor`、`aten::subtract.Scalar`、`aten::sub.Tensor`、`aten::sub.Scalar` |
| `torch.svd` | `compute_uv=True` | `torch.svd` -> `aten::svd` -> `CompositeImplicitAutograd(native::svd)` -> `aten::linalg_svd` -> `CompositeImplicitAutograd(native::linalg_svd)` -> `aten::_linalg_svd` -> `Autograd` -> CPU/CUDA 后端 wrapper（`meta`+`impl`） -> `svd_stub` -> LAPACK/cuSOLVER -> `aten::mH`（把返回的 `Vh` 变成 `V`） | `aten::svd`<br>`aten::linalg_svd`<br>`aten::_linalg_svd`<br>`aten::mH` |
| `torch.svd` | `compute_uv=False`（按无 grad 前向链路） | `torch.svd` -> `aten::svd` -> `CompositeImplicitAutograd(native::svd)` -> `aten::linalg_svdvals` -> `CompositeImplicitAutograd(native::linalg_svdvals)` -> `aten::_linalg_svd(full_matrices=false, compute_uv=false)` -> CPU/CUDA 后端 wrapper（`meta`+`impl`） -> `svd_stub` -> `native::svd` 用 `zeros` 补 `U/Vh` -> `aten::mH` | `aten::svd`<br>`aten::linalg_svdvals`<br>`aten::_linalg_svd`<br>`aten::mH` |
| `torch.swapdims` | 普通 dense 张量，关注 view 语义 | `torch.swapdims` -> `aten::swapdims` -> `CompositeImplicitAutograd(native::swapdims)` -> `aten::transpose.int` -> `ADInplaceOrView::transpose_int(as_view)` -> `native::transpose` -> `aten::as_strided` | `aten::swapdims`<br>`aten::transpose.int`<br>`aten::as_strided` |

## 备注

- `torch.substract` 在该仓库中不存在；最接近且真实存在的 API 是 `torch.subtract`。
- `torch.special.zeta` 的 backward 只给 `other` 提供公式；`self` 方向在 `derivatives.yaml` 中是 `not_implemented("zeta")`。
- `torch.svd(..., compute_uv=False)` 若输入需要 grad，`linalg_svdvals` 内部会把 `_linalg_svd` 的 `compute_uv` 置为真以支持梯度；上表该行按无 grad 的前向主路径写。
