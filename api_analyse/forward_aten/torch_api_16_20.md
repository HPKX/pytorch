# torch API 16-20 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.block_diag` | 输入全为 2D dense Tensor | `torch.block_diag` -> `aten::block_diag` -> `Autograd` -> `CompositeExplicitAutograd(block_diag)` -> `at::native::block_diag` -> `aten::zeros` -> 循环中 `aten::slice.Tensor` -> `aten::copy_` | `aten::block_diag`、`aten::zeros`、`aten::slice.Tensor`、`aten::copy_` |
| `torch.block_diag` | 含 0D / 1D 输入 | `torch.block_diag` -> `aten::block_diag` -> `Autograd` -> `CompositeExplicitAutograd(block_diag)` -> `at::native::block_diag` -> 先把 0D/1D 通过 `aten::expand` 视图扩成 2D -> `aten::zeros` -> `aten::slice.Tensor` -> `aten::copy_` | `aten::block_diag`、`aten::expand`、`aten::zeros`、`aten::slice.Tensor`、`aten::copy_` |
| `torch.bucketize` | `input` 为 Tensor，`out=None` | `torch.bucketize` -> `aten::bucketize.Tensor` -> `Autograd` wrapper -> `CPU/CUDA/MPS` backend kernel -> `at::native::bucketize_{cpu/cuda/mps}` -> `aten::empty` -> 内部直接调 `native::bucketize_out_*` -> `native::searchsorted_out_*`（后两步不是 dispatcher ATen hop） | `aten::bucketize.Tensor`、`aten::empty` |
| `torch.cholesky` | 常规 dense SPD/Hermitian PD Tensor，`upper=False/True` | `torch.cholesky` / `Tensor.cholesky` -> `aten::cholesky` -> `Autograd` -> `CPU/CUDA/MPS` backend kernel wrapper -> `at::native::cholesky` -> `aten::empty`（分配 `info`） -> `cholesky_stub` -> backend `cholesky_kernel` / `mps::cholesky_stub_impl` -> `aten::_linalg_check_errors` -> `aten::tril_`（`upper=False`）或 `aten::triu_`（`upper=True`） | `aten::cholesky`、`aten::empty`、`aten::_linalg_check_errors`、`aten::tril_`、`aten::triu_` |
| `torch.cholesky_solve` | CPU/CUDA dense Tensor，`out=None` | `torch.cholesky_solve` / `Tensor.cholesky_solve` -> `aten::cholesky_solve` -> `Autograd` -> `CompositeExplicitAutograd(cholesky_solve)` -> `at::native::cholesky_solve` -> `_linalg_broadcast_batch_dims`（helper，非 ATen hop） -> `aten::_cholesky_solve_helper` -> `CPU/CUDA` backend kernel `_cholesky_solve_helper_{cpu/cuda}` | `aten::cholesky_solve`、`aten::_cholesky_solve_helper` |
| `torch.clip` | 标量 `min` / `max` | `torch.clip` -> `aten::clip` -> `CompositeImplicitAutograd(clip)` -> `at::native::clip` -> `aten::clamp` -> `Autograd` -> `CPU/CUDA/MPS` structured wrapper -> `meta + clamp_out impl` -> `clamp_scalar_stub` / `clamp_min_scalar_stub` / `clamp_max_scalar_stub`。其中 `structured_delegate: clamp.out` 只是 codegen 关系，不是额外 runtime hop | `aten::clip`、`aten::clamp` |
| `torch.clip` | Tensor `min` / `max` | `torch.clip` -> `aten::clip.Tensor` -> `CompositeImplicitAutograd(clip.Tensor)` -> `at::native::clip` -> `aten::clamp.Tensor` -> `Autograd` -> `CPU/CUDA/MPS` structured wrapper -> `meta + clamp_Tensor_out impl` -> `clamp_stub` / `maximum_stub` / `minimum_stub`。其中 `structured_delegate: clamp.Tensor_out` 只是 codegen 关系，不是额外 runtime hop | `aten::clip.Tensor`、`aten::clamp.Tensor` |

## 备注

- `torch.clip` 不是纯 Python 包装；ATen 里有独立的 `clip` / `clip.Tensor` schema，但 native 实现本质转发到 `aten::clamp` / `aten::clamp.Tensor`。
- `torch.cholesky` 也不是 runtime 上转到 `aten::linalg_cholesky_ex`；它走自己独立的 `aten::cholesky` backend 实现，并在 native C++ 里发 deprecated warning。
- `torch.bucketize` backend 实现内部复用 `searchsorted_out_*` 的 native C++ 函数，但这一步不是新的 dispatcher `aten::searchsorted` hop。
