# torch API 146-150 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.nn.functional.huber_loss` | `weight=None`，标准 Huber | `F.huber_loss -> torch._C._nn.huber_loss -> aten::huber_loss -> AutogradCPU/CUDA/MPS -> redispatch -> CPU/CUDA: native::huber_loss / MPS: huber_loss_mps -> huber_stub(TensorIterator) -> backend kernel -> aten::mean / aten::sum（按 reduction）` | `aten::huber_loss`, `aten::mean`, `aten::sum` |
| `torch.nn.functional.huber_loss` | `weight!=None`，Python 侧加权 | `F.huber_loss -> aten::huber_loss(reduction=None) -> AutogradCPU/CUDA/MPS -> backend huber kernel -> aten::mul(weight) -> aten::mean / aten::sum 或直接返回` | `aten::huber_loss`, `aten::mul`, `aten::mean`, `aten::sum` |
| `torch.nn.functional.lp_pool1d` | 常规 1D Lp pooling | `F.lp_pool1d -> aten::pow.Tensor_Scalar -> aten::avg_pool1d -> CompositeImplicitAutograd::native::avg_pool1d -> aten::unsqueeze -> aten::avg_pool2d -> AutogradCPU/CUDA/MPS -> structured avg_pool2d(meta+impl) -> CPU/CUDA/MPS kernel -> aten::squeeze.dim -> aten::sign / aten::abs / aten::relu / aten::mul.Scalar -> aten::pow.Tensor_Scalar` | `aten::pow.Tensor_Scalar`, `aten::avg_pool1d`, `aten::unsqueeze`, `aten::avg_pool2d`, `aten::squeeze.dim`, `aten::sign`, `aten::abs`, `aten::relu`, `aten::mul.Scalar` |
| `torch.nn.functional.lp_pool2d` | 常规 2D Lp pooling | `F.lp_pool2d -> aten::pow.Tensor_Scalar -> aten::avg_pool2d -> AutogradCPU/CUDA/MPS -> structured avg_pool2d(meta+impl) -> CPU/CUDA/MPS kernel -> aten::sign / aten::abs / aten::relu / aten::mul.Scalar -> aten::pow.Tensor_Scalar` | `aten::pow.Tensor_Scalar`, `aten::avg_pool2d`, `aten::sign`, `aten::abs`, `aten::relu`, `aten::mul.Scalar` |
| `torch.nn.functional.margin_ranking_loss` | `reduction='mean'` | `F.margin_ranking_loss -> torch.margin_ranking_loss -> aten::margin_ranking_loss -> CompositeImplicitAutograd::native::margin_ranking_loss -> aten::sub -> aten::mul -> aten::neg -> aten::add.Scalar -> aten::clamp_min -> aten::mean / aten::sum（按 reduction）` | `aten::margin_ranking_loss`, `aten::sub`, `aten::mul`, `aten::neg`, `aten::add.Scalar`, `aten::clamp_min`, `aten::mean`, `aten::sum` |
| `torch.nn.functional.max_pool3d` | `return_indices=False`，常规 dense tensor | `F.max_pool3d -> torch.max_pool3d -> aten::max_pool3d -> CompositeImplicitAutograd::native::max_pool3d -> dense 路径转到 aten::max_pool3d_with_indices -> AutogradCPU/CUDA/MPS -> redispatch -> max_pool3d_with_indices_cpu / cuda / mps -> 取 result[0]` | `aten::max_pool3d`, `aten::max_pool3d_with_indices` |
| `torch.nn.functional.max_pool3d` | `return_indices=True` | `F.max_pool3d_with_indices -> torch._C._nn.max_pool3d_with_indices -> aten::max_pool3d_with_indices -> AutogradCPU/CUDA/MPS -> redispatch -> max_pool3d_with_indices_cpu / cuda / mps -> 返回 output, indices` | `aten::max_pool3d_with_indices` |

## 备注

- `lp_pool1d`、`lp_pool2d` 都是 Python 组合逻辑，不存在单一 fused 的 LP pooling ATen kernel。
- `margin_ranking_loss` 与 `max_pool3d` 公开 schema 都是 composite 包装层；前者把 autograd 交给内部逐元素/归约算子，后者的 dense 训练路径最终依赖 `aten::max_pool3d_with_indices`。
