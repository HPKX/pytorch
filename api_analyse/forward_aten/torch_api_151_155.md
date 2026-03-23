# torch API 151-155 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.nn.functional.max_unpool1d` | 常规前向（1D 反池化，Python 包装到 2D unpool） | `F.max_unpool1d -> aten::unsqueeze(-1)`（`input/indices`）`-> torch._C._nn.max_unpool2d -> aten::max_unpool2d -> AutogradCPU/CUDA/MPS 包装保存 indices -> redispatch -> CPU/CUDA/MPS: max_unpooling2d_forward_cpu/cuda/mps`；CPU 路径内部再到 `max_unpool2d_kernel`，最后 `aten::squeeze.dim(-1)` 返回 1D 结果 | `aten::unsqueeze`, `aten::max_unpool2d`, `aten::squeeze.dim` |
| `torch.nn.functional.max_unpool3d` | 常规前向（3D 反池化） | `F.max_unpool3d -> torch._C._nn.max_unpool3d -> aten::max_unpool3d -> AutogradCPU/CUDA/MPS 包装保存 indices -> redispatch -> CPU/CUDA/MPS: max_unpooling3d_forward_cpu/cuda/mps`；CPU 路径内部再到 `max_unpool3d_kernel` | `aten::max_unpool3d` |
| `torch.nn.functional.multi_margin_loss` | 常规前向/训练（`p=1` 或 `p=2`，可选 `weight`） | `F.multi_margin_loss -> torch._C._nn.multi_margin_loss -> aten::multi_margin_loss -> AutogradCPU/CUDA 包装保存 self/target/weight/p/margin/reduction -> redispatch -> CPU/CUDA: multi_margin_loss_cpu / multi_margin_loss_cuda` | `aten::multi_margin_loss` |
| `torch.nn.functional.multilabel_margin_loss` | 常规前向/训练 | `F.multilabel_margin_loss -> torch._C._nn.multilabel_margin_loss -> aten::multilabel_margin_loss -> CompositeImplicitAutograd::native::multilabel_margin_loss -> aten::multilabel_margin_loss_forward -> AutogradCPU/CUDA 包装保存 self/target/reduction -> redispatch -> CPU/CUDA: multilabel_margin_loss_forward_cpu / multilabel_margin_loss_forward_cuda`；外层仅取 `output` | `aten::multilabel_margin_loss`, `aten::multilabel_margin_loss_forward` |
| `torch.nn.functional.multilabel_soft_margin_loss` | `weight=None`, `reduction='mean'` | `F.multilabel_soft_margin_loss` 为 Python 组合：`logsigmoid(input) -> torch._C._nn.log_sigmoid -> aten::log_sigmoid -> CompositeImplicitAutograd -> aten::log_sigmoid_forward -> backend kernel`；并行计算 `aten::neg -> aten::log_sigmoid(-input)` 后做 `aten::rsub.Scalar(1, target)`、`aten::mul`、`aten::add`，再 `aten::sum(dim=class) -> aten::div.Scalar -> aten::mean` | `aten::log_sigmoid`, `aten::log_sigmoid_forward`, `aten::neg`, `aten::rsub.Scalar`, `aten::mul`, `aten::add`, `aten::sum`, `aten::div.Scalar`, `aten::mean` |
| `torch.nn.functional.multilabel_soft_margin_loss` | `weight!=None`，带类别权重 | `F.multilabel_soft_margin_loss` 先按上一路径构造逐元素 loss：`aten::log_sigmoid(input)` 与 `aten::neg(input) -> aten::log_sigmoid(-input)`，两次 `aten::log_sigmoid` 都进入 `aten::log_sigmoid_forward`；随后 `aten::rsub.Scalar(1, target) -> aten::mul -> aten::add -> aten::neg`；再额外 `aten::mul(weight)`，然后 `aten::sum(dim=class) -> aten::div.Scalar`，最后按 `reduction` 做 `aten::mean` 或 `aten::sum` | `aten::log_sigmoid`, `aten::neg`, `aten::log_sigmoid_forward`, `aten::rsub.Scalar`, `aten::mul`, `aten::add`, `aten::sum`, `aten::div.Scalar`, `aten::mean` |

## 备注

- `max_unpool1d` 没有独立的 ATen unpool schema，实质是 `unsqueeze -> aten::max_unpool2d -> squeeze`。
- `max_unpool2d`/`max_unpool3d` 的反向没有专门的 `aten::max_unpool*_backward` schema，`derivatives.yaml` 里直接写成 `max_pool_double_backward(...)`。
- `multilabel_soft_margin_loss` 没有单独的 fused loss ATen 接口，当前仓库实现是 Python 侧围绕 `aten::log_sigmoid` 的组合。
