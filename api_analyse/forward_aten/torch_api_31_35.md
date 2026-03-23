# torch API 31-35 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.diagonal_scatter` | 普通张量 | `torch.diagonal_scatter` -> `aten::diagonal_scatter` -> `CompositeExplicitAutogradNonFunctional` wrapper -> `at::native::diagonal_scatter` -> `aten::diagonal` -> `aten::copy_` | `aten::diagonal_scatter`, `aten::diagonal`, `aten::copy_` |
| `torch.diagonal_scatter` | `requires_grad=True` | `torch.diagonal_scatter` -> `aten::diagonal_scatter` -> `Autograd(DiagonalScatterBackward0)` -> redispatch -> `CompositeExplicitAutogradNonFunctional` -> `at::native::diagonal_scatter` -> `aten::diagonal` -> `aten::copy_` | `aten::diagonal_scatter`, `aten::diagonal`, `aten::copy_` |
| `torch.digamma` | 普通 `CPU/CUDA/MPS` 张量 | `torch.digamma` -> `aten::digamma` -> backend structured wrapper -> `op.meta()` -> `op.impl()` -> `digamma_stub` -> backend kernel | `aten::digamma` |
| `torch.digamma` | `requires_grad=True` | `torch.digamma` -> `aten::digamma` -> `Autograd(DigammaBackward0)` -> redispatch -> backend structured wrapper -> `op.meta()` -> `op.impl()` -> `digamma_stub` -> backend kernel | `aten::digamma` |
| `torch.dist` | 普通张量 | `torch.dist` -> `aten::dist` -> `CompositeExplicitAutograd` wrapper -> `at::native::dist` -> `aten::sub` -> `aten::norm` | `aten::dist`, `aten::sub`, `aten::norm` |
| `torch.dist` | `requires_grad=True` | `torch.dist` -> `aten::dist` -> `Autograd(DistBackward0)` -> redispatch -> `CompositeExplicitAutograd` -> `at::native::dist` -> `aten::sub` -> `aten::norm` | `aten::dist`, `aten::sub`, `aten::norm` |
| `torch.distribution.gamma.Gamma` | 构造（张量参数） | `Gamma.__init__` -> `broadcast_all` -> `torch.broadcast_tensors` -> `aten::broadcast_tensors` -> `CompositeImplicitAutograd` wrapper -> `at::native::broadcast_tensors` -> `expand_outplace` | `aten::broadcast_tensors` |
| `torch.distribution.gamma.Gamma` | `rsample()` | `Gamma.rsample` -> `aten::expand` -> `aten::_standard_gamma` -> `Autograd(_standard_gamma)`（若 `concentration` 参与求导） -> CPU/CUDA kernel (`_s_gamma_cpu/_s_gamma_cuda`) -> `aten::expand` -> `aten::div` | `aten::expand`, `aten::_standard_gamma`, `aten::div` |
| `torch.distribution.gamma.Gamma` | `log_prob()` | `Gamma.log_prob` -> `aten::as_tensor` -> `aten::xlogy(concentration, rate)` -> `aten::sub(concentration, 1)` -> `aten::xlogy` -> `aten::mul(rate, value)` -> `aten::sub` -> `aten::lgamma` | `aten::xlogy`, `aten::sub`, `aten::mul`, `aten::lgamma` |
| `torch.distribution.laplace.Laplace` | 构造（张量参数） | `Laplace.__init__` -> `broadcast_all` -> `torch.broadcast_tensors` -> `aten::broadcast_tensors` -> `CompositeImplicitAutograd` wrapper -> `at::native::broadcast_tensors` -> `expand_outplace` | `aten::broadcast_tensors` |
| `torch.distribution.laplace.Laplace` | `rsample()`（常规 eager） | `Laplace.rsample` -> `self.loc.new(shape).uniform_(...)` -> `aten::uniform_` -> backend RNG kernel -> `aten::sign` -> `aten::abs` -> `aten::neg` -> `aten::log1p` -> `aten::mul` -> `aten::sub` | `aten::uniform_`, `aten::sign`, `aten::abs`, `aten::neg`, `aten::log1p`, `aten::mul`, `aten::sub` |
| `torch.distribution.laplace.Laplace` | `rsample()`（tracing 分支） | `Laplace.rsample` -> `aten::rand` -> `aten::mul` -> `aten::sub` -> `aten::sign` -> `aten::abs` -> `aten::clamp` -> `aten::neg` -> `aten::log1p` -> `aten::mul` -> `aten::sub` | `aten::rand`, `aten::mul`, `aten::sub`, `aten::sign`, `aten::abs`, `aten::clamp`, `aten::neg`, `aten::log1p` |
| `torch.distribution.laplace.Laplace` | `log_prob()` | `Laplace.log_prob` -> `aten::mul(2, scale)` -> `aten::log` -> `aten::sub(value, loc)` -> `aten::abs` -> `aten::div` -> `aten::neg/sub` | `aten::mul`, `aten::log`, `aten::sub`, `aten::abs`, `aten::div`, `aten::neg` |

## 备注

- `torch.dist` 的前向实现就是 `at::norm(self - other, p)`，不走 `aten::cdist` 或 `aten::_euclidean_dist`。
- `Gamma`、`Laplace` 是 Python 分布类，不存在单独的 `aten::Gamma` / `aten::Laplace`；真正的 dispatcher 入口发生在构造时的 `broadcast_tensors` 和各实例方法内部。
- 题目清单里写成 `torch.distribution.*`，但仓库中的公开类路径实际是 `torch.distributions.gamma.Gamma` 与 `torch.distributions.laplace.Laplace`。
