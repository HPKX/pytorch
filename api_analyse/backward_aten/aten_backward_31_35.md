# torch API 31-35 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 31 | `torch.diagonal_scatter` | inline 公式 | `DiagonalScatterBackward0` | `self: diagonal_scatter(grad, zeros_like(src), offset, dim1, dim2)`；`src: grad.diagonal(offset, dim1, dim2)` | `aten::diagonal_scatter`、`aten::zeros_like`、`aten::diagonal` | CEA non-functional；self 梯度通过将 zeros_like(src) scatter 回 grad 实现；src 梯度通过 diagonal 从 grad 中提取 |
| 32 | `torch.digamma` | inline 公式 | `DigammaBackward0` | `self: grad * polygamma(1, self)` | `aten::mul.Tensor`、`aten::polygamma` | digamma 的导数为 trigamma 函数即 `polygamma(1, self)` |
| 33 | `torch.dist` | inline + helper（`norm_backward`） | `DistBackward0` | `self: norm_backward(grad, self - other, p, result)`；`other: -norm_backward(grad, self - other, p, result)` | `aten::sub.Tensor`、`aten::sgn`、`aten::mul.Tensor`、`aten::div.Tensor`、`aten::masked_fill_`、`aten::eq`、`aten::neg`；p=2 额外 `aten::abs`；p=inf 额外 `aten::abs`、`aten::eq`、`aten::isnan`、`aten::logical_or`、`aten::sum`、`aten::div.Tensor` | `norm_backward` 是手写 helper（`FunctionsManual.cpp:242`），根据 p 值走不同分支 |
| 34 | `torch.distribution.gamma.Gamma` | `rsample()` 内部 `_standard_gamma` 的 inline 公式 | `StandardGammaBackward0`（挂在 `_standard_gamma` 上） | `self: grad * _standard_gamma_grad(self, result)` | `aten::mul.Tensor`、`aten::_standard_gamma_grad` | 纯 Python 分布类；`rsample()` 调用 `aten::_standard_gamma`（有 backward），后续 `expand`/`div` 也各自录制 autograd 节点 |
| 35 | `torch.distribution.laplace.Laplace` | 纯 Python 分布类，各子 op 独立 autograd | 无统一 backward node | 无统一 derivatives.yaml 条目 | `aten::sgn`、`aten::mul.Tensor`、`aten::neg`、`aten::add.Scalar`、`aten::div.Tensor`、`aten::mul.Scalar` | 纯 Python 分布类；top-level backward 由 `rsample()` / `log_prob()` 链内基础 ATen op 递归展开 |

## 详细分析

### 31. torch.diagonal_scatter

**反向来源类型**：inline 公式

**Forward 路径**：`aten::diagonal_scatter` → `CompositeExplicitAutogradNonFunctional` → `at::native::diagonal_scatter` → `aten::diagonal` + `aten::copy_`

**derivatives.yaml 条目**：
```yaml
- name: diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor
  self: diagonal_scatter(grad, zeros_like(src), offset, dim1, dim2)
  src: grad.diagonal(offset, dim1, dim2)
  result: auto_linear
```

**Backward Node**：`DiagonalScatterBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
// self 梯度
auto grad_result = diagonal_scatter(grad, src_info.zeros(), offset, dim1, dim2);
// src 梯度
auto grad_result = grad.diagonal(offset, dim1, dim2);
```

**反向 ATen 依赖**：
- `aten::diagonal_scatter` — 将 zeros 写回 grad 的对角线位置，从而将对角线部分的梯度置零，保留非对角线部分的梯度
- `aten::zeros_like`（通过 `src_info.zeros()` 生成） — 创建与 src 同形状的零张量
- `aten::diagonal` — 从 grad 中提取对角线视图作为 src 的梯度

---

### 32. torch.digamma

**反向来源类型**：inline 公式

**Forward 路径**：`aten::digamma` → structured backend wrapper → `digamma_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: digamma(Tensor self) -> Tensor
  self: grad * polygamma(1, self)
  result: auto_element_wise
```

**Backward Node**：`DigammaBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = any_grad_defined ? (grad * polygamma(1, self)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::polygamma` — 计算 `polygamma(1, self)` 即 trigamma 函数
- `aten::mul.Tensor` — `grad * polygamma_result`

---

### 33. torch.dist

**反向来源类型**：inline + helper（`norm_backward`）

**Forward 路径**：`aten::dist` → `CompositeExplicitAutograd` → `at::native::dist` → `aten::sub` → `aten::norm`

**derivatives.yaml 条目**：
```yaml
- name: dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
  self: norm_backward(grad, self - other, p, result)
  other: -norm_backward(grad, self - other, p, result)
  result: norm_jvp(self_p - other_p, self_t - other_t, p, result, {}, false)
```

**Backward Node**：`DistBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
// self 梯度
auto grad_result = norm_backward(grad, self - other, p, result);
// other 梯度
auto grad_result = -norm_backward(grad, self - other, p, result);
```

**`norm_backward` 实现**（`FunctionsManual.cpp:242`）：
```cpp
Tensor norm_backward(Tensor grad, const Tensor& self,
    const optional<Scalar>& p_, Tensor norm, IntArrayRef dim, bool keepdim) {
  double p = p_.value_or(2.0).toDouble();
  if (p == 0.0) return {};
  else if (p == 1.0) return self.sgn() * grad;
  else if (p == 2.0) return grad * (self / norm).masked_fill_(norm == 0, 0);
  else if (isinf(p)) {
    auto self_abs = self.abs();
    auto mask = self_abs.eq(norm).logical_or(self_abs.isnan());
    return self.sgn() * ((grad / mask.sum(dim, true)) * mask);
  } else if (p < 1.0) {
    self_scaled = self.sgn() * self.abs().pow_(p - 1).masked_fill_(self == 0, 0);
    return self_scaled * grad * norm.pow(1 - p);
  } else if (p < 2.0) {
    self_scaled = self.sgn() * self.abs().pow_(p - 1);
    scale_v = grad / norm.pow(p - 1); scale_v.masked_fill_(norm == 0, 0);
    return self_scaled * scale_v;
  } else {
    self_scaled = self * self.abs().pow_(p - 2);
    scale_v = grad / norm.pow(p - 1); scale_v.masked_fill_(norm == 0, 0);
    return self_scaled * scale_v;
  }
}
```

**反向 ATen 依赖**（因 p 值分支不同，列出所有可能）：
- `aten::sub.Tensor` — 计算 `self - other`
- `aten::neg` — other 梯度的取负
- p=1 分支：`aten::sgn`、`aten::mul.Tensor`
- p=2 分支（默认）：`aten::div.Tensor`、`aten::masked_fill_`、`aten::eq.Scalar`、`aten::mul.Tensor`
- p=inf 分支：`aten::abs`、`aten::eq.Tensor`、`aten::isnan`、`aten::logical_or`、`aten::sgn`、`aten::sum`、`aten::div.Tensor`、`aten::mul.Tensor`
- 其他 p 分支：`aten::sgn`、`aten::abs`、`aten::pow_`（Tensor inplace）、`aten::masked_fill_`、`aten::mul.Tensor`、`aten::pow`（Scalar）、`aten::div.Tensor`

---

### 34. torch.distribution.gamma.Gamma

**反向来源类型**：纯 Python 分布类；核心采样 op `_standard_gamma` 有 inline 公式

**Forward 路径**（`rsample()`）：`Gamma.rsample` → `aten::_standard_gamma`（CPU/CUDA kernel）→ `aten::expand` → `aten::div`

`_standard_gamma` 有 autograd 注册（`StandardGammaBackward0`）；`expand` 和 `div` 各自录制独立的 autograd 节点。

**derivatives.yaml 条目**（`_standard_gamma`）：
```yaml
- name: _standard_gamma(Tensor self, Generator? generator=None) -> Tensor
  self: grad * _standard_gamma_grad(self, result)
```

**生成代码**（`Functions.cpp` `StandardGammaBackward0`）：
```cpp
auto grad_result = any_grad_defined ? (grad * _standard_gamma_grad(self, result)) : Tensor();
```

**反向 ATen 依赖**（`_standard_gamma` 节点）：
- `aten::_standard_gamma_grad` — 专用 backward kernel（CPU: `_standard_gamma_grad_cpu`、CUDA: `_standard_gamma_grad_cuda`）
- `aten::mul.Tensor` — `grad * _standard_gamma_grad_result`

**其余子 op 的反向**（`rsample()` 内部）：
- `aten::expand` → backward: `aten::sum_to` / `aten::reshape`
- `aten::div` → backward: `aten::div.Tensor`、`aten::neg`、`aten::mul.Tensor`

**`log_prob()` 路径**：由 `aten::xlogy`、`aten::sub`、`aten::mul`、`aten::lgamma` 组成，各自有独立 backward。

---

### 35. torch.distribution.laplace.Laplace

**反向来源类型**：纯 Python 分布类，各子 op 独立 autograd，无统一 backward node

**Forward 路径**（`rsample()` 常规 eager）：
`Laplace.rsample` → `aten::uniform_`（不可微采样） → `aten::sign` → `aten::abs` → `aten::neg` → `aten::log1p` → `aten::mul` → `aten::sub`

Laplace 重参数化技巧：`sample = loc - scale * sign(u) * log1p(-abs(u))`，其中 `u ~ Uniform(-0.5, 0.5)`。

**derivatives.yaml 条目**（各子 op 独立）：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `uniform_` | `self: zeros_like(grad)` | 无有效梯度（零梯度） |
| `sign` | `self: zeros_like(grad)` | 无有效梯度 |
| `abs` | `self: grad * self.sgn()` | `aten::sgn`、`aten::mul.Tensor` |
| `neg` | `self: grad.neg()` | `aten::neg` |
| `log1p` | `self: grad / (self + 1)` | `aten::add.Scalar`、`aten::div.Tensor` |
| `mul` | `self/other: grad * other / grad * self` | `aten::mul.Tensor` |
| `sub` | `self: grad`；`other: -grad * alpha` | `aten::neg`、`aten::mul.Scalar` |

梯度通过重参数化路径从 `loc` 和 `scale` 参数传播：`uniform_` 和 `sign` 的梯度为零，但 `log1p(-abs(u))`、`mul(scale, ...)`、`sub(loc, ...)` 的梯度非零，使得 `loc` 和 `scale` 可以获得有效梯度。

**`log_prob()` 路径**：由 `aten::mul`、`aten::log`、`aten::sub`、`aten::abs`、`aten::div`、`aten::neg` 组成，各自有独立 backward。
