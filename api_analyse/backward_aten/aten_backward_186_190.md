# torch API 186-190 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 186 | `torch.select_scatter` | inline 公式 | `SelectScatterBackward0` | `self: select_scatter_symint(grad, zeros_like(src), dim, index)`；`src: grad.select_symint(dim, index)` | `aten::select_scatter`、`aten::zeros_like`、`aten::select.int` | backward 直接复用同一个 scatter schema 回填 `self` 梯度；`src` 梯度则是反向 `select` |
| 187 | `torch.sgn` | helper 公式 | `SgnBackward0` | `self: sgn_backward(self, grad, result)` | helper 展开后主要依赖 `aten::abs`、`aten::mul.Tensor`、`aten::conj`、`aten::div.Tensor`、`aten::masked_fill_.Scalar`、`aten::eq.Scalar`；实数分支返回零张量 | 复数分支才有非零梯度；实数输入按定义返回零梯度 |
| 188 | `torch.signbit` | 不可微 | 无 | 无条目 | 无 | 输出为布尔 Tensor；autograd wrapper 只 redispatch，不创建 `grad_fn` |
| 189 | `torch.slogdet` | 前端别名 / CIA 包装，真实反向挂在 `_linalg_slogdet` helper 上 | `LinalgSlogdetBackward0` | `_linalg_slogdet: A: slogdet_backward(grad_sign, grad_logabsdet, A, sign, LU, pivots)` | helper 展开后主要依赖 `aten::imag`、`aten::conj`、`aten::diag_embed`、`aten::unsqueeze`、`aten::expand_as`、`aten::mT`、`aten::linalg_lu_solve` / `aten::linalg_solve` | `slogdet` / `linalg_slogdet` 自己无导数条目；真正存图的是 `_linalg_slogdet` |
| 190 | `torch.special.bessel_j0` | 不可微 | 无 | `self: non_differentiable` | 无 | generated wrapper 只 redispatch 并返回结果，不设置 history |

## 详细分析

### 186. torch.select_scatter

**反向来源类型**：inline 公式

**锁定 schema**：
`aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
  self: select_scatter_symint(grad, zeros_like(src), dim, index)
  src: grad.select_symint(dim, index)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `SelectScatterBackward0`
- `Functions.cpp` 中为：
```cpp
select_scatter_symint(grad, src_info.zeros(), dim, index)
grad.select_symint(dim, index)
```

**反向 ATen 依赖**：
- `aten::select_scatter`
- `aten::zeros_like`
- `aten::select.int`

**备注**：
- 前向 CEA 实现本身是 `clone_preserve_strides -> select.int -> copy_`，但 backward 不回到 `copy_`，而是直接使用 derivatives.yaml 给出的 `select_scatter_symint` 公式。

---

### 187. torch.sgn

**反向来源类型**：helper 公式

**锁定 schema**：
`aten::sgn(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: sgn(Tensor self) -> Tensor
  self: sgn_backward(self, grad, result)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `SgnBackward0`
- `Functions.cpp` 中为：
```cpp
auto grad_result = any_grad_defined ? (sgn_backward(self, grad, result)) : Tensor();
```

**helper 展开**（`FunctionsManual.cpp`）：
```cpp
if (x.is_complex()) {
  auto abs = x.abs();
  return ((gx - (sgn * sgn) * gx.conj()) / (2. * abs))
      .masked_fill_(abs == 0., 0.);
} else {
  return at::_efficientzerotensor(sgn.sizes(), sgn.options());
}
```

**反向 ATen 依赖**：
- 复数分支：
  - `aten::abs`
  - `aten::mul.Tensor`
  - `aten::conj`
  - `aten::div.Tensor`
  - `aten::masked_fill_.Scalar`
  - `aten::eq.Scalar`
- 实数分支：
  - 零梯度张量构造（等价于 `zeros_like` 语义）

**备注**：
- 这是少数“复数可导、实数梯度恒零”的 unary op。

---

### 188. torch.signbit

**反向来源类型**：不可微

**锁定 schema**：
`aten::signbit(Tensor self) -> Tensor`

`derivatives.yaml` 中没有 `signbit` 条目。`VariableTypeEverything.cpp` 的 wrapper 只有 unpack + redispatch：
```cpp
auto _tmp = ([&]() {
  return at::redispatch::signbit(..., self_);
})();
return result;
```

没有 `compute_requires_grad(...)`、没有 `grad_fn`、没有 `set_history(...)`。

**结论**：
- `signbit` 输出是布尔 Tensor，不可微。
- 不创建 backward node，也没有反向 ATen 依赖。

---

### 189. torch.slogdet

**反向来源类型**：前端别名 / CIA 包装，真实反向挂在 `_linalg_slogdet`

**锁定 schema**：
- 前端：`aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)`
- 内层：`aten::linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet)`
- 真正可微 schema：`aten::_linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)`

`derivatives.yaml` 的条目是：
```yaml
- name: _linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)
  A: slogdet_backward(grad_sign, grad_logabsdet, A, sign, LU, pivots)
```

**生成代码**：
- `VariableTypeEverything.cpp` 在 `_linalg_slogdet` 上生成 `LinalgSlogdetBackward0`
- `Functions.cpp` 调用 `slogdet_backward(...)`

**helper 展开**（`FunctionsManual.cpp`）的主要 ATen 依赖：
- `aten::imag`
- `aten::conj`
- `aten::diag_embed`
- `aten::unsqueeze`
- `aten::expand_as`
- `aten::mT`
- 一阶 backward 时：
  - `aten::linalg_lu_solve`
- 高阶梯度开启时：
  - `aten::linalg_solve`

**备注**：
- `slogdet` / `linalg_slogdet` 自己只是包装层；分析 backward 必须锁定 `_linalg_slogdet`。
- 复数场景会把 `grad_sign` 转成对 `g` 的复数修正项，因此出现 `aten::imag` 和 `aten::conj`。

---

### 190. torch.special.bessel_j0

**反向来源类型**：不可微

**锁定 schema**：
`aten::special_bessel_j0(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: special_bessel_j0(Tensor self) -> Tensor
  self: non_differentiable
```

`VariableTypeEverything.cpp` wrapper 只做 redispatch 并直接返回结果，不设置 history。

**结论**：
- 没有 backward node。
- 没有反向 ATen 依赖。
