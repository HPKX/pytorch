# torch API 196-200 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 196 | `torch.special.i1e` | helper 展开，无独立 backward schema | `SpecialI1EBackward0` | `self: i1e_backward(grad, self, result)` | `aten::special_i0e`、`aten::sgn`、`aten::reciprocal`、`aten::where`、`aten::abs`、`aten::mul.Tensor`、`aten::sub.Tensor` | `i1e_backward` 是 `FunctionsManual.cpp` helper，不是公开 ATen schema；真正 ATen 依赖来自 helper 体内展开 |
| 197 | `torch.special.zeta` | inline 公式 / mixed | `SpecialZetaBackward0`、`SpecialZetaBackward1`、`SpecialZetaBackward2` | `special_zeta: other: grad * -self * special_zeta(self + 1., other)`；`special_zeta.self_scalar: other: grad * -self * special_zeta(self.toDouble() + 1., other)`；`self` 方向是 `not_implemented("zeta")` | `aten::special_zeta`、`aten::add.Scalar`、`aten::mul.Tensor`、`aten::neg` | 只有 `other` 有梯度公式；`self` / `other_scalar` 的 `self` 梯度都未实现 |
| 198 | `torch.substract` | 仓库中无此 API / schema | 无 | 无 | 无；若用户意图是 `torch.subtract`，则对应 `aten::sub.Tensor` / `aten::sub.Scalar` | 该项是拼写错误；真实存在的是 `torch.subtract`，其 backward 继承 `sub` 的导数规则 |
| 199 | `torch.svd` | CIA 前向分解到 `_linalg_svd`，再叠加 `mH` | `LinalgSvdBackward0`；`V = Vh.mH()` 还会产生 `TransposeBackward0`，复数时再叠加 `ConjBackward0` | `_linalg_svd: A: svd_backward(...)` | `aten::matmul`、`aten::mH`、`aten::transpose.int`、`aten::conj`、`aten::diagonal`、`aten::diag_embed`、`aten::narrow`、`aten::div.Tensor`、`aten::add.Tensor`、`aten::sub.Tensor`、`aten::mul.Tensor` | 公开 `aten::svd` 本身无专属 node；梯度主链挂在 `_linalg_svd` 上，返回 `V` 的那层 `mH` 另外贡献 view/conj backward |
| 200 | `torch.swapdims` | view backward（alias 到 `transpose.int`） | `TransposeBackward0` | `transpose.int: self: grad.transpose(dim0, dim1)` | `aten::transpose.int` | `swapdims` 只是 `transpose` 别名；前向底层 view 由 `transpose/as_strided` 构造，但反向直接按 `transpose.int` 公式回传 |

## 详细分析

### 196. torch.special.i1e

**反向来源类型**：helper 展开，无独立 backward schema

**Forward 路径**：
- `torch.special.i1e` → `aten::special_i1e`
- `aten::special_i1e` 是 structured op，前向 redispatch 到 backend `special_i1e_stub`

**derivatives.yaml 条目**：
```yaml
- name: special_i1e(Tensor self) -> Tensor
  self: i1e_backward(grad, self, result)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::special_i1e` 生成 `SpecialI1EBackward0`
- `Functions.cpp` 中 `SpecialI1EBackward0::apply()` 直接调用 `i1e_backward(grad, self, result)`

**helper 展开**（`FunctionsManual.cpp`）：
```cpp
auto self_is_not_tiny = self.abs() > eps;
auto safe_self = at::where(self_is_not_tiny, self, at::scalar_tensor(eps, self.options()));
auto gradx =
    (at::special_i0e(safe_self) -
     result * (safe_self.sgn() + safe_self.reciprocal()));
return grad *
    at::where(self_is_not_tiny, gradx, at::scalar_tensor(0.5, self.options()));
```

**反向 ATen 依赖**：
- `aten::special_i0e`
- `aten::sgn`
- `aten::reciprocal`
- `aten::where`
- `aten::abs`
- `aten::mul.Tensor`
- `aten::sub.Tensor`

**备注**：
- `i1e_backward` 是 manual helper，不是 `aten::i1e_backward` 这样的公开 schema。
- `where` 分支是为了避开 `x=0` 附近的数值问题；`x=0` 的梯度被手工设成 `0.5`。

---

### 197. torch.special.zeta

**反向来源类型**：inline 公式 / mixed

**前向 schema**：
- `torch.special.zeta(x: Tensor, y: Tensor)` → `aten::special_zeta`
- `torch.special.zeta(x: Scalar, y: Tensor)` → `aten::special_zeta.self_scalar`
- `torch.special.zeta(x: Tensor, y: Scalar)` → `aten::special_zeta.other_scalar`

其中标量重载在 `native_functions.yaml` 中通过 `CompositeExplicitAutograd` 包装到 Tensor-Tensor 主实现：
```cpp
Tensor special_zeta(const Scalar& x, const Tensor& y) {
  return at::special_zeta(wrapped_scalar_tensor(x), y);
}
```

**derivatives.yaml 条目**：
```yaml
- name: special_zeta(Tensor self, Tensor other) -> Tensor
  self: not_implemented("zeta")
  other:  grad * -self * special_zeta(self + 1., other)

- name: special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
  other:  grad * -self * special_zeta(self.toDouble() + 1., other)

- name: special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
  self: not_implemented("zeta")
```

**生成代码**：
- `special_zeta(Tensor, Tensor)` 生成 `SpecialZetaBackward0`
- `special_zeta.self_scalar` 生成 `SpecialZetaBackward1`
- `special_zeta.other_scalar` 生成 `SpecialZetaBackward2`
- `Functions.cpp` 中 `apply()` 与 `derivatives.yaml` 一致：只有 `other` 方向实际返回梯度

**反向 ATen 依赖**：
- `aten::special_zeta` — 递归调用同名 op 计算 `zeta(self + 1, other)`
- `aten::add.Scalar` — 构造 `self + 1.`
- `aten::mul.Tensor`
- `aten::neg` — Tensor-Tensor 路径中的 `-self`

**备注**：
- `self` 方向明确是 `not_implemented("zeta")`，不是“零梯度”。
- `self_scalar` 路径里 `-self` 和 `self.toDouble() + 1.` 都是标量运算，真正新增的 ATen hop 主要来自递归 `special_zeta(...)`。

---

### 198. torch.substract

**反向来源类型**：仓库中无此 API / schema

对应正向文档已经确认：
- 仓库中没有 `torch.substract` Python 入口
- `native_functions.yaml` 中也没有 `substract` schema

因此这里**没有可追踪的 backward 对象**。

如果用户真实意图是 `torch.subtract`，则它只是 `sub` 的别名：
- `torch.subtract` → `aten::subtract.Tensor` / `aten::subtract.Scalar`
- 再进入 `CompositeImplicitAutograd(native::subtract)` → `aten::sub.Tensor` / `aten::sub.Scalar`

此时反向会继承 `sub` 的导数规则：
```yaml
- name: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(-grad, alpha.conj()))
```

**备注**：
- 本文档对 198 仅记录“无实际 API”；不把 `torch.subtract` 当成同一接口替代写入主结论。

---

### 199. torch.svd

**反向来源类型**：`CompositeImplicitAutograd` 前向分解到 `_linalg_svd`，再叠加 `mH`

**Forward 路径**（`compute_uv=True`）：
- `torch.svd` → `aten::svd`
- `aten::svd`（CIA）→ `aten::linalg_svd`
- `aten::linalg_svd`（CIA）→ `aten::_linalg_svd`
- 返回 `(U, S, Vh.mH())`

`BatchLinearAlgebra.cpp` 中核心代码：
```cpp
std::tie(U, S, Vh) = at::linalg_svd(self, /*full_matrices=*/!some);
return std::make_tuple(std::move(U), std::move(S), Vh.mH());
```

**关键 derivative 条目**：
```yaml
- name: _linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
  A: "svd_backward(full_matrices && grad_U.defined() ? grad_U.narrow_symint(-1, 0, S.sym_size(-1)) : grad_U,
                   grad_S,
                   full_matrices && grad_Vh.defined() ? grad_Vh.narrow_symint(-2, 0, S.sym_size(-1)) : grad_Vh,
                   full_matrices ? U.narrow_symint(-1, 0, S.sym_size(-1)) : U,
                   S,
                   full_matrices ? Vh.narrow_symint(-2, 0, S.sym_size(-1)) : Vh)"
```

**生成代码**：
- `VariableTypeEverything.cpp` 只为 `aten::_linalg_svd` 生成 `LinalgSvdBackward0`
- `Functions.cpp` 中 `LinalgSvdBackward0::apply()` 直接调用 `svd_backward(...)`
- `V = Vh.mH()` 这层不是 `_linalg_svd` 内部公式的一部分，会额外引入：
  - `TransposeBackward0`
  - 复数时还会有 `ConjBackward0`

**`svd_backward` 展开依赖**（`FunctionsManual.cpp`）：
- `aten::matmul`
- `aten::mH`
- `aten::diagonal`
- `aten::diag_embed`
- `aten::narrow`
- `aten::unsqueeze`
- `aten::div.Tensor`
- `aten::mul.Tensor`
- `aten::add.Tensor`
- `aten::sub.Tensor`

其中 `mH` 自身在 `TensorShape.cpp` 中又会展开为：
- 实数：`transpose(-2, -1)`
- 复数：`transpose(-2, -1).conj()`

因此公开图层面常见的 backward 依赖可记为：
- `aten::matmul`
- `aten::mH`
- `aten::transpose.int`
- `aten::conj`
- `aten::diagonal`
- `aten::diag_embed`
- `aten::narrow`
- `aten::div.Tensor`
- `aten::add.Tensor`
- `aten::sub.Tensor`
- `aten::mul.Tensor`

**备注**：
- `aten::svd` 本身不生成独立 backward node；主节点是内层 `_linalg_svd` 的 `LinalgSvdBackward0`。
- `compute_uv=False` 的无 grad 主路径会走 `linalg_svdvals` + 零填充 `U/V`；但只要进入 autograd，上游仍会依赖 `_linalg_svd` 的可导路径，这也是 `derivatives.yaml` 特别说明 “We never call _linalg_svd with compute_uv=False in an autograd context” 的原因。

---

### 200. torch.swapdims

**反向来源类型**：view backward（alias 到 `transpose.int`）

**Forward 路径**：
- `torch.swapdims` → `aten::swapdims`
- `aten::swapdims`（CIA alias）→ `self.transpose(dim0, dim1)`
- `transpose.int` 前向最终构造 view

`TensorShape.cpp` 中实现非常直接：
```cpp
Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose(dim0, dim1);
}
```

**derivatives.yaml 条目**（真实挂点在 `transpose.int`）：
```yaml
- name: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
  self: grad.transpose(dim0, dim1)
```

**生成代码**：
- `swapdims` 自身没有独立 backward node
- `transpose.int` 在 `VariableTypeEverything.cpp` 中生成 `TransposeBackward0`
- `Functions.cpp` 中 `TransposeBackward0::apply()` 直接执行 `grad.transpose(dim0, dim1)`

**反向 ATen 依赖**：
- `aten::transpose.int`

**备注**：
- 这是标准 view 语义，但和 `view/reshape` 不同，`transpose` 已有显式 derivative 条目，所以公开 backward 依赖直接记 `aten::transpose.int` 即可。

