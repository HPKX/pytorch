# torch API 76-80 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 76 | `torch.lu_unpack` | helper 函数 | `LuUnpackBackward0` | `LU_data: lu_unpack_backward(grad_L, grad_U, LU_data.sym_size(-2), LU_data.sym_size(-1))` | `aten::tril`、`aten::triu`、`aten::narrow_symint`、`aten::add.Tensor`、`aten::cat`、`aten::zeros_symint` | P 不可微（`output_differentiability: [False, True, True]`）；LU_pivots 标记 `non_differentiable` |
| 77 | `torch.moveaxis` | CIA 前向分解 → `aten::movedim` → `aten::permute` | 无（子 op `PermuteBacward0`） | 无条目（CIA，反向挂在 `aten::permute` 上） | `aten::permute`（`permute_backwards` helper） | `moveaxis` 是 `movedim` 的 CIA 别名，二者都 CIA 分解为 `aten::permute` |
| 78 | `torch.movedim` | CIA 前向分解 → `aten::permute` | 无（子 op `PermuteBackward0`） | 无条目（CIA，反向挂在 `aten::permute` 上） | `aten::permute`（`permute_backwards` helper） | CIA 分解为 `aten::permute`；autograd 录制 `PermuteBackward0` |
| 79 | `torch.msort` | CIA 前向分解 → `aten::sort` | 无（子 op `SortBackward0` / `SortBackward1`） | 无条目（CIA，反向挂在 `aten::sort.stable` 上）；sort.stable 公式：`self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), true)` | `aten::zeros_symint`、`aten::scatter_` / `aten::scatter`、`aten::unsqueeze`（条件） | `msort` CIA 分解为 `sort(self, 0, false)` → `sort.stable` |
| 80 | `torch.multiply` | CIA 前向分解 → `aten::mul` 的 inline+helper | 无（子 op `MulBackward0`/`MulBackward1`） | Tensor 重载：`self: mul_tensor_backward(grad, other, self.scalar_type())`；`other: mul_tensor_backward(grad, self, other.scalar_type())`；Scalar 重载：`self: mul_tensor_backward(grad, other, self.scalar_type())` | `aten::mul.Tensor`、`aten::conj`；复数→实数时额外 `aten::real` | `multiply` 是 `mul` 的 CIA 别名；backward 挂在 `mul.Tensor`/`mul.Scalar` 上 |

## 详细分析

### 76. torch.lu_unpack

**反向来源类型**：helper 函数

**Forward 路径**：`aten::lu_unpack` → structured wrapper（`meta + impl`）→ `lu_unpack_out`

**derivatives.yaml 条目**：
```yaml
- name: lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
  LU_data: lu_unpack_backward(grad_L, grad_U, LU_data.sym_size(-2), LU_data.sym_size(-1))
  LU_pivots: non_differentiable
  output_differentiability: [False, True, True]
```

**Backward Node**：`LuUnpackBackward0`

**`lu_unpack_backward` 实现**（`FunctionsManual.cpp:6146`）：
```cpp
Tensor lu_unpack_backward(
    const Tensor& L_grad, const Tensor& U_grad,
    const c10::SymInt& m, const c10::SymInt& n) {
  if (!L_grad.defined() && !U_grad.defined()) return {};
  const auto k = std::min(m, n);

  // m == n: L_grad.tril(-1) + U_grad.triu()
  // m != n: cat(get_L1(L_grad) + get_U1(U_grad), get_L2/U2)
  // get_L1: m == k ? L.tril(-1) : L.narrow_symint(-2, 0, k).tril(-1)
  // get_U1: n == k ? U.triu() : U.narrow_symint(-1, 0, k).triu()
  // 单边定义时使用 tril(-1)/triu() + 可能的 cat + zeros_symint
}
```

**反向 ATen 依赖**：
- `aten::tril` — 提取下三角（对角线下移 -1）
- `aten::triu` — 提取上三角
- `aten::narrow_symint` — 非方阵时切片
- `aten::add.Tensor` — 合并 L 和 U 梯度
- `aten::cat` — 非方阵时拼接主部分与补部分
- `aten::zeros_symint` — 单边定义时填充零块

---

### 77. torch.moveaxis

**反向来源类型**：CIA 前向分解，无专属 backward

**Forward 分解路径**：`aten::moveaxis` (CIA) → `aten::movedim` (CIA) → `aten::permute`

**无 derivatives.yaml 条目**。`moveaxis` 和 `movedim` 都是 CIA，最终分解为 `aten::permute`，autograd 录制 `PermuteBackward0`。

**`permute` 的 derivatives.yaml 条目**：
```yaml
- name: permute(Tensor(a) self, int[] dims) -> Tensor(a)
  self: permute_backwards(grad, dims)
  result: auto_linear
```

**`permute_backwards` 实现**（`FunctionsManual.cpp:646`）：
```cpp
Tensor permute_backwards(const Tensor& grad, IntArrayRef fwd_dims) {
  auto ndims = fwd_dims.size();
  std::vector<int64_t> dims(ndims);
  for (const auto i : c10::irange(ndims))
    dims[at::maybe_wrap_dim(fwd_dims[i], ndims)] = i;
  return grad.permute(dims);  // aten::permute
}
```

**反向 ATen 依赖**：
- `aten::permute` — 逆排列梯度

---

### 78. torch.movedim

**反向来源类型**：CIA 前向分解，无专属 backward

**Forward 分解路径**：`aten::movedim` (CIA) → 计算排列顺序 → `aten::permute`

与 `moveaxis` 完全相同。autograd 录制 `PermuteBackward0`，反向调用 `permute_backwards`。

**反向 ATen 依赖**：
- `aten::permute` — 逆排列梯度

---

### 79. torch.msort

**反向来源类型**：CIA 前向分解 → `aten::sort` → `aten::sort.stable`

**Forward 分解路径**：`aten::msort` (CIA) → `std::get<0>(aten::sort(self, 0, false))` → `aten::sort.stable(self, stable=false, dim=0, descending=false)`

**无 `msort` derivatives.yaml 条目**。反向挂在 `sort.stable` 上。

**`sort.stable` 的 derivatives.yaml 条目**：
```yaml
- name: sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), true)
  output_differentiability: [True, False]
```

**Backward Node**：`SortBackward1`

**`value_selecting_reduction_backward_symint` 实现**（`ReduceOps.cpp:2350`）：
```cpp
Tensor value_selecting_reduction_backward_symint(
    const Tensor& grad, int64_t dim, const Tensor& indices,
    c10::SymIntArrayRef sizes, bool keepdim) {
  // keepdim=true 时直接 scatter
  auto grad_in = at::zeros_symint(sizes, grad.options());
  return grad_in.scatter_(dim, indices, grad);  // 或 scatter（subclass 场景）
}
```

**反向 ATen 依赖**：
- `aten::zeros_symint` — 创建全零梯度张量
- `aten::scatter_` / `aten::scatter` — 将梯度按索引散布回原位置

---

### 80. torch.multiply

**反向来源类型**：CIA 前向分解 → `aten::mul` 的 inline + helper

**Forward 分解路径**：
- Tensor 重载：`aten::multiply.Tensor` (CIA) → `aten::mul.Tensor`
- Scalar 重载：`aten::multiply.Scalar` (CIA) → `aten::mul.Scalar`

`multiply` 是 `mul` 的 CIA 别名，backward 挂在 `mul.Tensor`/`mul.Scalar` 上。

**`mul.Tensor` 的 derivatives.yaml 条目**：
```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
```

**`mul.Scalar` 的 derivatives.yaml 条目**：
```yaml
- name: mul.Scalar(Tensor self, Scalar other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
```

**Backward Node**：`MulBackward0`（Tensor 重载）/ `MulBackward1`（Scalar 重载）

**`mul_tensor_backward` 实现**（`FunctionsManual.cpp:602`）：
```cpp
template <typename T>
Tensor mul_tensor_backward(const Tensor& grad, T other, ScalarType self_st) {
  auto out = grad * other.conj();         // aten::mul.Tensor / aten::mul.Scalar + aten::conj
  return handle_r_to_c(self_st, std::move(out));
}
```

**反向 ATen 依赖**：
- `aten::mul.Tensor` — `grad * other.conj()`（Tensor 重载）
- `aten::mul.Scalar` — `grad * other`（Scalar 重载，`conj()` 对实数 Scalar 无效）
- `aten::conj` — 复数共轭（实数时为 no-op view）

**条件依赖**：`handle_r_to_c` 在复数梯度 + 实数输入场景调用 `aten::real`。
