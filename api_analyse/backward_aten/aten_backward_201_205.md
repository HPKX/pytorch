# torch API 201-205 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 201 | `torch.tanhshrink` | Python 组合前向，无专属 backward | 无统一 node；由 `SubBackward0` 与 `TanhBackward0` 组合 | 无单独条目；继承 `tanh: tanh_backward(grad, result)` 与 `sub.Tensor` 公式 | `aten::tanh_backward`、`aten::sub.Tensor` | 前向就是 `input - input.tanh()`；没有独立 `aten::tanhshrink` schema |
| 202 | `torch.tensor_split` | CIA 前向分解为多个 `slice.Tensor` view | 无统一 node；每个切片通常对应 `SliceBackward0` | 无 `tensor_split.*` 条目；继承 `slice.Tensor: slice_backward_wrapper(...)` | `aten::slice_backward` | `.tensor_indices_or_sections` 只是再分发到 `.sections` 或 `.indices`；梯度按各个切片分别回传并自动累加 |
| 203 | `torch.tensordot` | CIA 前向分解，无专属 backward | 常见为 `MmBackward0` / `DotBackward0` / `SumBackward0/1` + `PermuteBackward0` / `ViewBackward0` | 无 `tensordot` 条目；继承内部 `mm`、`dot`、`sum`、`permute`、`view/reshape`、`mul` 等规则 | `aten::mm`、`aten::dot`、`aten::sum`、`aten::permute`、`aten::reshape`、`aten::squeeze`、`aten::mul.Tensor` | 是否走 `mm` 还是 `dot` / `mul+sum` 取决于收缩形态、连续性和设备 |
| 204 | `torch.trapezoid` | CIA 前向分解，无专属 backward | 无统一 node；由 `SliceBackward0`、`SubBackward0`、`AddBackward0`、`MulBackward0`、`SumBackward0/1`、`DivBackward0` 等组合 | 无 `trapezoid.*` 条目 | `aten::slice_backward`、`aten::sub.Tensor`、`aten::add.Tensor`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::sum`、`aten::div.Scalar`、`aten::view` | `x` 版会先构造 `dx = x_right - x_left`；`dx` 标量版用化简公式，额外含 `select.int` 的回传 |
| 205 | `torch.tril_indices` | factory op，不参与可导 Tensor→Tensor 链 | 无 | 无 | 无 | 没有 Tensor 输入；`VariableTypeEverything.cpp` 对该 schema 注册的是 `autogradNotImplementedFallback()`，返回结果是普通 leaf tensor |

## 详细分析

### 201. torch.tanhshrink

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**（`torch/nn/functional.py`）：
```python
return input - input.tanh()
```

因此 backward 完全由两个基础 ATen 子算子承担：

| 子 op | backward 来源 | 反向依赖 |
| --- | --- | --- |
| `aten::tanh` | `TanhBackward0` | `aten::tanh_backward` |
| `aten::sub.Tensor` | `SubBackward0` | `aten::sub.Tensor` |

**相关 derivative 条目**：
```yaml
- name: tanh(Tensor self) -> Tensor
  self: tanh_backward(grad, result)

- name: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(-grad, alpha.conj()))
```

**反向 ATen 依赖**：
- `aten::tanh_backward`
- `aten::sub.Tensor`

**备注**：
- 仓库里没有单独的 `aten::tanhshrink` 前向 schema，因此也不存在统一的 `TanhshrinkBackward` node。

---

### 202. torch.tensor_split

**反向来源类型**：`CompositeImplicitAutograd` 前向分解为多个 `slice.Tensor` view

**前向 schema**：
- `aten::tensor_split.sections`
- `aten::tensor_split.indices`
- `aten::tensor_split.tensor_indices_or_sections`

其中：
- `.sections` 逐段调用 `at::slice_symint(self, dim, start, end)`
- `.indices` 逐段调用 `at::symint::slice(self, dim, start, end)`
- `.tensor_indices_or_sections` 只是把 0-D / 1-D CPU `long` Tensor 转成前两种重载后再重入

`TensorShape.cpp` 的核心模式：
```cpp
splits[split_idx] =
    at::slice_symint(self, dim_, start_idx, start_idx + split_size);
```

**derivatives.yaml** 没有 `tensor_split.*` 条目，因此 autograd 直接记录内部 `slice.Tensor`：
```yaml
- name: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)
```

**生成代码**：
- `tensor_split.*` 自身无专属 backward node
- 每个切片输出通常对应 `SliceBackward0`
- `Functions.cpp` 中 `SliceBackward0::apply()` 调用 `slice_backward_wrapper(...)`
- `slice_backward_wrapper` 再转到 `aten::slice_backward`

**反向 ATen 依赖**：
- `aten::slice_backward`

**备注**：
- `tensor_split` 返回的是 Tensor 列表；反向时每个切片各自把梯度 scatter 回原张量对应区间，autograd 会把这些贡献自动累加。
- `.tensor_indices_or_sections` 只是 wrapper，不单独贡献新的 backward 机制。

---

### 203. torch.tensordot

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 路径**（`Linear.cpp`）：
- 可能先对广播收缩维做 `sum`
- 再做 `permute`
- 再按分支：
  - 一般路径：`reshape` → `mm` → `reshape`
  - 全缩并且 contiguous：`permute` → `flatten`(view) → `dot` → `reshape`
  - 全缩并但非连续：`permute` → `squeeze` → `mul.Tensor` → `sum` → `reshape`

核心实现：
```cpp
if ((t1.device().type() == at::kMPS || t2.device().type() == at::kMPS) || size1 != 1 || size2 != 1) {
  t1 = t1.permute(p1).reshape_symint({size1, csize});
  t2 = t2.permute(p2).reshape_symint({csize, size2});
  return at::mm(t1, t2).reshape_symint(rsizes);
} else {
  t1 = t1.permute(p1);
  t2 = t2.permute(p2);
  if (t1.is_contiguous() && t2.is_contiguous()) {
    return at::dot(t1.flatten(), t2.flatten()).reshape_symint(rsizes);
  } else {
    return (t1.squeeze() * t2.squeeze()).sum(t1.scalar_type()).reshape_symint(rsizes);
  }
}
```

**关键内部子 op 的 backward**：
- `aten::mm` → `MmBackward0`
  - 展开到 `mm_mat1_backward` / `mm_mat2_backward`
  - helper 内部进一步依赖 `mm`、`t()` / `transpose`、`conj`
- `aten::dot` → `DotBackward0`
  - `self: grad * tensor.conj()`
  - `tensor: grad * self.conj()`
- `aten::sum` / `aten::sum.dim_IntList`
- `aten::permute`
- `aten::reshape` / `flatten` / `view`
- `aten::squeeze`
- `aten::mul.Tensor`

**反向 ATen 依赖**：
- `aten::mm`
- `aten::dot`
- `aten::sum`
- `aten::permute`
- `aten::reshape`
- `aten::squeeze`
- `aten::mul.Tensor`

**备注**：
- `tensordot` 没有统一 backward node，实际图取决于命中 `mm`、`dot` 还是 `mul+sum` 分支。
- `reshape` / `flatten` 这类 view 在反向里通常体现为 `ViewBackward0`，本质是把梯度 reshape 回原形状。

---

### 204. torch.trapezoid

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**前向公式**（`Integration.cpp`）：

`x` Tensor 版本：
```cpp
Tensor left = y.slice(dim, 0, -1);
Tensor right = y.slice(dim, 1);
return ((left + right) * dx).sum(dim) / 2.;
```
其中 `dx = x_right - x_left`，而 `x_right/x_left` 也来自 `slice`。

常量 `dx` 版本：
```cpp
return (y.sum(dim) - (y.select(dim, 0) + y.select(dim, -1)) * 0.5) * dx;
```

若 `x.dim() == 1` 或 `x.dim() < y.dim()`，前向还会先做一层 `view` 对齐维度。

**因为 `trapezoid.x` / `trapezoid.dx` 在 `derivatives.yaml` 中都无条目，autograd 直接继承内部子图**：

| 子 op | backward 来源 | 主要反向依赖 |
| --- | --- | --- |
| `aten::slice.Tensor` | `SliceBackward0` | `aten::slice_backward` |
| `aten::sub.Tensor` | `SubBackward0` | `aten::sub.Tensor` |
| `aten::add.Tensor` | `AddBackward0` | `aten::add.Tensor` |
| `aten::mul.Tensor` / `aten::mul.Scalar` | `MulBackward0/1` | `aten::mul.Tensor` / `aten::mul.Scalar` |
| `aten::sum` / `aten::sum.dim_IntList` | `SumBackward0/1` | `aten::sum` |
| `aten::div.Scalar` | `DivBackward0` | `aten::div.Scalar` |
| `aten::view` | `ViewBackward0` | reshape/view 回传 |

**反向 ATen 依赖**：
- `aten::slice_backward`
- `aten::sub.Tensor`
- `aten::add.Tensor`
- `aten::mul.Tensor`
- `aten::mul.Scalar`
- `aten::sum`
- `aten::div.Scalar`
- `aten::view`

**备注**：
- `x` 版和 `dx` 版的 backward 结构不同；`x` 版额外有 `dx = x_right - x_left` 这条链。
- 常量 `dx` 版中的 `select.int` 也会各自把梯度散回 `y` 的首尾位置，但公开总结里记录为该 composite 主体上的直接算子即可。

---

### 205. torch.tril_indices

**反向来源类型**：factory op，不参与可导 Tensor→Tensor 链

**前向 schema**：
```yaml
- func: tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
```

这个接口没有 Tensor 输入，只根据整数参数和 `TensorOptions` 创建结果张量。对应生成代码中：
- Python 绑定可以接收 `requires_grad=False/True`
- 但 `VariableTypeEverything.cpp` 对 `aten::tril_indices` 注册的是 `autogradNotImplementedFallback()`

**结论**：
- 没有 backward node
- 没有 `derivatives.yaml` 条目
- 没有可追踪的 backward ATen 依赖

**备注**：
- 即使 Python API 允许传 `requires_grad=True`，返回值也只是一个没有 `grad_fn` 的 leaf tensor，因为前向根本没有可导 Tensor 输入。

