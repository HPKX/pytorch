# torch API 41-45 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | `torch.fmax` | inline 公式 | `FmaxBackward0` | `self: grad.masked_fill((self >= other).logical_or_(other.isnan()).logical_not_(), 0)`；`other: grad.masked_fill((self >= other).logical_or_(other.isnan()), 0)` | `aten::ge.Tensor`、`aten::isnan`、`aten::logical_or_`、`aten::logical_not_`、`aten::masked_fill_.Scalar` | `fmax` 忽略 NaN 取最大值；梯度通过 mask 选择传递给 self 或 other |
| 42 | `torch.fmod` | inline 公式 | `FmodBackward0`（Scalar）/ `FmodBackward1`（Tensor） | Scalar: `self: grad`；Tensor: `self: grad`；`other: -grad * self.div(other, "trunc")` | Scalar 重载：无额外 ATen op；Tensor 重载：`aten::div.Tensor_mode`、`aten::mul.Tensor`、`aten::neg` | Scalar 重载的 self 梯度直接传递 grad；Tensor 重载的 other 梯度需要 trunc 除法 |
| 43 | `torch.gather` | backward ATen op（`gather_backward`） | `GatherBackward0` | `self: gather_backward(grad, self, dim, index, sparse_grad)`；`index: non_differentiable` | `sparse_grad=False`：`aten::new_zeros_symint`（→`aten::zeros`）、`aten::scatter_add_`/`aten::scatter_add`；`sparse_grad=True`：`aten::_gather_sparse_backward` | `gather_backward` 是 `aten::` 注册的 native op（`TensorAdvancedIndexing.cpp:2085`），非 FunctionsManual.cpp helper |
| 44 | `torch.gcd` | 不可微 | 无 | 无 derivatives.yaml 条目 | 无 | 整数专用 op；`autogradNotImplementedFallback`；输出无 autograd 历史 |
| 45 | `torch.ge` | 不可微 | 无 | `output_differentiability: [False]` | 无 | 比较运算符，返回 bool 张量；`output_differentiability: [False]` 标记输出不可微 |

## 详细分析

### 41. torch.fmax

**反向来源类型**：inline 公式

**Forward 路径**：`aten::fmax` → Autograd wrapper（`FmaxBackward0`）→ CPU/CUDA structured wrapper → `fmax_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: fmax(Tensor self, Tensor other) -> Tensor
  self: grad.masked_fill((self >= other).logical_or_(other.isnan()).logical_not_(), 0)
  other: grad.masked_fill((self >= other).logical_or_(other.isnan()), 0)
  result: other_t + (self_p > other_p).logical_or_(other_p.isnan()) * (self_t - other_t)
```

**Backward Node**：`FmaxBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
// self 梯度：当 self >= other 或 other 是 NaN 时传递梯度
auto grad_result = grad.masked_fill(
    (self >= other).logical_or_(other.isnan()).logical_not_(), 0);
// other 梯度：当 self < other 且 other 不是 NaN 时传递梯度
auto grad_result = grad.masked_fill(
    (self >= other).logical_or_(other.isnan()), 0);
```

**反向 ATen 依赖**：
- `aten::ge.Tensor` — `self >= other`
- `aten::isnan` — `other.isnan()`
- `aten::logical_or_` — 原地逻辑或
- `aten::logical_not_` — 原地逻辑非（仅 self 梯度路径）
- `aten::masked_fill_.Scalar` — 根据 mask 填零

---

### 42. torch.fmod

**反向来源类型**：inline 公式

**Forward 路径**：
- Scalar 重载：`aten::fmod.Scalar` → Autograd wrapper（`FmodBackward0`）→ `CompositeExplicitAutograd` → 转为 Tensor 重载 `aten::fmod.Tensor` → structured wrapper
- Tensor 重载：`aten::fmod.Tensor` → Autograd wrapper（`FmodBackward1`）→ structured wrapper → `fmod_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: fmod.Scalar(Tensor self, Scalar other) -> Tensor
  self: grad
  result: auto_element_wise

- name: fmod.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad
  other: -grad * self.div(other, /*rounding_mode=*/"trunc")
  result: self_t - other_t * self_p.div(other_p, /*rounding_mode=*/"trunc")
```

**Backward Node**：`FmodBackward0`（Scalar 重载）/ `FmodBackward1`（Tensor 重载）

**生成代码**（`Functions.cpp`）：

Scalar 重载（`FmodBackward0`）：
```cpp
// self 梯度：直接传递 grad
auto grad_result = any_grad_defined ? (grad) : Tensor();
```

Tensor 重载（`FmodBackward1`）：
```cpp
// self 梯度：直接传递 grad
auto grad_result = any_grad_defined ? (grad) : Tensor();
// other 梯度
auto grad_result = any_grad_defined ? (-grad * self.div(other, "trunc")) : Tensor();
```

**反向 ATen 依赖**：

Scalar 重载：
- 无额外 ATen op（self 梯度直接传递 grad）

Tensor 重载：
- `aten::div.Tensor_mode` — `self.div(other, "trunc")`（带 rounding mode 的除法）
- `aten::mul.Tensor` — `grad * ...`
- `aten::neg` — 取负

---

### 43. torch.gather

**反向来源类型**：backward ATen op（`gather_backward`）

**Forward 路径**：`aten::gather` → Autograd wrapper（`GatherBackward0`）→ structured wrapper → `gather_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
  self: gather_backward(grad, self, dim, index, sparse_grad)
  index: non_differentiable
  result: auto_linear
```

**Backward Node**：`GatherBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = any_grad_defined ? (gather_backward(grad, self, dim, index, sparse_grad)) : Tensor();
```

**`gather_backward` 实现**（`TensorAdvancedIndexing.cpp:2085`）：
```cpp
Tensor gather_backward(const Tensor& grad, const Tensor& self,
    int64_t dim, const Tensor& index, bool sparse_grad) {
  if (sparse_grad) {
    return at::_gather_sparse_backward(self, dim, index, grad);
  }
  auto result = grad.new_zeros_symint(self.sym_sizes());
  if (areAnyTensorSubclassLike({index, grad})) {
    return result.scatter_add(dim, index, grad);      // out-of-place
  }
  result.scatter_add_(dim, index, grad);              // inplace
  return result;
}
```

**反向 ATen 依赖**（`sparse_grad=False`，默认路径）：
- `aten::new_zeros_symint`（→ `aten::zeros`）— 创建与 self 同形状的零张量
- `aten::scatter_add_` — 将 grad 按 index 原地散布到零张量（inplace）
- `aten::scatter_add` — 当 index/grad 是 Tensor Subclass 时使用 out-of-place 版本

**反向 ATen 依赖**（`sparse_grad=True`）：
- `aten::_gather_sparse_backward` — 稀疏梯度专用 backward

---

### 44. torch.gcd

**反向来源类型**：不可微

**Forward 路径**：`aten::gcd` → structured wrapper（`gcd.out`）→ `gcd_stub` → backend kernel

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp**：
```cpp
m.impl("gcd", torch::autograd::autogradNotImplementedFallback());
m.impl("gcd_", torch::autograd::autogradNotImplementedFallback());
m.impl("gcd.out", torch::autograd::autogradNotImplementedFallback());
```

`gcd` 是整数专用操作（计算最大公约数），不存在有意义的梯度。autograd 使用 `autogradNotImplementedFallback` 直接 redispatch，不建立 backward node，输出无 autograd 历史。

**反向 ATen 依赖**：无。

---

### 45. torch.ge

**反向来源类型**：不可微

**Forward 路径**：`aten::ge.Tensor` / `aten::ge.Scalar` → Autograd wrapper → structured wrapper（`ge_Tensor_out` / `ge_Scalar_out`）→ backend kernel

**derivatives.yaml 条目**：
```yaml
- name: ge.Scalar(Tensor self, Scalar other) -> Tensor
  output_differentiability: [False]

- name: ge.Tensor(Tensor self, Tensor other) -> Tensor
  output_differentiability: [False]
```

`ge` 是比较运算符，返回 bool 张量。`output_differentiability: [False]` 标记输出不可微，autograd 不会为输出设置 `requires_grad=True`，也不会建立梯度传播路径。

注意：`ge_` inplace 变体在 derivatives.yaml 中有条目 `self: zeros_like(self)`，表示对 self 的梯度为零。

**反向 ATen 依赖**：无。
