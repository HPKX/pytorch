# torch API 51-55 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 51 | `torch.hsplit` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::zeros`、`aten::slice.Tensor`、`aten::copy_` | CIA 分解到 `tensor_split` / `slice` 后，递归展开最终落到 `slice_backward` 的基础 ATen op |
| 52 | `torch.hstack` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::narrow`、`aten::squeeze.dim`、`aten::zeros`、`aten::real` | `cat` 负责主梯度切分，`atleast_1d` 引入的 view 反向再经 `squeeze.dim`；复数到实数时条件性经过 `real` |
| 53 | `torch.hypot` | inline 公式 | `HypotBackward0` | `self: grad * self / result`；`other: grad * other / result` | `aten::mul.Tensor`、`aten::div.Tensor` | sqrt(self^2 + other^2) 的导数；链式法则直接给出 |
| 54 | `torch.i0` | inline 公式 | `I0Backward0` | `self: grad * special_i1(self)` | `aten::mul.Tensor`、`aten::special_i1` | I0 的导数为 I1（第一类一阶修正贝塞尔函数） |
| 55 | `torch.igamma` | inline 公式（other 参数）+ not_implemented（self 参数） | `IgammaBackward0` | `self: not_implemented("igamma: input")`；`other: grad * exp((self - 1) * log(other) - other - lgamma(self))` | `aten::sub.Scalar`、`aten::log`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::lgamma`、`aten::exp` | self 梯度未实现（调用时抛异常）；other 梯度基于不完全 Gamma 函数的解析导数 |

## 详细分析

### 51. torch.hsplit

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- `aten::hsplit.int` → CIA → `at::native::hsplit` → `at::tensor_split(self, sections, dim)`（1-D 时 dim=0，否则 dim=1）→ CIA → `tensor_split_sections_symint` → 循环调用 `at::slice_symint(self, dim, start, end)` → `aten::slice.Tensor`
- `aten::hsplit.array` → CIA → `at::native::hsplit` → `at::tensor_split(self, indices, dim)` → CIA → `tensor_split_indices_symint` → 循环调用 `at::slice_symint` → `aten::slice.Tensor`

**无 derivatives.yaml 条目**（`hsplit`、`tensor_split` 均无条目）。autograd 引擎穿透到底层 `slice.Tensor` 视图操作。

`slice.Tensor` 的 derivatives.yaml 条目：
```yaml
- name: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)
  result: auto_linear
```

`slice_backward_wrapper` 内部创建 `aten::zeros` 并将 grad 通过 `aten::slice.Tensor` + `aten::copy_` 填入对应位置。

**反向 ATen 依赖**（来自 `slice` 的 backward）：
- `aten::zeros` — 创建全零张量
- `aten::slice.Tensor` — 创建 slice 视图
- `aten::copy_` — 将 grad 填入对应 slice

---

### 52. torch.hstack

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
`aten::hstack` → CIA → `at::native::hstack` → `aten::atleast_1d.Sequence`（CIA → 对 0-D 输入做 `unsqueeze(0)` / `reshape`）→ `aten::cat`（1-D 输入时 dim=0，否则 dim=1）→ structured backend kernel

**无 `hstack` 的 derivatives.yaml 条目**。autograd 穿透到 `cat` 和各 `unsqueeze`/view 子 op。

`cat` 的 derivatives.yaml 条目：
```yaml
- name: cat(Tensor[] tensors, int dim=0) -> Tensor
  tensors: cat_tensors_backward(grad, to_args_sizes_symint(tensors), to_args_scalartypes(tensors), dim)
  result: cat_jvp(tensors, dim)
```

**`cat_tensors_backward` 实现**（`FunctionsManual.cpp:1058`）：
```cpp
std::vector<Tensor> cat_tensors_backward(const Tensor& grad,
    const std::vector<std::vector<c10::SymInt>>& sizes,
    const std::vector<ScalarType>& dtypes, int64_t dim) {
  for (const auto i : c10::irange(sizes.size())) {
    grad_inputs[i] = grad_val.narrow_symint(dim, accumulate - size, size);  // aten::narrow
  }
  return grad_inputs;
}
```

**反向 ATen 依赖**（来自 `cat` 的 backward）：
- `aten::narrow`（→ `aten::slice.Tensor`）— 从 grad 中按 dim 切分出各输入的梯度
- `aten::zeros` — 空输入返回空梯度

**各 view op 的 backward**：
- `unsqueeze` → backward: `aten::squeeze`（`grad.squeeze(dim)`）

**条件依赖**：
- `aten::real` — 复数 grad + 实数输入场景

---

### 53. torch.hypot

**反向来源类型**：inline 公式

**Forward 路径**：`aten::hypot` → Autograd wrapper（`HypotBackward0`）→ structured wrapper → `hypot_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: hypot(Tensor self, Tensor other) -> Tensor
  self: grad * self / result
  other: grad * other / result
  result: self_t * self_p / result + other_t * other_p / result
```

**Backward Node**：`HypotBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
// self 梯度
auto grad_result = any_grad_defined ? (grad * self / result) : Tensor();
// other 梯度
auto grad_result = any_grad_defined ? (grad * other / result) : Tensor();
```

`hypot(self, other) = sqrt(self^2 + other^2)`，导数为：
- d/d(self) = self / sqrt(self^2 + other^2) = self / result
- d/d(other) = other / sqrt(self^2 + other^2) = other / result

**反向 ATen 依赖**：
- `aten::mul.Tensor` — `grad * self`、`grad * other`
- `aten::div.Tensor` — `... / result`

---

### 54. torch.i0

**反向来源类型**：inline 公式

**Forward 路径**：`aten::i0` → Autograd wrapper（`I0Backward0`）→ structured wrapper → `i0_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: i0(Tensor self) -> Tensor
  self: grad * at::special_i1(self)
  result: auto_element_wise
```

**Backward Node**：`I0Backward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = any_grad_defined ? (grad * at::special_i1(self)) : Tensor();
```

I0（第一类零阶修正贝塞尔函数）的导数为 I1（第一类一阶修正贝塞尔函数）：`d/dx I0(x) = I1(x)`。

**反向 ATen 依赖**：
- `aten::special_i1` — 计算 I1(self)
- `aten::mul.Tensor` — `grad * I1_result`

---

### 55. torch.igamma

**反向来源类型**：inline 公式（other 参数）+ `not_implemented`（self 参数）

**Forward 路径**：`aten::igamma` → Autograd wrapper（`IgammaBackward0`）→ structured wrapper → `igamma_stub` → backend kernel

**derivatives.yaml 条目**：
```yaml
- name: igamma(Tensor self, Tensor other) -> Tensor
  self: 'not_implemented("igamma: input")'
  other: grad * exp((self - 1) * log(other) - other - lgamma(self))
```

**Backward Node**：`IgammaBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
// self 梯度：未实现，调用时抛异常
auto grad_result = not_implemented("igamma: input");
// other 梯度
auto grad_result = any_grad_defined
    ? (grad * exp((self - 1) * log(other) - other - lgamma(self)))
    : Tensor();
```

正则化下不完全 Gamma 函数 `P(a, x) = igamma(a, x)` 对 x（即 other）的导数为：
`dP/dx = x^(a-1) * exp(-x) / Gamma(a) = exp((a-1)*log(x) - x - lgamma(a))`

对 a（即 self）的导数涉及 digamma 和积分，PyTorch 尚未实现。

**反向 ATen 依赖**（other 梯度路径）：
- `aten::sub.Scalar` — `self - 1`
- `aten::log` — `log(other)`
- `aten::mul.Tensor` — `(self - 1) * log(other)`
- `aten::sub.Tensor` — `... - other`、`... - lgamma(self)`
- `aten::lgamma` — `lgamma(self)`
- `aten::exp` — `exp(...)`
- `aten::mul.Tensor` — `grad * exp_result`

**注意**：`torch.special.gammainc` 是 `torch.igamma` 的别名，共享同一 backward 逻辑。
