# torch API 46-50 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 46 | `torch.geqrf` | not_implemented | `GeqrfBackward0` | `self: not_implemented("geqrf")` | 无（调用 backward 时抛异常） | derivatives.yaml 标记为 `not_implemented`；调用 backward 会抛出 RuntimeError；推荐使用 `torch.linalg.qr` |
| 47 | `torch.ger` | CIA 前向分解 → `aten::outer` → `aten::mul` 的 inline 公式 | `MulBackward0`（挂在底层 `mul.Tensor` 上） | 无 `ger` 专属条目；底层 `mul.Tensor`: `self: …mul_tensor_backward(grad, other, …)`；`other: …mul_tensor_backward(grad, self, …)` | `aten::mul.Tensor`、`aten::conj`（复数时） | `ger` 是 CIA 别名→ `outer` → `reshape` + `mul.Tensor`；autograd 穿透到各子 op |
| 48 | `torch.hamming_window` | 不可微（工厂函数） | 无 | 无 derivatives.yaml 条目 | 无 | 窗函数工厂方法，无 Tensor 输入；CEA；输出不带 autograd 历史 |
| 49 | `torch.hann_window` | 不可微（工厂函数） | 无 | 无 derivatives.yaml 条目 | 无 | 窗函数工厂方法，无 Tensor 输入；CEA；输出不带 autograd 历史 |
| 50 | `torch.heaviside` | 不可微 | 无 | 无 derivatives.yaml 条目 | 无 | `autogradNotImplementedFallback`；Heaviside 阶跃函数在不连续点无定义导数 |

## 详细分析

### 46. torch.geqrf

**反向来源类型**：`not_implemented`（有 backward node 但调用时抛异常）

**Forward 路径**：`aten::geqrf` → Autograd wrapper（`GeqrfBackward0`）→ `at::native::geqrf` → `aten::empty`（创建输出）→ `aten::geqrf.a` → `geqrf_out_helper` → `geqrf_stub` → CPU LAPACK / CUDA cuSOLVER kernel

**derivatives.yaml 条目**：
```yaml
- name: geqrf(Tensor self) -> (Tensor a, Tensor tau)
  self: not_implemented("geqrf")
```

**Backward Node**：`GeqrfBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = not_implemented("geqrf");
```

`not_implemented` 函数在被调用时抛出 `RuntimeError`，提示该操作的梯度尚未实现。

虽然 `geqrf` 有 autograd 注册（会建立 backward node），但实际 backward 计算会失败。若需要 QR 分解的梯度，应使用 `torch.linalg.qr`。

**反向 ATen 依赖**：无（backward 抛异常，不执行任何 ATen op）。

---

### 47. torch.ger

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
`aten::ger` → CIA → `at::native::ger` → `self.outer(vec2)` → `aten::outer` → CIA → `at::native::outer` → `aten::reshape`（将 self 变为列向量）→ `aten::reshape`（将 vec2 变为行向量）→ `aten::mul.Tensor`

**无 `ger` 的 derivatives.yaml 条目**。autograd 穿透到 `reshape` 和 `mul.Tensor`。

`mul.Tensor` 的 derivatives.yaml 条目（梯度的核心来源）：
```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
```

`reshape` 是 view 操作，backward 为对应的逆 reshape。

**反向 ATen 依赖**（来自各子 op 的 backward 组合）：
- `aten::mul.Tensor` — `mul_tensor_backward` 内部调用
- `aten::conj` — 复数共轭（`mul_tensor_backward` 内部，复数时）
- `aten::reshape` — `reshape` 的逆操作（view backward）

---

### 48. torch.hamming_window

**反向来源类型**：不可微（工厂函数）

**Forward 路径**：`aten::hamming_window`（及其重载）→ `CompositeExplicitAutograd` → 窗口构造逻辑 → `aten::arange` → `aten::mul_.Scalar` → `aten::cos_` → `aten::mul_.Scalar` → `aten::add_.Scalar` → （periodic 时）`aten::narrow`

**derivatives.yaml**：无条目。

`hamming_window` 是窗函数工厂方法，所有参数均为标量（`window_length`、`periodic`、`alpha`、`beta`），没有 Tensor 输入。因此不存在需要梯度传播的输入张量。输出张量是全新创建的，不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 49. torch.hann_window

**反向来源类型**：不可微（工厂函数）

**Forward 路径**：`aten::hann_window`（及其重载）→ `CompositeExplicitAutograd` → 以 `alpha=0.5, beta=0.5` 进入 Hamming 公式 → `aten::arange` → `aten::mul_.Scalar` → `aten::cos_` → `aten::mul_.Scalar` → `aten::add_.Scalar` → （periodic 时）`aten::narrow`

**derivatives.yaml**：无条目。

`hann_window` 是 Hann 窗函数工厂方法，实现为 Hamming 公式的特例。所有参数均为标量，没有 Tensor 输入。输出张量是全新创建的，不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 50. torch.heaviside

**反向来源类型**：不可微

**Forward 路径**：`aten::heaviside` → structured wrapper → `heaviside_out` → `heaviside_stub` → backend kernel

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp**：
```cpp
m.impl("heaviside", torch::autograd::autogradNotImplementedFallback());
m.impl("heaviside_", torch::autograd::autogradNotImplementedFallback());
m.impl("heaviside.out", torch::autograd::autogradNotImplementedFallback());
```

Heaviside 阶跃函数 `H(x) = {0 if x<0, values if x==0, 1 if x>0}` 在 x=0 处不连续，不存在有意义的导数。autograd 使用 `autogradNotImplementedFallback` 直接 redispatch，不建立 backward node，输出无 autograd 历史。

对 `requires_grad=True` 的输入，如果用户对输出调用 `.backward()`，由于输出不带 autograd 历史，会报 "element 0 of tensors does not require grad" 错误。

**反向 ATen 依赖**：无。
