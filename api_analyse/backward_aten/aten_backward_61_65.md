# torch API 61-65 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 61 | `torch.isnan` | 不可微（`non_differentiable`） | 无 | `self: non_differentiable` | 无 | 返回 bool 张量；derivatives.yaml 显式标记 `non_differentiable`，autograd wrapper 不建立梯度图 |
| 62 | `torch.isposinf` | 不可微 | 无 | 无条目 | 无 | 返回 bool 张量；derivatives.yaml 无条目；autograd wrapper 直接 redispatch，不建立 backward node |
| 63 | `torch.isreal` | 不可微 | 无 | 无条目 | 无 | CIA，返回 bool 张量；无 derivatives.yaml 条目；输出不参与梯度计算 |
| 64 | `torch.kaiser_window` | 不可微（工厂函数） | 无 | 无条目 | 无 | CEA 工厂函数，无 Tensor 输入；`autogradNotImplementedFallback` |
| 65 | `torch.kron` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::reshape`、`aten::mul.Tensor`、`aten::conj` | `kron` 是 CIA，前向分解为 `_unsafe_view` + `mul.Tensor` + `_unsafe_view`；递归展开后 `_unsafe_view` 回落到 `reshape` |

## 详细分析

### 61. torch.isnan

**反向来源类型**：不可微（`non_differentiable`）

**Forward 路径**：`aten::isnan` → CPU/CUDA/MPS/MTIA backend kernel（实现为 `self != self`，即 `aten::ne.Tensor`）

**derivatives.yaml 条目**：
```yaml
- name: isnan(Tensor self) -> Tensor
  self: non_differentiable
```

**VariableTypeEverything.cpp** 中的 autograd wrapper：
```cpp
at::Tensor isnan(c10::DispatchKeySet ks, const at::Tensor & self) {
  // redispatch, 无 set_history、无 backward node
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::isnan(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  return result;
}
```

`isnan` 返回 bool 张量，标记为 `non_differentiable`，输出不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 62. torch.isposinf

**反向来源类型**：不可微

**Forward 路径**：`aten::isposinf`（`structured_delegate: isposinf.out`）→ CPU/CUDA/MPS backend kernel

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp** 中的 autograd wrapper 直接 redispatch，不建立 backward node：
```cpp
at::Tensor isposinf(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::isposinf(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  return result;
}
```

`isposinf` 返回 bool 张量，不存在有意义的梯度。输出 tensor 不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 63. torch.isreal

**反向来源类型**：不可微

**Forward 路径**：`aten::isreal`（CIA）→ 整型/浮点返回 `aten::ones_like(..., bool)`；复数路径为 `aten::imag(self)` → `aten::eq.Scalar(..., 0)`

**native_functions.yaml**：
```yaml
- func: isreal(Tensor self) -> Tensor
  variants: function, method
```
无 dispatch key，为 CompositeImplicitAutograd。

**derivatives.yaml**：无条目。

`isreal` 返回 bool 张量，不存在有意义的梯度。由于是 CIA 且所有子 op（`ones_like`、`imag`、`eq.Scalar`）的输出均为 bool tensor，整个链路不参与梯度计算。

**反向 ATen 依赖**：无。

---

### 64. torch.kaiser_window

**反向来源类型**：不可微（工厂函数）

**Forward 路径**：`aten::kaiser_window` / `aten::kaiser_window.periodic` / `aten::kaiser_window.beta`（CEA）→ `aten::empty` / `aten::ones` / `aten::arange` → `kaiser_window_stub` → backend kernel → 可能 `aten::narrow`

**native_functions.yaml**：
```yaml
- func: kaiser_window(int window_length, ...) -> Tensor
  dispatch:
    CompositeExplicitAutograd: kaiser_window
```
三个重载均为 CEA。

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp**：
```cpp
m.impl("kaiser_window", torch::autograd::autogradNotImplementedFallback());
m.impl("kaiser_window.periodic", torch::autograd::autogradNotImplementedFallback());
m.impl("kaiser_window.beta", torch::autograd::autogradNotImplementedFallback());
```

`kaiser_window` 是工厂函数，无 Tensor 输入，输出为 fresh tensor 不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 65. torch.kron

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
`aten::kron`（CIA）→ `at::native::KronImpl(self, other).kron()` → `aten::_unsafe_view(self, a_reshape)` + `aten::_unsafe_view(other, b_reshape)` → `aten::mul.Tensor` → `aten::_unsafe_view(result, result_reshape)`

**native_functions.yaml**：
```yaml
- func: kron(Tensor self, Tensor other) -> Tensor
  variants: function, method
```
无 dispatch key，为 CompositeImplicitAutograd。

**derivatives.yaml**：无 `kron` 条目。

由于 `kron` 是 CIA，PyTorch autograd 引擎在前向执行时对每个子操作独立录制 autograd 节点。反向时各子 op 独立求导：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `_unsafe_view` | `self: grad.reshape(self.sym_sizes())` | `aten::reshape`（view 操作） |
| `mul.Tensor` | `self: mul_tensor_backward(grad, other, 0)`；`other: mul_tensor_backward(grad, self, 1)` | `aten::mul.Tensor`、`aten::conj`（复数时） |

因此 `torch.kron` 的反向不产生单一 backward node，而是由前向分解出的各 op 的 backward 组合而成。最终反向 ATen 依赖为：
- `aten::reshape` — `_unsafe_view` 的反向
- `aten::mul.Tensor` — `mul.Tensor` 的反向（grad × other、grad × self）
- `aten::conj` — 复数张量时的共轭
