# torch API 11-15 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | `torch.atleast_3d` | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无条目 | 前向分解产生的各 op 各自反向：dim==0 时 `aten::reshape`；dim==1 时 `aten::squeeze.dim`（×2）；dim==2 时 `aten::squeeze.dim`；dim≥3 返回 self 无额外 op | CIA；dim==0 分解为 `aten::reshape`，dim==1 分解为两次 `aten::unsqueeze`，dim==2 分解为一次 `aten::unsqueeze`，dim≥3 直接返回 self |
| 12 | `torch.bartlett_window` | 工厂函数，不可微 | 无 | 无条目；使用 `autogradNotImplementedFallback` | 无 | CEA 工厂函数，输入为 `int window_length`（非 Tensor），输出不参与梯度图；autograd 层注册为 not-implemented fallback |
| 13 | `torch.bitwise_left_shift` | 不可微 | 无 | 无条目；使用 `autogradNotImplementedFallback` | 无 | 仅支持整数类型，不存在有意义的梯度；autograd 层注册为 not-implemented fallback |
| 14 | `torch.bitwise_right_shift` | 不可微 | 无 | 无条目；使用 `autogradNotImplementedFallback` | 无 | 仅支持整数类型，不存在有意义的梯度；autograd 层注册为 not-implemented fallback |
| 15 | `torch.blackman_window` | 工厂函数，不可微 | 无 | 无条目；使用 `autogradNotImplementedFallback` | 无 | CEA 工厂函数，输入为 `int window_length`（非 Tensor），输出不参与梯度图；autograd 层注册为 not-implemented fallback |

## 详细分析

### 11. torch.atleast_3d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**（`TensorTransformations.cpp:233`）：
- `dim == 0`：`atleast_3d` → `self.reshape({1, 1, 1})` → `aten::reshape`
  - `reshape` 本身也是 CIA → contiguous 时调用 `self.view(shape)` → `aten::view`；non-contiguous 时 `_reshape_alias`
- `dim == 1`：`atleast_3d` → `self.unsqueeze(0).unsqueeze(-1)` → 两次 `aten::unsqueeze`
- `dim == 2`：`atleast_3d` → `self.unsqueeze(-1)` → 一次 `aten::unsqueeze`
- `dim >= 3`：`atleast_3d` → 直接返回 `self`（alias-like）

```cpp
Tensor atleast_3d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1, 1});
    case 1: {
      return self.unsqueeze(0).unsqueeze(-1);
    }
    case 2: {
      return self.unsqueeze(-1);
    }
    default:
      return self;
  }
}
```

**无 derivatives.yaml 条目**。

**各子 op 反向**：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `reshape`/`view` | `self: grad.reshape_symint(self.sym_sizes())` | `aten::reshape` |
| `unsqueeze` | `self: grad.squeeze(dim)` | `aten::squeeze.dim` |

dim==0 反向：grad shape `[1,1,1]` → reshape 回 `[]`（标量）。dim==1 反向：先 `grad.squeeze(-1)` 再 `grad.squeeze(0)`（两次 squeeze，逆序对应两次 unsqueeze）。dim==2 反向：`grad.squeeze(-1)` 去掉最后一维。dim≥3 无额外 op，梯度直接传递。

---

### 12. torch.bartlett_window

**反向来源类型**：工厂函数，不可微

**Forward 路径**：`aten::bartlett_window` → `CompositeExplicitAutograd` → `at::native::bartlett_window`

**native 实现**（`TensorFactories.cpp:1901`）：
```cpp
Tensor bartlett_window(int64_t window_length, bool periodic, ...) {
  // ...
  auto window = native::arange(window_length, ...)
                    .mul_(2. / static_cast<double>(window_length - 1));
  const int64_t first_half_size = ((window_length - 1) >> 1) + 1;
  window.narrow(0, first_half_size, window_length - first_half_size)
      .mul_(-1)
      .add_(2);
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}
```

该函数是工厂函数，所有输入均为 `int` / `ScalarType` 等非 Tensor 参数。输出 tensor 不带 `requires_grad`，不参与 autograd 图。

**VariableTypeEverything.cpp** 中注册为：
```cpp
m.impl("bartlett_window", torch::autograd::autogradNotImplementedFallback());
m.impl("bartlett_window.periodic", torch::autograd::autogradNotImplementedFallback());
```

**反向 ATen 依赖**：无。

---

### 13. torch.bitwise_left_shift

**反向来源类型**：不可微（`autogradNotImplementedFallback`）

**Forward 路径**：`aten::bitwise_left_shift.Tensor` → `structured_delegate: bitwise_left_shift.Tensor_out` → CPU/CUDA/MPS: `bitwise_left_shift_out` backend kernel

**Schema 变体**：
- `bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor` — 主重载
- `bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` — inplace
- `bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor` — CEA
- `bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` — CEA

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp** 中所有变体均注册为：
```cpp
m.impl("bitwise_left_shift.Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_left_shift.Tensor_Scalar", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_left_shift.Scalar_Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_left_shift_.Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_left_shift_.Tensor_Scalar", torch::autograd::autogradNotImplementedFallback());
```

位移操作仅对整数类型有意义，不存在连续的梯度定义。对 `requires_grad=True` 的浮点输入，autograd 引擎不会录制该 op（整数类型不支持 `requires_grad`）。

**反向 ATen 依赖**：无。

---

### 14. torch.bitwise_right_shift

**反向来源类型**：不可微（`autogradNotImplementedFallback`）

**Forward 路径**：`aten::bitwise_right_shift.Tensor` → `structured_delegate: bitwise_right_shift.Tensor_out` → CPU/CUDA/MPS: `bitwise_right_shift_out` backend kernel

**Schema 变体**：
- `bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor` — 主重载
- `bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` — inplace
- `bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor` — CEA
- `bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor` — CEA

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp** 中所有变体均注册为：
```cpp
m.impl("bitwise_right_shift.Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_right_shift.Tensor_Scalar", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_right_shift.Scalar_Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_right_shift_.Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("bitwise_right_shift_.Tensor_Scalar", torch::autograd::autogradNotImplementedFallback());
```

与 `bitwise_left_shift` 完全对称，仅对整数类型有意义，不可微。

**反向 ATen 依赖**：无。

---

### 15. torch.blackman_window

**反向来源类型**：工厂函数，不可微

**Forward 路径**：`aten::blackman_window` → `CompositeExplicitAutograd` → `at::native::blackman_window`

**native 实现**（`TensorFactories.cpp:1955`）：
```cpp
Tensor blackman_window(int64_t window_length, bool periodic, ...) {
  // from https://en.wikipedia.org/wiki/Window_function#Blackman_window
  auto window =
      native::arange(window_length, ...)
          .mul_(c10::pi<double> / static_cast<double>(window_length - 1));
  window =
      window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}
```

该函数是工厂函数，所有输入均为 `int` / `ScalarType` 等非 Tensor 参数。输出 tensor 不带 `requires_grad`，不参与 autograd 图。

**VariableTypeEverything.cpp** 中注册为：
```cpp
m.impl("blackman_window", torch::autograd::autogradNotImplementedFallback());
m.impl("blackman_window.periodic", torch::autograd::autogradNotImplementedFallback());
```

**反向 ATen 依赖**：无。
