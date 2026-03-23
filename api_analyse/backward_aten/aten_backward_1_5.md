# torch API 1-5 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `torch.absolute` | inline 公式（经 `aten::abs`） | `AbsBackward0` | `self: grad * self.sgn()` | `aten::mul.Tensor`、`aten::sgn` | `absolute` 是 CIA 别名，直接分解为 `aten::abs`；反向挂在 `abs` 上；复数→实数输入时 `handle_r_to_c` 额外调用 `aten::real` |
| 2 | `torch.addcdiv` | inline 公式 | `AddcdivBackward0` | `self: handle_r_to_c(…, grad)`；`tensor1: handle_r_to_c(…, grad * (value / tensor2).conj())`；`tensor2: handle_r_to_c(…, -grad * (value * tensor1 / (tensor2 * tensor2)).conj())` | `aten::mul.Tensor`、`aten::mul.Scalar`、`aten::div.Tensor`、`aten::neg`、`aten::conj` | 复数→实数输入时 `handle_r_to_c` 额外调用 `aten::real` |
| 3 | `torch.addcmul` | inline 公式 | `AddcmulBackward0` | `self: handle_r_to_c(…, grad)`；`tensor1: handle_r_to_c(…, grad * (tensor2 * value).conj())`；`tensor2: handle_r_to_c(…, grad * (tensor1 * value).conj())` | `aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj` | 复数→实数输入时 `handle_r_to_c` 额外调用 `aten::real` |
| 4 | `torch.addr` | inline + helper（`maybe_multiply`） | `AddrBackward0` | `self: maybe_multiply(grad, beta.conj())`；`vec1: maybe_multiply(grad.mv(vec2.conj()), alpha.conj())`；`vec2: maybe_multiply(grad.t().mv(vec1.conj()), alpha.conj())` | `aten::mul.Scalar`（`maybe_multiply`，当 scalar≠1 时）、`aten::mv`、`aten::conj`、`aten::t` | `maybe_multiply` 当 scalar==1 时短路返回原 tensor，不产生 `aten::mul` 调用 |
| 5 | `torch.adjoint` | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无 derivatives.yaml 条目 | 前向分解产生的各 op 各自反向：`aten::transpose`（来自 `transpose.int` 反向）、`aten::conj`（来自 `_conj` 反向，仅复数）、`as_strided_backward`（来自 `as_strided` 反向） | `adjoint` 是 `CompositeImplicitAutograd`，前向分解为 `transpose.int` → `as_strided`（非复数）或再 + `conj` → `_conj` → `alias`（复数）；autograd 对每个子 op 独立录制，无统一 backward node |

## 详细分析

### 1. torch.absolute → aten::abs

**反向来源类型**：inline 公式

**Forward 路径**：`aten::absolute`（CIA）→ `aten::abs`（CEA）→ `aten::abs.out`（backend kernel）

**derivatives.yaml 条目**：
```yaml
- name: abs(Tensor self) -> Tensor
  self: grad * self.sgn()
```

**生成代码**（`Functions.cpp` `AbsBackward0`）：
```cpp
auto grad_result = any_grad_defined ? (grad * self.sgn()) : Tensor();
```

**反向 ATen 依赖**：
- `aten::sgn` — 计算 `self` 的符号
- `aten::mul.Tensor` — `grad * sgn_result`

**辅助函数**：`handle_r_to_c` 仅在 forward-mode JVP 的 `result` 公式中出现，backward 路径上实际不涉及。

---

### 2. torch.addcdiv

**反向来源类型**：inline 公式

**Forward 路径**：`aten::addcdiv`（`structured_delegate: addcdiv.out`）→ backend kernel

**derivatives.yaml 条目**：
```yaml
- name: addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  tensor1: handle_r_to_c(tensor1.scalar_type(), grad * (value / tensor2).conj())
  tensor2: handle_r_to_c(tensor2.scalar_type(), -grad * (value * tensor1 / (tensor2 * tensor2)).conj())
```

**生成代码**（`Functions.cpp` `AddcdivBackward0`）：
```cpp
// self 梯度
handle_r_to_c(self_scalar_type, grad)
// tensor1 梯度
handle_r_to_c(tensor1_scalar_type, grad * (value / tensor2).conj())
// tensor2 梯度
handle_r_to_c(tensor2_scalar_type, -grad * (value * tensor1 / (tensor2 * tensor2)).conj())
```

**反向 ATen 依赖**：
- `aten::mul.Tensor` — grad 与中间结果相乘
- `aten::mul.Scalar` — `value * tensor1`（Scalar × Tensor）
- `aten::div.Tensor` — `value / tensor2`、`… / (tensor2 * tensor2)`
- `aten::neg` — tensor2 梯度的取负
- `aten::conj` — 复数共轭（实数张量时为 no-op view）

**条件依赖**：`handle_r_to_c` 在复数梯度 + 实数输入场景调用 `aten::real`。

---

### 3. torch.addcmul

**反向来源类型**：inline 公式

**Forward 路径**：`aten::addcmul`（`structured_delegate: addcmul.out`）→ backend kernel

**derivatives.yaml 条目**：
```yaml
- name: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  tensor1: handle_r_to_c(tensor1.scalar_type(), grad * (tensor2 * value).conj())
  tensor2: handle_r_to_c(tensor2.scalar_type(), grad * (tensor1 * value).conj())
```

**生成代码**（`Functions.cpp` `AddcmulBackward0`）：
```cpp
// self 梯度
handle_r_to_c(self_scalar_type, grad)
// tensor1 梯度
handle_r_to_c(tensor1_scalar_type, grad * (tensor2 * value).conj())
// tensor2 梯度
handle_r_to_c(tensor2_scalar_type, grad * (tensor1 * value).conj())
```

**反向 ATen 依赖**：
- `aten::mul.Tensor` — grad 与中间结果相乘
- `aten::mul.Scalar` — `tensor2 * value`、`tensor1 * value`（Tensor × Scalar）
- `aten::conj` — 复数共轭

**条件依赖**：`handle_r_to_c` 在复数梯度 + 实数输入场景调用 `aten::real`。

---

### 4. torch.addr

**反向来源类型**：inline + helper（`maybe_multiply`）

**Forward 路径**：`aten::addr` → CPU/CUDA: `addr` kernel；MPS: `addr_mps`；CEA fallback: `math_addr`

**derivatives.yaml 条目**：
```yaml
- name: addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  vec1: maybe_multiply(grad.mv(vec2.conj()), alpha.conj())
  vec2: maybe_multiply(grad.t().mv(vec1.conj()), alpha.conj())
```

**生成代码**（`Functions.cpp` `AddrBackward0`）：
```cpp
// self 梯度
maybe_multiply(grad, beta.conj())
// vec1 梯度
maybe_multiply(grad.mv(vec2.conj()), alpha.conj())
// vec2 梯度
maybe_multiply(grad.t().mv(vec1.conj()), alpha.conj())
```

**`maybe_multiply` 展开**（`FunctionsManual.cpp:130`）：
```cpp
if (scalar == 1) return tensor;     // 无 ATen 调用
else return tensor * scalar;        // aten::mul.Scalar
```

**反向 ATen 依赖**：
- `aten::mul.Scalar` — `maybe_multiply` 当 beta/alpha ≠ 1 时
- `aten::mv` — 矩阵-向量乘法
- `aten::conj` — 复数共轭
- `aten::t` — 矩阵转置（`grad.t()`）

---

### 5. torch.adjoint

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- 非复数 `ndim≥2`：`adjoint` → `transpose.int` → `as_strided`
- 复数 `ndim≥2`：`adjoint` → `transpose.int` → `as_strided` → `conj` → `_conj` → `alias`
- 0-D 复数：`adjoint` → `conj` → `_conj` → `alias`
- 0-D 非复数：`adjoint` → 直接返回 `self`

**无 derivatives.yaml 条目**。由于 `adjoint` 是 CIA，PyTorch autograd 引擎在前向执行时对每个子操作独立录制 autograd 节点。反向时各子 op 独立求导：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `transpose.int` | `self: grad.transpose(dim0, dim1)` | `aten::transpose` |
| `as_strided` | `self: as_strided_backward(grad, TensorGeometry(self), size, stride, storage_offset)` | `as_strided_backward` helper（内部使用 `aten::zeros`、`aten::as_strided` 等） |
| `_conj` | `self: grad.conj()` | `aten::conj` |
| `alias` | `self: grad` | 无（identity） |

因此 `torch.adjoint` 的反向不产生单一 backward node，而是由前向分解出的各 view op 的 backward 组合而成。
