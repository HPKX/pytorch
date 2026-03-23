# torch API 56-60 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 56 | `torch.igammac` | inline 公式 | `IgammacBackward0` | `self: not_implemented("igammac: input")`；`other: -grad * exp((self - 1) * log(other) - other - lgamma(self))` | `aten::neg`、`aten::mul.Tensor`、`aten::exp`、`aten::sub.Scalar`、`aten::log`、`aten::sub.Tensor`、`aten::lgamma` | `self` 的梯度未实现，运行时抛异常；仅 `other` 可微 |
| 57 | `torch.inner` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::mul.Tensor`、`aten::mm`、`aten::dot`、`aten::permute`、`aten::reshape` | 递归展开后同时覆盖标量 `mul` 路径和非标量 `tensordot` 路径 |
| 58 | `torch.is_complex` | 不可微 | 无 | 无条目 | 无 | 返回 Python `bool`，无 Tensor 输出，不参与 autograd |
| 59 | `torch.is_floating_point` | 不可微 | 无 | 无条目 | 无 | 返回 Python `bool`，无 Tensor 输出，不参与 autograd |
| 60 | `torch.is_nonzero` | 不可微 | 无 | 无条目 | 无 | 返回 Python `bool`，无 Tensor 输出，不参与 autograd |

## 详细分析

### 56. torch.igammac

**反向来源类型**：inline 公式

**Forward 路径**：`aten::igammac`（`structured_delegate: igammac.out`）→ CPU/CUDA/MPS backend kernel

**derivatives.yaml 条目**：
```yaml
- name: igammac(Tensor self, Tensor other) -> Tensor
  self: 'not_implemented("igammac: input")'
  other: -grad * exp((self - 1) * log(other) - other - lgamma(self))
```

**Backward Node**：`IgammacBackward0`

**生成代码**（`Functions.cpp` `IgammacBackward0`）：
```cpp
// other 梯度
auto grad_result = any_grad_defined ? (-grad * exp((self - 1) * log(other) - other - lgamma(self))) : Tensor();
// self 梯度
auto grad_result = not_implemented("igammac: input");
```

**反向 ATen 依赖**（仅 `other` 梯度路径）：
- `aten::sub.Scalar` — `self - 1`
- `aten::log` — `log(other)`
- `aten::mul.Tensor` — `(self - 1) * log(other)`
- `aten::sub.Tensor` — `... - other`、`... - lgamma(self)`
- `aten::lgamma` — `lgamma(self)`
- `aten::exp` — `exp(...)`
- `aten::neg` — 取负 `-grad * ...`（或 `aten::mul.Tensor` 与负号合并）

**注意**：`self` 的梯度标记为 `not_implemented`，若尝试对 `self` 求导会抛出运行时异常。

---

### 57. torch.inner

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- 标量输入：`aten::inner`（CIA）→ `aten::mul.Tensor`
- 非标量输入：`aten::inner`（CIA）→ `aten::tensordot` → `aten::permute` → `aten::reshape` → `aten::mm`（或 `aten::dot` 全缩并时）

**native_functions.yaml**：
```yaml
- func: inner(Tensor self, Tensor other) -> Tensor
  variants: function, method
```
无 dispatch key，为 CompositeImplicitAutograd。

**derivatives.yaml**：无 `inner` 条目。

由于 `inner` 是 CIA，PyTorch autograd 引擎在前向执行时对每个子操作独立录制 autograd 节点。反向时各子 op 独立求导：

| 路径 | 子 op | 反向依赖 |
| --- | --- | --- |
| 标量路径 | `mul.Tensor` | `aten::mul.Tensor`（grad × other / grad × self） |
| 非标量路径 | `tensordot` → `mm` | `aten::mm`、`aten::permute`、`aten::reshape` 等各自的反向 |
| 非标量路径 | `tensordot` → `dot`（全缩并） | `aten::dot` 的反向 |

---

### 58. torch.is_complex

**反向来源类型**：不可微

**Forward 路径**：`aten::is_complex`（CIA，`manual_cpp_binding: True`）→ 直接查询 dtype 元数据

**derivatives.yaml**：无条目。

`is_complex` 返回 Python `bool`（非 Tensor），不存在有意义的梯度。输出不参与 autograd 计算图。

**反向 ATen 依赖**：无。

---

### 59. torch.is_floating_point

**反向来源类型**：不可微

**Forward 路径**：`aten::is_floating_point`（CIA，`manual_cpp_binding: True`）→ 直接查询 dtype 元数据

**derivatives.yaml**：无条目。

`is_floating_point` 返回 Python `bool`（非 Tensor），不存在有意义的梯度。输出不参与 autograd 计算图。

**反向 ATen 依赖**：无。

---

### 60. torch.is_nonzero

**反向来源类型**：不可微

**Forward 路径**：`aten::is_nonzero`（CIA）→ `aten::item` → `aten::_local_scalar_dense` → bool 判定

**derivatives.yaml**：无条目。

`is_nonzero` 返回 Python `bool`（非 Tensor），不存在有意义的梯度。输出不参与 autograd 计算图。

**反向 ATen 依赖**：无。
