# torch API 181-185 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 181 | `torch.rad2deg` | helper 公式 | `Rad2DegBackward0` | `self: rad2deg_backward(grad)` | `aten::mul.Scalar` | helper 只把梯度乘上常数 `180 / pi` |
| 182 | `torch.range` | 不可微（工厂函数） | 无 | 无条目；`autogradNotImplementedFallback` | 无 | 输入是标量参数，不是 Tensor；结果是 fresh tensor，不带可微历史 |
| 183 | `torch.renorm` | helper 公式 | `RenormBackward0` | `self: renorm_backward(grad, self, p, dim, maxnorm)` | 直接依赖 helper；展开后主要依赖 `aten::linalg_vector_norm`、`aten::conj`、`aten::real`、`aten::sum`、`aten::reciprocal`、`aten::mul`、`aten::sub`、`aten::where` | helper 内还调用 `norm_backward(...)`，因此会继续展开到 norm 系列依赖 |
| 184 | `torch.rot90` | inline 公式 | `Rot90Backward0` | `self: grad.rot90(-k, dims)` | 直接依赖 `aten::rot90`；继续展开该调用会落到 `aten::flip`、`aten::transpose_`、`aten::clone` | backward 直接再次调用前向 `rot90`，只是把 `k` 取负 |
| 185 | `torch.row_stack` | CIA 前向分解，无专属 backward | 无统一 node；主要为 `CatBackward0` + view backward | 无条目 | `aten::narrow`、`aten::squeeze.dim`、`aten::reshape` | `row_stack` 是 `vstack` 纯别名，反向依赖与 `vstack`(#211) 完全一致 |

## 详细分析

### 181. torch.rad2deg

**反向来源类型**：helper 公式

**锁定 schema**：
`aten::rad2deg(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: rad2deg(Tensor self) -> Tensor
  self: rad2deg_backward(grad)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `Rad2DegBackward0`
- `Functions.cpp` 里调用 `rad2deg_backward(...)`

**helper 展开**（`FunctionsManual.cpp`）：
```cpp
Tensor rad2deg_backward(const Tensor& grad) {
  return at::mul(grad, Scalar(180 / pi));
}
```

**反向 ATen 依赖**：
- `aten::mul.Scalar`

---

### 182. torch.range

**反向来源类型**：不可微（工厂函数）

`torch.range` 对应的 schema 是：
- `aten::range.step(...) -> Tensor`
- `aten::range(...) -> Tensor`
- `aten::range.out(...) -> Tensor(a!)`

这些接口在 `VariableTypeEverything.cpp` 中都注册为 `autogradNotImplementedFallback()`，且参数是标量而非 Tensor 输入，因此不生成有意义的梯度定义。

**反向 ATen 依赖**：无。

**备注**：
- 即使 Python wrapper 允许 `requires_grad=True`，那也只是结果 tensor 的 flag，不代表该工厂函数对标量参数可导。

---

### 183. torch.renorm

**反向来源类型**：helper 公式

**锁定 schema**：
`aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
  self: renorm_backward(grad, self, p, dim, maxnorm)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `RenormBackward0`
- `Functions.cpp` 中为：
```cpp
auto grad_result = any_grad_defined ? (renorm_backward(grad, self, p, dim, maxnorm)) : Tensor();
```

**helper 展开**（`FunctionsManual.cpp`）可见主要步骤：
- `aten::linalg_vector_norm`
- `aten::conj`
- 复数时条件调用 `aten::real`
- `aten::sum`
- `norm_backward(...)`
- `aten::reciprocal`
- `aten::mul`
- `aten::sub`
- `aten::where`

**反向 ATen 依赖**：
- 直接依赖：helper `renorm_backward`
- helper 展开后的主要 ATen op：
  - `aten::linalg_vector_norm`
  - `aten::conj`
  - `aten::real`
  - `aten::sum`
  - `aten::reciprocal`
  - `aten::mul`
  - `aten::sub`
  - `aten::where`

**备注**：
- `norm_backward(...)` 自身还会进一步展开到 norm 家族的内部依赖；这里记录到 helper 的直接展开层。

---

### 184. torch.rot90

**反向来源类型**：inline 公式

**锁定 schema**：
`aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
  self: grad.rot90(-k, dims)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `Rot90Backward0`
- `Functions.cpp` 中为：
```cpp
auto grad_result = any_grad_defined ? (grad.rot90(-k, dims)) : Tensor();
```

**反向 ATen 依赖**：
- 直接依赖：`aten::rot90`
- 将这个 backward 调用继续按前向 CEA 实现展开，会落到：
  - `aten::flip`
  - `aten::transpose_`（`k % 4 == 1/3`）
  - `aten::clone`（`k % 4 == 0`）

**备注**：
- backward 本质上就是“反方向再旋转一次”。

---

### 185. torch.row_stack

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**锁定 schema**：
- `aten::row_stack(Tensor[] tensors) -> Tensor`
- `aten::vstack(Tensor[] tensors) -> Tensor`

`native_functions.yaml` 直接注明：
```yaml
# row_stack is the alias of vstack
```

`TensorShape.cpp` 中实现：
```cpp
Tensor row_stack(TensorList tensors) {
  return at::vstack(tensors);
}
```

`vstack` 再分解为：
```cpp
auto rep = at::atleast_2d(tensors);
return at::cat(rep, 0);
```

而 `atleast_2d` 对每个输入：
- 0D：`reshape({1, 1})`
- 1D：`unsqueeze(0)`
- 2D 及以上：直接返回 self

**因此 backward 不在 `row_stack` 本身生成 node，而是落到内部子 op**：
- `aten::cat` → `CatBackward0`
- `aten::unsqueeze` / `aten::reshape` → view backward

**反向 ATen 依赖**：
- `aten::cat`
- `aten::unsqueeze`
- `aten::reshape`

**备注**：
- 对 rank≥2 的输入，`atleast_2d` 直接返回 self，不产生额外 backward 依赖。
