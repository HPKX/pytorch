# torch API 26-30 反向 ATen 依赖分析


| 序号  | API                | 反向来源                  | Backward Node          | derivatives.yaml 公式                                                          | 反向依赖的 ATen 接口                                                                                                                                                                                     | 备注                                                                                                   |
| --- | ------------------ | --------------------- | ---------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 26  | `torch.cov`        | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无条目                                                                          | 前向分解产生的各 op 各自反向：`aten::view` → `aten::reshape`；`aten::mul` → `aten::mul`；`aten::sub` → `aten::neg`；`aten::mm` → `aten::mm`；`aten::t` → `aten::t`；`aten::true_divide` → `aten::div`/`aten::mul` 等 | CIA；分解到大量基础算术/线性代数 op，各自有独立 backward                                                                 |
| 27  | `torch.deg2rad`    | helper 函数             | `Deg2RadBackward0`     | `self: deg2rad_backward(grad)`                                               | `aten::mul.Scalar`                                                                                                                                                                                | `deg2rad_backward` 是手写 helper（`FunctionsManual.cpp:663`），仅做 `grad * (π/180)`                         |
| 28  | `torch.diag_embed` | inline 公式             | `DiagEmbedBackward0`   | `self: grad.diagonal(offset, dim1, dim2)`                                    | `aten::diagonal`                                                                                                                                                                                  | 反向为从 grad 矩阵提取对角线；`diagonal` 是 view op（→ `aten::as_strided`）                                         |
| 29  | `torch.diagflat`   | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无条目                                                                          | 前向分解产生的各 op 各自反向：`aten::view` → `aten::reshape`；`aten::diag_embed` → `aten::diagonal`；`aten::contiguous` 无额外反向                                                                                    | CIA；分解为 `contiguous` → `view(-1)` → `diag`（CIA → `diag_embed`）                                       |
| 30  | `torch.diagonal`   | backward ATen op      | `DiagonalBackward0`    | `self: diagonal_backward_symint(grad, self.sym_sizes(), offset, dim1, dim2)` | `aten::diagonal_backward`（→ `aten::zeros`、`aten::diagonal`、`aten::copy_`）                                                                                                                         | `diagonal` 是 view op（CEA → `aten::as_strided`）；反向通过 `diagonal_backward` ATen op 创建零矩阵后在对角线位置 copy 梯度 |


## 详细分析

### 26. torch.cov

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**（`Correlation.cpp` `native::cov`）：

```
cov(self, correction, fweights, aweights)
  → 1D 输入: self.view({1, -1})                    // aten::view
  → 加权路径:
    → w = fweights * aweights（若均存在）              // aten::mul.Tensor
    → w_sum = w.sum()                                // aten::sum
  → 中心化: self - self.mul(w).sum(-1).unsqueeze(-1) / w_sum  // aten::mul, aten::sum, aten::unsqueeze, aten::sub, aten::true_divide
  → 协方差矩阵: (in - mean).t().conj().mm(in - mean * w) / norm_factor
                                                     // aten::t, aten::conj, aten::mm, aten::true_divide
  → 复数单变量特例:
    → at::real(result) + at::complex(real, zeros_like)  // aten::real, aten::zeros_like, aten::complex
  → squeeze 恢复维度                                    // aten::squeeze
```

**无 derivatives.yaml 条目**。

由于 `cov` 是 CIA，autograd 引擎对每个子操作独立录制。反向时各 op 各自求导，涉及的反向 ATen 依赖取决于具体路径中使用的子 op 组合。主要反向依赖包括：


| 子 op          | 反向依赖                                              |
| ------------- | ------------------------------------------------- |
| `view`        | `aten::reshape`                                   |
| `mul.Tensor`  | `aten::mul.Tensor`、`aten::conj`（复数）               |
| `sum`         | `aten::expand`（sum_backward）                      |
| `unsqueeze`   | `aten::squeeze.dim`                               |
| `sub`         | `aten::neg`                                       |
| `t`           | `aten::t`                                         |
| `mm`          | `aten::mm`（×2，左右矩阵分别）                             |
| `true_divide` | `aten::div.Tensor`/`aten::mul.Scalar`/`aten::neg` |
| `squeeze`     | `aten::unsqueeze`                                 |


---

### 27. torch.deg2rad

**反向来源类型**：helper 函数

**Forward 路径**：`aten::deg2rad` → `CompositeExplicitAutograd` → `at::native::deg2rad` → `aten::empty_like` → `aten::deg2rad.out` → `aten::mul.out(self, π/180)`

**derivatives.yaml 条目**：

```yaml
- name: deg2rad(Tensor self) -> Tensor
  self: deg2rad_backward(grad)
  result: auto_element_wise
```

**Backward Node**：`Deg2RadBackward0`

`**deg2rad_backward` 实现**（`FunctionsManual.cpp:663`）：

```cpp
Tensor deg2rad_backward(const Tensor& grad) {
  constexpr double M_PI_180 = 0.017453292519943295769236907684886127134428718885417;
  return at::mul(grad, Scalar(M_PI_180));    // aten::mul.Scalar
}
```

**数学含义**：`deg2rad(x) = x * π/180`，梯度为常数 `π/180`。

**反向 ATen 依赖**：

- `aten::mul.Scalar` — `grad * (π/180)`

---

### 28. torch.diag_embed

**反向来源类型**：inline 公式

**Forward 路径**：`aten::diag_embed` → `CompositeExplicitAutogradNonFunctional` → `at::native::diag_embed` → `aten::zeros`（分配结果）→ `aten::diagonal`（→ `aten::as_strided` view）→ `aten::copy`_（写入对角线）

**derivatives.yaml 条目**：

```yaml
- name: diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  self: grad.diagonal(offset, dim1, dim2)
  result: auto_linear
```

**Backward Node**：`DiagEmbedBackward0`

**生成代码**（`Functions.cpp` `DiagEmbedBackward0`）：

```cpp
auto grad_result = any_grad_defined ? (grad.diagonal(offset, dim1, dim2)) : Tensor();
```

**反向 ATen 依赖**：

- `aten::diagonal` — 从 grad 矩阵提取对角线元素（view op → 内部使用 `aten::as_strided`）

反向是前向的逆操作：前向将 1D 嵌入对角线，反向从 grad 矩阵中提取对角线。

---

### 29. torch.diagflat

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**（`TensorShape.cpp` `native::diagflat`）：

```
diagflat(self, offset)
  → self.contiguous()              // aten::contiguous（非 contiguous 时 → aten::clone）
  → .view(-1)                      // aten::view
  → diag(flattened, offset)        // aten::diag（CIA）
    → 1D 输入走 diag_embed 分支:
      → diag_embed(input, offset)  // aten::diag_embed（CEA-NF）
        → aten::zeros → aten::diagonal → aten::copy_
```

**无 derivatives.yaml 条目**（`diagflat` 和 `diag` 均无）。

`diagflat` 是 CIA → `diag` 也是 CIA → 最终 `diag_embed` 有显式 derivative entry。autograd 引擎录制链为：


| 子 op         | derivatives.yaml 公式                           | 反向依赖             |
| ------------ | --------------------------------------------- | ---------------- |
| `view`       | `self: grad.reshape_symint(self.sym_sizes())` | `aten::reshape`  |
| `diag_embed` | `self: grad.diagonal(offset, dim1, dim2)`     | `aten::diagonal` |


反向时：先通过 `diag_embed` 的 backward 从 grad 矩阵提取对角线（`aten::diagonal`），然后通过 `view` 的 backward reshape 回原始形状（`aten::reshape`）。`contiguous` 在反向时为 identity。

---

### 30. torch.diagonal

**反向来源类型**：backward ATen op

**Forward 路径**：`aten::diagonal` → `CompositeExplicitAutograd` → `at::native::diagonal` → `aten::as_strided`（view）

`diagonal` 是 view op，前向返回的是 `as_strided` 视图。

**derivatives.yaml 条目**：

```yaml
- name: diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
  self: diagonal_backward_symint(grad, self.sym_sizes(), offset, dim1, dim2)
  result: auto_linear
```

**Backward Node**：`DiagonalBackward0`

**生成代码**（`Functions.cpp` `DiagonalBackward0`）：

```cpp
auto grad_result = any_grad_defined ?
    (diagonal_backward_symint(grad, self_sym_sizes, offset, dim1, dim2)) : Tensor();
```

`**diagonal_backward_symint` 实现**（`TensorShape.cpp:4644`）：

```cpp
Tensor diagonal_backward_symint(
    const Tensor& grad, SymIntArrayRef input_sizes,
    int64_t offset, int64_t dim1, int64_t dim2) {
  auto grad_input = at::zeros_symint(input_sizes, grad.options());  // aten::zeros
  auto diag = grad_input.diagonal(offset, dim1, dim2);               // aten::diagonal（view）
  diag.copy_(grad);                                                   // aten::copy_
  return grad_input;
}
```

**反向 ATen 依赖**：

- `aten::diagonal_backward`（注册为独立 ATen op）
  - `aten::zeros` — 创建全零矩阵（原始 shape）
  - `aten::diagonal` — 获取对角线视图
  - `aten::copy_` — 将 grad 写入对角线位置

**注意**：`diagonal_backward` 本身也有 derivative entry（用于高阶梯度）：

```yaml
- name: diagonal_backward(...) -> Tensor
  grad_output: grad.diagonal(offset, dim1, dim2)
```

高阶反向时产生 `aten::diagonal` 依赖。