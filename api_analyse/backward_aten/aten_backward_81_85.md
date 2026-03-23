# torch API 81-85 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 81 | `torch.mvlgamma` | helper 函数 | `MvlgammaBackward0` | `self: mvlgamma_backward(grad, self, p)` | `aten::arange`、`aten::unsqueeze`、`aten::add.Tensor`、`aten::digamma_`、`aten::sum`、`aten::mul.Tensor` | CEA op；helper 在 `FunctionsManual.cpp:572` |
| 82 | `torch.nanmean` | CIA 前向分解，无专属 backward | 无统一 backward node（子 op `NansumBackward0` + `DivBackward1` 等） | 无条目；内部 `nansum` 公式：`self: nansum_backward(grad.to(self.scalar_type()), self, dim, keepdim)` | `aten::unsqueeze`、`aten::expand_symint`、`aten::isnan`、`aten::logical_not`、`aten::mul.Tensor`、`aten::div.Tensor`、`aten::neg` | CIA 分解为 `nansum` + `div`，这里按递归展开口径展示到基础 ATen op |
| 83 | `torch.nanmedian` | 全归约：helper 函数；按维归约：ATen backward op | 全归约：`NanmedianBackward0`；按维归约：`NanmedianBackward1` | 全归约：`self: evenly_distribute_backward(grad, self, result)`；按维归约：`self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)` | 全归约：`aten::isnan`、`aten::logical_and_`/`aten::logical_or_`、`aten::eq.Tensor`、`aten::sum`、`aten::div.Scalar`、`aten::mul.Tensor`、`aten::masked_fill_`；按维归约：`aten::zeros_symint`、`aten::scatter_`/`aten::scatter`、`aten::unsqueeze` | 全归约使用 `evenly_distribute_backward`；按维归约使用 `value_selecting_reduction_backward_symint` |
| 84 | `torch.nextafter` | 不可微（`not_implemented`） | `NextafterBackward0` | `self: not_implemented("nextafter")`；`other: not_implemented("nextafter")` | 无（调用时抛出异常） | 虽有 backward node，但尝试反向传播时会抛出 `NotImplementedError` |
| 85 | `torch.nn.AdaptiveMaxPool3d` | backward ATen op | `AdaptiveMaxPool3DBackward0` | `self: adaptive_max_pool3d_backward(grad, self, result1)` | `aten::adaptive_max_pool3d_backward`（structured，CPU/CUDA backend kernel） | `result1` 是 indices；`output_differentiability: [True, False]` |

## 详细分析

### 81. torch.mvlgamma

**反向来源类型**：helper 函数

**Forward 路径**：`aten::mvlgamma` → `CompositeExplicitAutograd` → `at::native::mvlgamma` → `aten::arange` → `aten::unsqueeze` → `aten::add.Tensor` → `aten::lgamma_` → `aten::sum` → `aten::add_.Scalar`

**derivatives.yaml 条目**：
```yaml
- name: mvlgamma(Tensor self, int p) -> Tensor
  self: mvlgamma_backward(grad, self, p)
  result: auto_element_wise
```

**Backward Node**：`MvlgammaBackward0`

**`mvlgamma_backward` 实现**（`FunctionsManual.cpp:572`）：
```cpp
Tensor mvlgamma_backward(const Tensor& grad, const Tensor& self, int64_t p) {
  Tensor args = at::arange(
      -static_cast<double>(p) / 2. + 0.5, 0.5, 0.5,
      self.options().layout(c10::kStrided));      // aten::arange
  args = args.add(self.unsqueeze(-1));            // aten::unsqueeze + aten::add.Tensor
  return grad * args.digamma_().sum(-1);          // aten::digamma_ + aten::sum + aten::mul.Tensor
}
```

**反向 ATen 依赖**：
- `aten::arange` — 生成 `[-p/2+0.5, -p/2+1, ..., 0]` 参数序列
- `aten::unsqueeze` — 给 self 增加最后一维
- `aten::add.Tensor` — `args + self.unsqueeze(-1)`
- `aten::digamma_` — 原地计算 digamma 函数
- `aten::sum` — 沿最后一维求和
- `aten::mul.Tensor` — `grad * ...`

---

### 82. torch.nanmean

**反向来源类型**：CIA 前向分解，无专属 backward

**Forward 分解路径**：`aten::nanmean` (CIA) → 计算非 NaN 计数：`aten::detach` → `aten::isnan` → `aten::logical_not_` → `aten::sum`；数值分支：`aten::nansum` → `aten::div.Tensor`

`nanmean` 是 `CompositeImplicitAutograd`，没有 derivatives.yaml 条目。autograd 对每个内部 op 独立录制：

**`nansum` 的 derivatives.yaml 条目**：
```yaml
- name: nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: nansum_backward(grad.to(self.scalar_type()), self, dim, keepdim)
```

**`nansum_backward` 实现**（`FunctionsManual.cpp:722`）：
```cpp
Tensor nansum_backward(const Tensor& grad, const Tensor& self,
    at::OptionalIntArrayRef dims, bool keepdim) {
  return sum_backward(grad, self.sym_sizes(), dims, keepdim) *
      self.isnan().logical_not();
  // sum_backward: unsqueeze_multiple + expand_symint（或直接 expand_symint）
  // aten::isnan + aten::logical_not + aten::mul.Tensor
}
```

**`div.Tensor` 的反向**另由 `DivBackward1` 独立处理（`self: grad / other.conj()`，`other: -grad * self / (other * other).conj()`）。

**反向 ATen 依赖（nansum 部分）**：
- `aten::unsqueeze` — `sum_backward` 中的 `unsqueeze_multiple`
- `aten::expand_symint` — 广播梯度
- `aten::isnan` — 检测 NaN
- `aten::logical_not` — 取非 NaN 掩码
- `aten::mul.Tensor` — 将梯度乘以非 NaN 掩码

**反向 ATen 依赖（div 部分）**：
- `aten::div.Tensor` — self 梯度
- `aten::mul.Tensor`、`aten::neg` — other 梯度

---

### 83. torch.nanmedian

**反向来源类型**：全归约：helper 函数；按维归约：ATen backward op（`value_selecting_reduction_backward_symint`）

#### 全归约：`torch.nanmedian(input)` → `aten::nanmedian`

**derivatives.yaml 条目**：
```yaml
- name: nanmedian(Tensor self) -> Tensor
  self: evenly_distribute_backward(grad, self, result)
```

**Backward Node**：`NanmedianBackward0`

**`evenly_distribute_backward` 实现**（`FunctionsManual.cpp:1841`）：
```cpp
Tensor evenly_distribute_backward(const Tensor& grad, const Tensor& input, const Tensor& value) {
  // CUDA/subclass 路径：
  const auto input_isnan = input.isnan();           // aten::isnan
  const auto value_isnan = value.isnan();            // aten::isnan
  const auto& input_and_value_isnan = input_isnan.logical_and_(value_isnan);  // aten::logical_and_
  const auto mask = (input == value).logical_or_(input_and_value_isnan);       // aten::eq.Tensor + aten::logical_or_
  return mask * (grad / mask.sum());                 // aten::sum + aten::div.Scalar + aten::mul.Tensor

  // CPU 路径：
  auto mask = value.isnan().item<bool>() ? input.isnan() : input == value;
  return grad.new_zeros(...).masked_fill_(mask, grad / mask.sum());
  // aten::new_zeros + aten::masked_fill_ + aten::sum + aten::div.Scalar
}
```

**反向 ATen 依赖（全归约）**：
- `aten::isnan` — 检测 NaN
- `aten::eq.Tensor` — 比较 input 与 median 值
- `aten::logical_and_` — NaN 联合掩码
- `aten::logical_or_` — 合并掩码
- `aten::sum` — 计算匹配数量
- `aten::div.Scalar` — `grad / count`
- `aten::mul.Tensor` — `mask * (grad / count)`
- `aten::new_zeros`、`aten::masked_fill_`（CPU 路径）

#### 按维归约：`torch.nanmedian(input, dim)` → `aten::nanmedian.dim`

**derivatives.yaml 条目**：
```yaml
- name: nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward_symint(grad, dim, indices, self.sym_sizes(), keepdim)
```

**Backward Node**：`NanmedianBackward1`

**`value_selecting_reduction_backward_symint` 实现**（`ReduceOps.cpp:2350`）：
```cpp
Tensor value_selecting_reduction_backward_symint(...) {
  auto grad_in = at::zeros_symint(sizes, grad.options());   // aten::zeros_symint
  // keepdim=false 时先 unsqueeze
  auto grad_ = grad.unsqueeze(dim);                          // aten::unsqueeze
  auto indices_ = indices.unsqueeze(dim);                    // aten::unsqueeze
  return grad_in.scatter_(dim, indices_, grad_);             // aten::scatter_
}
```

**反向 ATen 依赖（按维归约）**：
- `aten::zeros_symint` — 创建全零梯度
- `aten::unsqueeze` — keepdim=false 时恢复维度
- `aten::scatter_` / `aten::scatter` — 按索引散布梯度

---

### 84. torch.nextafter

**反向来源类型**：不可微（`not_implemented`）

**Forward 路径**：`aten::nextafter` → structured wrapper → CPU/CUDA/MPS backend kernel

**derivatives.yaml 条目**：
```yaml
- name: nextafter(Tensor self, Tensor other) -> Tensor
  self: not_implemented("nextafter")
  other: not_implemented("nextafter")
```

**Backward Node**：`NextafterBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = not_implemented("nextafter");  // 抛出 NotImplementedError
```

虽然有 backward node 并保存了输入，但实际尝试反向传播时会抛出 `NotImplementedError`。`nextafter` 在数学上是几乎处处常数的阶梯函数，梯度无意义。

**反向 ATen 依赖**：无（运行时抛异常）。

---

### 85. torch.nn.AdaptiveMaxPool3d

**反向来源类型**：backward ATen op

**Forward 路径**：`aten::adaptive_max_pool3d` → structured wrapper → CPU/CUDA backend kernel（输出 values + indices）

**derivatives.yaml 条目**：
```yaml
- name: adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
  self: adaptive_max_pool3d_backward(grad, self, result1)
  output_differentiability: [True, False]
```

**Backward Node**：`AdaptiveMaxPool3DBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = adaptive_max_pool3d_backward(grad, self, result1);
// result1 = indices
```

**`adaptive_max_pool3d_backward`** 本身是 structured ATen op，有 CPU 和 CUDA 独立的 backend kernel：
- CPU: `adaptive_max_pool3d_backward_out_cpu`
- CUDA: `adaptive_max_pool3d_backward_out_cuda`

**反向 ATen 依赖**：
- `aten::adaptive_max_pool3d_backward` — 整体作为一个 backend kernel 执行，内部不再分解为其他 ATen op
