# torch API 146-150 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 146 | `torch.nn.functional.huber_loss` | helper / backward ATen op | `HuberLossBackward0` | `self: huber_loss_backward(grad, self, target, reduction, delta)`；`target: huber_loss_backward(grad, target, self, reduction, delta)` | `aten::huber_loss_backward`、`aten::mul.Tensor`、`aten::mean`、`aten::sum` | 核心 loss 反向直接走 `huber_loss_backward`；Python 加权包装会在 unreduced loss 外面再叠一层 `mul + reduction` |
| 147 | `torch.nn.functional.lp_pool1d` | Python 组合前向，无专属 backward | 无统一 node；由 `PowBackward0`、`AvgPool2DBackward0`、`AbsBackward0`、`ReluBackward0`、乘法 backward 与 view backward 组合 | 无单独条目 | `aten::pow.Tensor_Scalar`、`aten::avg_pool2d_backward`、`aten::sgn`、`aten::threshold_backward`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::zeros_like`、`as_strided_backward` | `avg_pool1d` 是 CIA，反向实质落在 `avg_pool2d_backward`，并伴随 `unsqueeze/squeeze` 的 view backward |
| 148 | `torch.nn.functional.lp_pool2d` | Python 组合前向，无专属 backward | 无统一 node；由 `PowBackward0`、`AvgPool2DBackward0`、`AbsBackward0`、`ReluBackward0`、乘法 backward 组合 | 无单独条目 | `aten::pow.Tensor_Scalar`、`aten::avg_pool2d_backward`、`aten::sgn`、`aten::threshold_backward`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::zeros_like` | 与 147 相同，但没有 `avg_pool1d` 分解带来的额外 view backward |
| 149 | `torch.nn.functional.margin_ranking_loss` | CIA 前向分解，无专属 backward | 无统一 node；由逐元素算子与 reduction node 组合 | 无 derivatives.yaml 条目 | `aten::sub.Tensor`、`aten::mul.Tensor`、`aten::neg`、`aten::add.Scalar`、`aten::clamp_min`、`aten::mean`、`aten::sum` | `target` 非可导；反向直接由 composite body 里的基础 ATen 算子承担 |
| 150 | `torch.nn.functional.max_pool3d` | CIA 前向分解到带 indices 的实际 op | `MaxPool3DWithIndicesBackward0` | `self: max_pool3d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)` | `aten::max_pool3d_with_indices_backward` | `return_indices=False` 的公开 API 不单独生成 node；梯度挂在内层 `aten::max_pool3d_with_indices` 上 |

## 详细分析

### 146. torch.nn.functional.huber_loss

**反向来源类型**：helper / backward ATen op

**derivatives.yaml 条目**：
```yaml
- name: huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
  self: huber_loss_backward(grad, self, target, reduction, delta)
  target: huber_loss_backward(grad, target, self, reduction, delta)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `HuberLossBackward0`
- `Functions.cpp` 中 `HuberLossBackward0::apply()` 对 `self` 和 `target` 都调用 `huber_loss_backward(...)`

**反向 ATen 依赖**：
- `aten::huber_loss_backward`
- 若 Python 包装传入 `weight`，还会额外有：
  - `aten::mul.Tensor`
  - `aten::mean` / `aten::sum`

**备注**：
- native `huber_loss_backward` 直接进入 backend `huber_backward_stub`，没有再向下分解成更多公开 ATen schema。

---

### 147. torch.nn.functional.lp_pool1d

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**：
```python
out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1.0 / norm_type)
```

其中 `avg_pool1d` 本身是 CIA 分解：
`unsqueeze(-2) -> avg_pool2d -> squeeze(-2)`

**各子 op 反向依赖**：
- `aten::pow.Tensor_Scalar`
  - `pow_backward(...)`
  - 递归依赖 `aten::pow.Tensor_Scalar`、`aten::mul.Scalar`、`aten::mul.Tensor`、`aten::conj`
  - `exponent==0` 时退化为 `aten::zeros_like`
- `aten::avg_pool1d`
  - 分解后主反向是 `aten::avg_pool2d_backward`
  - 两侧 view 引入 `as_strided_backward`
- `aten::sign`
  - 梯度恒为零，对应 `aten::zeros_like`
- `aten::abs`
  - `aten::sgn` + `aten::mul.Tensor`
- `aten::relu`
  - `aten::threshold_backward`
- 乘法链
  - `aten::mul.Tensor`
  - `aten::mul.Scalar`

**反向 ATen 依赖**：
- `aten::pow.Tensor_Scalar`
- `aten::avg_pool2d_backward`
- `aten::sgn`
- `aten::threshold_backward`
- `aten::mul.Tensor`
- `aten::mul.Scalar`
- `aten::zeros_like`
- `as_strided_backward`

---

### 148. torch.nn.functional.lp_pool2d

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**：
```python
out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1.0 / norm_type)
```

**反向 ATen 依赖**：
- `aten::pow.Tensor_Scalar`
- `aten::avg_pool2d_backward`
- `aten::sgn`
- `aten::threshold_backward`
- `aten::mul.Tensor`
- `aten::mul.Scalar`
- `aten::zeros_like`

**备注**：
- 与 `lp_pool1d` 的区别仅在于池化主链直接是 `avg_pool2d`，不再额外依赖 `as_strided_backward`。

---

### 149. torch.nn.functional.margin_ranking_loss

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 公式**（`Loss.cpp`）：
```cpp
auto unclamped_output = (-target * (input1 - input2) + margin);
auto output = ... ? unclamped_output.clamp_min(0) : unclamped_output.clamp_min_(0);
return apply_loss_reduction(output, reduction);
```

**反向 ATen 依赖**：
- `aten::sub.Tensor`
- `aten::mul.Tensor`
- `aten::neg`
- `aten::add.Scalar`
- `aten::clamp_min` / `aten::clamp_min_`
- `aten::mean` / `aten::sum`

**备注**：
- `target` 为标签张量，不参与求导。
- 公开 schema `margin_ranking_loss` 没有单独 backward node，梯度完全由 composite body 中的基础算子承担。

---

### 150. torch.nn.functional.max_pool3d

**反向来源类型**：CIA 前向分解到带 indices 的实际 op

**Forward 路径**：
- `return_indices=False`：`aten::max_pool3d`（CIA）→ `aten::max_pool3d_with_indices` → 取 `result[0]`
- `return_indices=True`：直接 `aten::max_pool3d_with_indices`

**derivatives.yaml 条目**：
```yaml
- name: max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
  self: max_pool3d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::max_pool3d_with_indices` 生成 `MaxPool3DWithIndicesBackward0`
- `Functions.cpp` 的 `apply()` 直接调用 `max_pool3d_with_indices_backward(...)`

**反向 ATen 依赖**：
- `aten::max_pool3d_with_indices_backward`

**备注**：
- `indices` 被标记为 non-differentiable。
- 因为 `max_pool3d` 本身只是 CIA 包装层，最终应把 backward 归到 `max_pool3d_with_indices`。
