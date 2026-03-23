# torch API 151-155 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 151 | `torch.nn.functional.max_unpool1d` | Python 包装到 `max_unpool2d`，再走 helper | `MaxUnpool2DBackward0` + `ViewBackward0` | `max_unpool2d: self: max_pool_double_backward(grad, indices, 2)` | `aten::gather`、`as_strided_backward` | `max_unpool1d` 本质是 `unsqueeze -> max_unpool2d -> squeeze`；真正 backward helper `max_pool_double_backward` 展开后核心是 `gather` |
| 152 | `torch.nn.functional.max_unpool3d` | helper / backward helper 展开 | `MaxUnpool3DBackward0` | `self: max_pool_double_backward(grad, indices, 3)` | `aten::gather` | `max_pool_double_backward` 先把 `grad/indices` reshape 成 2D，再对最后一维执行 `gather` |
| 153 | `torch.nn.functional.multi_margin_loss` | helper / backward ATen op | `MultiMarginLossBackward0` | `self: multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)` | `aten::multi_margin_loss_backward` | `target`、`weight` 都不求导；native backward 是专门的 CPU/CUDA kernel |
| 154 | `torch.nn.functional.multilabel_margin_loss` | CIA 前向分解到 `multilabel_margin_loss_forward` | `MultilabelMarginLossBackward0` | `multilabel_margin_loss_forward: self: multilabel_margin_loss_backward(grad, self, target, reduction, is_target)` | `aten::multilabel_margin_loss_backward` | 公开 `multilabel_margin_loss` 只取 forward tuple 的第 0 个输出，梯度实际挂在 `multilabel_margin_loss_forward` 上 |
| 155 | `torch.nn.functional.multilabel_soft_margin_loss` | Python 组合前向，无专属 backward | 无统一 node；主要由两路 `LogSigmoidBackward0`、逐元素算子和 reduction 组合 | 无单独条目 | `aten::log_sigmoid_backward`、`aten::neg`、`aten::rsub.Scalar`、`aten::mul.Tensor`、`aten::add.Tensor`、`aten::sum.dim_IntList`、`aten::div.Scalar`、`aten::mean`、`aten::sum` | 没有 fused loss schema；梯度完全由 Python 组合里的基础 ATen 算子承担 |

## 详细分析

### 151. torch.nn.functional.max_unpool1d

**反向来源类型**：Python 包装到 `max_unpool2d`，再走 helper

**Forward 路径**：
`unsqueeze(-1)`（`input/indices`）→ `aten::max_unpool2d` → `squeeze(-1)`

**derivatives.yaml 条目**（真实落点是 `max_unpool2d`）：
```yaml
- name: max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size) -> Tensor
  self: max_pool_double_backward(grad, indices, 2)
```

**helper 展开**（`FunctionsManual.cpp`）：
```cpp
return grad.contiguous(memory_format)
    .view_symint(size)
    .gather(-1, indices_view)
    .view_symint(indices.sym_sizes());
```

**反向 ATen 依赖**：
- `aten::gather`
- `as_strided_backward` — 外层 `unsqueeze/squeeze` view backward

**备注**：
- `indices` 是 non-differentiable。
- `max_pool_double_backward` 只是 helper 名称，文档里应记录其展开后的 ATen 依赖。

---

### 152. torch.nn.functional.max_unpool3d

**反向来源类型**：helper / backward helper 展开

**derivatives.yaml 条目**：
```yaml
- name: max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding) -> Tensor
  self: max_pool_double_backward(grad, indices, 3)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::max_unpool3d` 生成 `MaxUnpool3DBackward0`
- `Functions.cpp` 中 `MaxUnpool3DBackward0::apply()` 调用 `max_pool_double_backward(...)`

**helper 展开后的反向 ATen 依赖**：
- `aten::gather`

**备注**：
- 与 `max_unpool2d` 相同，只是把最后一段空间维展平后做 `gather`。

---

### 153. torch.nn.functional.multi_margin_loss

**反向来源类型**：helper / backward ATen op

**derivatives.yaml 条目**：
```yaml
- name: multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
  self: multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `MultiMarginLossBackward0`
- `Functions.cpp` 中 `MultiMarginLossBackward0::apply()` 直接调用 `multi_margin_loss_backward(...)`

**反向 ATen 依赖**：
- `aten::multi_margin_loss_backward`

**备注**：
- native backward 在 `LossMultiMargin.cpp` / CUDA 对应实现里是专门 kernel，没有再向下复合成其他公开 ATen schema。
- `target` 与 `weight` 不可导。

---

### 154. torch.nn.functional.multilabel_margin_loss

**反向来源类型**：CIA 前向分解到 `multilabel_margin_loss_forward`

**Forward 路径**：
`aten::multilabel_margin_loss`（CIA）→ `aten::multilabel_margin_loss_forward` → 返回 `(output, is_target)`，外层只取 `output`

**derivatives.yaml 条目**：
```yaml
- name: multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
  self: multilabel_margin_loss_backward(grad, self, target, reduction, is_target)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::multilabel_margin_loss_forward` 生成 `MultilabelMarginLossBackward0`
- `Functions.cpp` 对应 `apply()` 调用 `multilabel_margin_loss_backward(...)`

**反向 ATen 依赖**：
- `aten::multilabel_margin_loss_backward`

**备注**：
- `multilabel_margin_loss` 自身没有单独 backward node，因为它只是从 `forward` tuple 中取第一个返回值。

---

### 155. torch.nn.functional.multilabel_soft_margin_loss

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**（`functional.py`）：
```python
loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
if weight is not None:
    loss = loss * weight
loss = loss.sum(dim=class_dim) / C
```

**反向 ATen 依赖**：
- `aten::log_sigmoid_backward` — 两路 `logsigmoid(input)` / `logsigmoid(-input)` 的核心 backward
- `aten::neg`
- `aten::rsub.Scalar`
- `aten::mul.Tensor`
- `aten::add.Tensor`
- `aten::sum.dim_IntList`
- `aten::div.Scalar`
- `aten::mean` / `aten::sum`

**备注**：
- 若 `weight is not None`，额外多一层 `aten::mul.Tensor`。
- 该接口没有 fused 的 `aten::multilabel_soft_margin_loss` schema，文档应按组合图来记 backward 依赖。
