# torch API 111-115 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 111 | `torch.nn.MaxPool2d` | mixed（外层 backward ATen op + CPU/CUDA CIA 前向分解） | `MaxPool2DBackward0` | `self: max_pool2d_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode)` | `aten::max_pool2d_backward` | 普通 dense CPU/CUDA 前向实际是 `aten::max_pool2d -> aten::max_pool2d_with_indices`，但 autograd wrapper 仍挂在外层 `aten::max_pool2d` 上 |
| 112 | `torch.nn.MaxPool3d` | CIA 前向分解 → with_indices backward | `MaxPool3DWithIndicesBackward0` | `max_pool3d_with_indices`: `self: max_pool3d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)` | `aten::max_pool3d_with_indices_backward` | `aten::max_pool3d` 本身在 `Pooling.cpp` 直接取 `max_pool3d_with_indices` 的第一个返回值 |
| 113 | `torch.nn.MaxUnpool1d` | Python/ATen 组合前向，无专属 `max_unpool1d` backward node | 主要为 `MaxUnpool2DBackward0` + view backward | `max_unpool2d`: `self: max_pool_double_backward(grad, indices, 2)` | `aten::contiguous`、`aten::view`、`aten::gather`、`aten::empty_like`、`as_strided_backward` | `F.max_unpool1d` 通过 `unsqueeze` 后调用 `aten::max_unpool2d`，再 `squeeze` 回去 |
| 114 | `torch.nn.MaxUnpool3d` | helper | `MaxUnpool3DBackward0` | `self: max_pool_double_backward(grad, indices, 3)` | `aten::contiguous`、`aten::view`、`aten::gather`、`aten::empty_like` | `indices` 明确标记为 `non_differentiable` |
| 115 | `torch.nn.MultiLabelMarginLoss` | backward ATen op | `MultilabelMarginLossBackward0` | `multilabel_margin_loss_forward`: `self: multilabel_margin_loss_backward(grad, self, target, reduction, is_target)` | `aten::multilabel_margin_loss_backward` | 外层 `aten::multilabel_margin_loss` 只是入口；真正建立 grad_fn 的是 `aten::multilabel_margin_loss_forward` |

## 详细分析

### 111. torch.nn.MaxPool2d

**反向来源类型**：mixed（外层 backward ATen op + CPU/CUDA CIA 前向分解）

**锁定 schema**：
- forward：`aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor`
- CPU/CUDA 前向委托：`aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)`
- backward：`aten::max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: max_pool2d(Tensor self, ...) -> Tensor
  self: max_pool2d_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode)
```

**dispatch / native 实现验证**：
- `native_functions.yaml` 为 `aten::max_pool2d` 注册 `CompositeImplicitAutograd: max_pool2d`，同时给 `MPS` 单独注册 `mps_max_pool2d`
- `Pooling.cpp` 中，普通 dense CPU/CUDA 路径会调用 `aten::max_pool2d_with_indices(...)` 后返回 `output`
- `aten::max_pool2d_backward` 的 functional schema 只显式列出 `MPS`；非 MPS 路径依赖其 `.out` 变体的 `CompositeExplicitAutograd` 包装

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `aten::max_pool2d` 建立 `MaxPool2DBackward0`
- `Functions.cpp` 中：

```cpp
auto grad_result = any_grad_defined ? (max_pool2d_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::max_pool2d_backward`

**前向委托链**：
- 普通 dense CPU/CUDA：`aten::max_pool2d -> aten::max_pool2d_with_indices`

**备注**：
- 对当前 skill 默认关注的普通 dense CPU/CUDA，应把“外层 autograd node 调 `aten::max_pool2d_backward`”和“前向 runtime 实际走 CIA 分解到 `aten::max_pool2d_with_indices`”同时写出，不能压成单一路径
- `MaxPool2DBackward0` 是挂在外层 `aten::max_pool2d` 上的 autograd node，不是 `aten::max_pool2d_with_indices` 的 node
- double backward 侧，`derivatives.yaml` 为 `max_pool2d_backward` 标记了 `error_for_max_pool2d_double_backward()`（MPS 特例）

---

### 112. torch.nn.MaxPool3d

**反向来源类型**：CIA 前向分解 → with_indices backward

**锁定 schema**：
- 外层：`aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor`
- 内层：`aten::max_pool3d_with_indices(...) -> (Tensor, Tensor)`

**derivatives.yaml**：
- `aten::max_pool3d` **无条目**
- `aten::max_pool3d_with_indices` 有条目：

```yaml
- name: max_pool3d_with_indices(...) -> (Tensor, Tensor)
  self: max_pool3d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)
```

**native 实现验证**（`Pooling.cpp`）：

```cpp
auto output_and_indices = at::max_pool3d_with_indices(
    self, kernel_size, stride, padding, dilation, ceil_mode);
return std::get<0>(output_and_indices);
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `max_pool3d_with_indices` 建立 autograd wrapper
- `Functions.cpp` 的 node 名称是 `MaxPool3DWithIndicesBackward0`

**反向 ATen 依赖**：
- `aten::max_pool3d_with_indices_backward`

**备注**：
- 量化和 MKLDNN 分支是其他 dispatch 路径；当前前向文档锁定普通 CPU/CUDA 张量

---

### 113. torch.nn.MaxUnpool1d

**反向来源类型**：Python/ATen 组合前向，无专属 `max_unpool1d` backward node

**锁定 schema**：
- Python 组合：`F.max_unpool1d`
- 内层真实求导 schema：`aten::max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size) -> Tensor`

**view / inplace 检查**：
- 前向含 `unsqueeze` / `squeeze`，因此有 view backward

**derivatives.yaml 条目**：

```yaml
- name: max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size) -> Tensor
  self: max_pool_double_backward(grad, indices, 2)
  indices: non_differentiable
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `aten::max_unpool2d` 建立 `MaxUnpool2DBackward0`
- `Functions.cpp` 对应 helper 调用 `max_pool_double_backward(grad, indices, 2)`

**helper 展开**（`FunctionsManual.cpp`）：
- 非空 indices：
  - `aten::contiguous`
  - `aten::view`
  - `aten::gather`
  - `aten::view`
- 空输入：
  - `aten::empty_like`

**反向 ATen 依赖**：
- `aten::contiguous`
- `aten::view`
- `aten::gather`
- `aten::empty_like`
- `as_strided_backward`

**备注**：
- `as_strided_backward` 来自 `unsqueeze` / `squeeze` 这两层 view 的反向

---

### 114. torch.nn.MaxUnpool3d

**反向来源类型**：helper

**锁定 schema**：
- `aten::max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: max_unpool3d(...) -> Tensor
  self: max_pool_double_backward(grad, indices, 3)
  indices: non_differentiable
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `aten::max_unpool3d` 建立 `MaxUnpool3DBackward0`
- `Functions.cpp` 中 `MaxUnpool3DBackward0::apply()` 调用 `max_pool_double_backward(grad, indices, 3)`

**helper 展开后的反向依赖**：
- `aten::contiguous`
- `aten::view`
- `aten::gather`
- `aten::empty_like`

**备注**：
- 与 `MaxUnpool1d` 的内层 `max_unpool2d` 机制相同，只是 `dim=3`

---

### 115. torch.nn.MultiLabelMarginLoss

**反向来源类型**：backward ATen op

**锁定 schema**：
- forward 入口：`aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor`
- 实际建立梯度：`aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)`
- backward：`aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
  self: multilabel_margin_loss_backward(grad, self, target, reduction, is_target)
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `multilabel_margin_loss_forward` 建立 autograd wrapper
- `Functions.cpp` 里 node 名称是 `MultilabelMarginLossBackward0`

```cpp
auto grad_result = any_grad_defined ? (multilabel_margin_loss_backward(grad, self, target, reduction, is_target)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::multilabel_margin_loss_backward`

**备注**：
- `target` 非可导
- 外层 `aten::multilabel_margin_loss` 只是调用 `multilabel_margin_loss_forward` 再取 `output`
