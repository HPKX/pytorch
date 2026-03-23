# torch API 141-145 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 141 | `torch.nn.functional.dropout3d` | CIA 前向分解，无专属 backward | 无统一 node；训练态主要为 `MulBackward0`，4D 无 batch 输入时再叠加 view backward | 无 derivatives.yaml 条目 | `aten::mul.Tensor`、`as_strided_backward` | 与 `dropout1d/2d` 相同；4D 输入的 `unsqueeze/squeeze` 额外引入 view backward |
| 142 | `torch.nn.functional.gaussian_nll_loss` | Python 组合前向，无专属 backward | 无统一 node；由 `LogBackward0`、`SubBackward0`、`PowBackward0`、`DivBackward0`、`MeanBackward0/SumBackward0`、view backward 组合 | 无单独条目 | `aten::log`、`aten::sub.Tensor`、`aten::pow.Tensor_Scalar`、`aten::div.Tensor`、`aten::add.Tensor`、`aten::mul.Scalar`、`aten::mean`、`aten::sum`、`as_strided_backward` | `var.clamp_` 在 `torch.no_grad()` 中执行，不进入梯度图 |
| 143 | `torch.nn.functional.gumbel_softmax` | Python 组合前向，无专属 backward | 无统一 node；soft 分支主要为 `SoftmaxBackward0`，hard 分支用 `detach` 做 straight-through | 无单独条目 | `aten::_softmax_backward_data`、`aten::div.Scalar`、`aten::add.Tensor`、`aten::neg`、`aten::sub.Tensor` | `hard=True` 时 one-hot 路径前向值来自 `scatter_`，反向仍完全回到 softmax 分支 |
| 144 | `torch.nn.functional.hardtanh` | helper / backward ATen op | `HardtanhBackward0` | `self: hardtanh_backward(grad, self, min_val, max_val)` | `aten::hardtanh_backward` | 前向真正计算是 `clamp.out`，但反向走专用 schema `hardtanh_backward` |
| 145 | `torch.nn.functional.hinge_embedding_loss` | CIA 前向分解，无专属 backward | 无统一 node；由 `WhereBackward0`、`ClampMinBackward0`、`AddBackward0`、`MeanBackward0/SumBackward0` 组合 | 无 derivatives.yaml 条目 | `aten::where`、`aten::clamp_min`、`aten::add.Tensor`、`aten::mean`、`aten::sum` | `target` 非可导；`zeros_like` 仅构造常量分支 |

## 详细分析

### 141. torch.nn.functional.dropout3d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 路径**：
- 5D 输入：`feature_dropout` → `mul`
- 4D 无 batch 输入：`unsqueeze` → `feature_dropout` → `mul` → `squeeze`

**反向 ATen 依赖**：
- `aten::mul.Tensor`
- `as_strided_backward` — 仅 4D 无 batch 输入时，由 `unsqueeze/squeeze` view backward 引入

**备注**：
- 与 `dropout1d/2d` 同理，mask 生成链路 (`new_empty` / `bernoulli_` / `div_`) 不对输入提供梯度。

---

### 142. torch.nn.functional.gaussian_nll_loss

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**（`functional.py`）：
```python
var = var.clone()
with torch.no_grad():
    var.clamp_(min=eps)
loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)
if full:
    loss += 0.5 * math.log(2 * math.pi)
```

**关键点**：
- `var.clamp_(min=eps)` 在 `torch.no_grad()` 中执行，因此该 in-place 钳位不出现在反向图里。
- `full=True` 时加的是常数项，不新增梯度依赖。

**反向 ATen 依赖**：
- `aten::log`
- `aten::sub.Tensor`
- `aten::pow.Tensor_Scalar`
- `aten::div.Tensor`
- `aten::add.Tensor`
- `aten::mul.Scalar`
- `aten::mean` / `aten::sum`
- `as_strided_backward` — 仅 `var` 先 `unsqueeze` 广播时由 view backward 引入

**备注**：
- `target` 通常作为监督信号不需要梯度；若对 `var` 求导，梯度同样由上述组合算子回传。

---

### 143. torch.nn.functional.gumbel_softmax

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**（`functional.py`）：
```python
gumbels = (-torch.empty_like(logits).exponential_().log() + logits) / tau
y_soft = gumbels.softmax(dim)
if hard:
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
else:
    ret = y_soft
```

**反向 ATen 依赖**：
- `aten::_softmax_backward_data` — `y_soft` 的核心反向
- `aten::div.Scalar`
- `aten::add.Tensor`
- `aten::neg`
- `aten::sub.Tensor`

**hard=True 的 straight-through 说明**：
- `y_hard` 由 `max` / `zeros_like` / `scatter_` 产生，但 `y_soft.detach()` 切断了该分支梯度。
- 最终梯度只通过 `+ y_soft` 那一支回到 softmax 链，因此没有专属 “hard backward”。

---

### 144. torch.nn.functional.hardtanh

**反向来源类型**：helper / backward ATen op

**derivatives.yaml 条目**：
```yaml
- name: hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
  self: hardtanh_backward(grad, self, min_val, max_val)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::hardtanh` / `aten::hardtanh_` 生成 `HardtanhBackward0`
- `Functions.cpp` 的 `HardtanhBackward0::apply()` 直接调用 `hardtanh_backward(...)`

**反向 ATen 依赖**：
- `aten::hardtanh_backward`

**备注**：
- `hardtanh_backward` 在 native 中走 `hardtanh_backward_stub`；虽然前向内部是 `clamp.out`，但 backward 并不复用 `clamp` 的通用节点。

---

### 145. torch.nn.functional.hinge_embedding_loss

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 公式**（`Loss.cpp`）：
```cpp
auto zeros = at::zeros_like(self);
auto margin_diff = (margin - self);
auto margin_clamp = ... ? margin_diff.clamp_min(0) : margin_diff.clamp_min_(0);
auto output_margin = at::where(target != 1, margin_clamp, zeros);
auto output_self = at::where(target != -1, self, zeros);
auto output = output_margin + output_self;
return apply_loss_reduction(output, reduction);
```

**反向 ATen 依赖**：
- `aten::where`
- `aten::clamp_min` / `aten::clamp_min_`
- `aten::add.Tensor`
- `aten::mean` / `aten::sum`

**备注**：
- `target` 是非可导标签张量。
- `zeros_like` 只是构造常量分支，不向上游产生梯度依赖。
