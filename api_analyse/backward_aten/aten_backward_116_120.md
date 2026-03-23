# torch API 116-120 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 116 | `torch.nn.MultiLabelSoftMarginLoss` | Python 组合前向，无专属 backward | 无统一 node（`LogSigmoidBackward0`、`MulBackward0`、`NegBackward0`、`MeanBackward0` 等组合） | 无单一条目；主链继承 `log_sigmoid: log_sigmoid_backward(...)` | `aten::log_sigmoid_backward`、`aten::neg`、`aten::rsub.Scalar`、`aten::mul.Tensor`、`aten::add.Tensor`、`aten::sum.dim_IntList`、`aten::div.Scalar`、`aten::mean` | 有 `weight` 时额外多一层 `aten::mul.Tensor` |
| 117 | `torch.nn.MultiMarginLoss` | backward ATen op | `MultiMarginLossBackward0` | `self: multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)` | `aten::multi_margin_loss_backward` | `target` 非可导；`weight` 只作为常量参与 backward op |
| 118 | `torch.nn.MultiheadAttention` | Python/module 组合前向，无统一 module backward node | 取决于实际 attention backend：`ScaledDotProductFlashAttentionBackward0` / `ScaledDotProductEfficientAttentionBackward0` / `ScaledDotProductCudnnAttentionBackward0` / `SoftmaxBackward0` + `NativeDropoutBackward0` + `AddmmBackward0` 等 | 无单一条目；继承选中的 SDPA backend、`linear`、`softmax`、`dropout` 各自公式 | `aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、math fallback 的 `aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::matmul` / `aten::bmm` / `aten::baddbmm`、`aten::addmm` / `aten::mm` / `aten::t` | `_native_multi_head_attention` fastpath 在 autograd key 上是 `autogradNotImplementedFallback()`，训练/需梯度时不会进入 |
| 119 | `torch.nn.PixelUnshuffle` | backward ATen op | `PixelUnshuffleBackward0` | `self: pixel_shuffle(grad, downscale_factor)` | `aten::pixel_shuffle` | `pixel_unshuffle` 自身在 CPU/MPS 有 kernel；反向直接复用逆变换 `pixel_shuffle` |
| 120 | `torch.nn.PoissonNLLLoss` | CIA 前向分解，无专属 backward | 无统一 node（各基础 op 独立 backward） | 无条目；forward composite 按 `log_input/full` 分支展开 | `aten::exp`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::add.Scalar`、`aten::log`、`aten::le`、`aten::masked_fill`、`aten::add_`、`aten::mean` / `aten::sum` | `aten::poisson_nll_loss` 注册在 `CompositeImplicitAutograd`；不同参数分支反向依赖不同 |

## 详细分析

### 116. torch.nn.MultiLabelSoftMarginLoss

**反向来源类型**：Python 组合前向，无专属 backward

**锁定前向表达式**（`functional.py`）：

```python
loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
if weight is not None:
    loss = loss * weight
loss = loss.sum(dim=class_dim) / C
return loss / reduction
```

**对应子 op**：
- `aten::log_sigmoid(input)`
- `aten::neg(input)` + `aten::log_sigmoid(-input)`
- `aten::rsub.Scalar(1, target)`
- `aten::mul.Tensor`
- `aten::add.Tensor`
- `aten::neg`
- 可选 `aten::mul.Tensor`（`weight`）
- `aten::sum.dim_IntList`
- `aten::div.Scalar`
- `aten::mean` / `aten::sum`

**关键 backward 落点**：
- `log_sigmoid` 继承 `aten::log_sigmoid_backward`
- 其余都是基础点算子 / reduction 的标准 backward

**反向 ATen 依赖**：
- `aten::log_sigmoid_backward`
- `aten::neg`
- `aten::rsub.Scalar`
- `aten::mul.Tensor`
- `aten::add.Tensor`
- `aten::sum.dim_IntList`
- `aten::div.Scalar`
- `aten::mean`

**备注**：
- 这里没有 `aten::multilabel_soft_margin_loss` schema，也就没有统一 backward node

---

### 117. torch.nn.MultiMarginLoss

**反向来源类型**：backward ATen op

**锁定 schema**：
- forward：`aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor`
- backward：`aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
  self: multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `aten::multi_margin_loss` 建立 `MultiMarginLossBackward0`
- `Functions.cpp` 中：

```cpp
auto grad_result = any_grad_defined ? (multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::multi_margin_loss_backward`

**备注**：
- `target` 非可导
- 当前区间要求只跟 ATen schema，因此不展开 CPU/CUDA kernel 内部实现细节

---

### 118. torch.nn.MultiheadAttention

**反向来源类型**：Python/module 组合前向，无统一 module backward node

**锁定前向场景**（对应 forward 文档）：
- 常规路径 `need_weights=False`：`linear -> scaled_dot_product_attention -> linear`
- 常规路径 `need_weights=True`：`linear -> bmm/baddbmm -> softmax -> [dropout] -> bmm -> linear`
- 推理 fastpath：`_native_multi_head_attention`

**关键 dispatch 结论**：
- `_native_multi_head_attention` 在 `VariableTypeEverything.cpp` 里注册的是 `autogradNotImplementedFallback()`
- 因此训练/需要梯度时不会命中这个 fastpath，真实 backward 只来自常规路径里的子 op

**need_weights=False 路径**：
- 若 selector 命中 fused backend，backward 直接落到：
  - `aten::_scaled_dot_product_flash_attention_backward`
  - `aten::_scaled_dot_product_efficient_attention_backward`
  - `aten::_scaled_dot_product_cudnn_attention_backward`
- 若退回 math fallback，则主链是：
  - `aten::matmul`
  - `aten::_softmax_backward_data`
  - `aten::native_dropout_backward`（仅 `dropout_p > 0`）
  - `aten::matmul`
  - `aten::addmm` / `aten::mm` / `aten::t`（来自 `linear`）

**need_weights=True 路径**：
- 主链是：
  - `aten::bmm` / `aten::baddbmm`
  - `aten::_softmax_backward_data`
  - `aten::native_dropout_backward`（若启用 dropout）
  - `aten::bmm`
  - `aten::addmm` / `aten::mm` / `aten::t`

**反向 ATen 依赖**：
- `aten::_scaled_dot_product_flash_attention_backward`
- `aten::_scaled_dot_product_efficient_attention_backward`
- `aten::_scaled_dot_product_cudnn_attention_backward`
- `aten::_softmax_backward_data`
- `aten::native_dropout_backward`
- `aten::matmul`
- `aten::bmm`
- `aten::baddbmm`
- `aten::addmm`

**备注**：
- 这里没有统一的 `MultiheadAttentionBackward0`
- 输出投影 `linear` 的 backward 最终会拆成 `addmm/mm/t/sum`

---

### 119. torch.nn.PixelUnshuffle

**反向来源类型**：backward ATen op

**锁定 schema**：
- forward：`aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor`
- backward 复用逆算子：`aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
  self: pixel_shuffle(grad, downscale_factor)
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `pixel_unshuffle` 建立 `PixelUnshuffleBackward0`
- `Functions.cpp` 里 `PixelUnshuffleBackward0::apply()` 直接调用 `pixel_shuffle`

**反向 ATen 依赖**：
- `aten::pixel_shuffle`

**备注**：
- 这是一个很干净的“逆变换即反向”案例

---

### 120. torch.nn.PoissonNLLLoss

**反向来源类型**：CIA 前向分解，无专属 backward

**锁定 schema**：
- `aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor`

**derivatives.yaml**：
- `aten::poisson_nll_loss` **无条目**

**native/composite 实现**（`Loss.cpp`）：

```cpp
if (log_input) {
    loss = at::exp(input) - target * input;
} else {
    loss = input - target * at::log(input + eps);
}
if (full) {
    auto stirling_term = target * at::log(target) - target + 0.5 * at::log(...);
    loss += stirling_term.masked_fill(target <= 1, 0);
}
return apply_loss_reduction(loss, reduction);
```

因此 backward 由 composite 体内基础 ATen op 分别承担：

**`log_input=True, full=False`**：
- `aten::exp`
- `aten::mul.Tensor`
- `aten::sub.Tensor`
- `aten::mean` / `aten::sum`

**`log_input=False, full=True`**：
- `aten::add.Scalar`
- `aten::log`
- `aten::mul.Tensor`
- `aten::sub.Tensor`
- `aten::le`
- `aten::masked_fill`
- `aten::add_`
- `aten::mean` / `aten::sum`

**备注**：
- `target` 通常不要求梯度
- `masked_fill` 这条支路只在 `full=True` 的 Stirling 修正项中出现

