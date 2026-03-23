# torch API 156-160 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 156 | `torch.nn.functional.pixel_unshuffle` | helper / backward ATen op | `PixelUnshuffleBackward0` | `self: pixel_shuffle(grad, downscale_factor)` | `aten::pixel_shuffle` | `pixel_unshuffle` 的 backward 直接是对应倍率的 `pixel_shuffle` |
| 157 | `torch.nn.functional.rms_norm` | CIA 前向分解到 `_fused_rms_norm` 或 composite 基础算子 | 常见 fused 路径为 `FusedRmsNormBackward0`；fallback 路径无统一 node | `_fused_rms_norm` 的 backward 在生成代码中调用 `_fused_rms_norm_backward(...)`，高阶梯度时改走 `infinitely_differentiable_native_rms_norm_backward(...)` | `aten::_fused_rms_norm_backward`、`aten::pow.Tensor_Scalar`、`aten::mean.dim`、`aten::add_`、`aten::rsqrt`、`aten::mul.Tensor` | `aten::rms_norm` 自身是 CIA 壳；CPU / fallback 路径完全继承 composite 展开链，CUDA contiguous 常走 fused backward |
| 158 | `torch.nn.functional.rrelu` | CIA 前向分解到 `rrelu_with_noise` | `RreluWithNoiseBackward0` | `self: rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)` | `aten::rrelu_with_noise_backward`、`aten::mul.Tensor`、`aten::leaky_relu_backward` | 训练态 helper 展开成 `noise * grad_output`；推理态 helper 退化为 `leaky_relu_backward` |
| 159 | `torch.nn.functional.soft_margin_loss` | helper / backward ATen op | `SoftMarginLossBackward0` | `self: soft_margin_loss_backward(grad, self, target, reduction)` | `aten::soft_margin_loss_backward` | 前向是 `CompositeExplicitAutograd`，但反向直接走专用 schema |
| 160 | `torch.nn.functional.softmin` | Python 组合前向，无专属 backward | 无统一 node；由 `NegBackward0` 与 `SoftmaxBackward0` 组合 | 无单独条目 | `aten::neg`、`aten::_softmax_backward_data` | `softmin(x)` 就是 `softmax(-x)`，因此反向等于 softmax backward 后再经过一层 `neg` |

## 详细分析

### 156. torch.nn.functional.pixel_unshuffle

**反向来源类型**：helper / backward ATen op

**derivatives.yaml 条目**：
```yaml
- name: pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
  self: pixel_shuffle(grad, downscale_factor)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::pixel_unshuffle` 生成 `PixelUnshuffleBackward0`
- `Functions.cpp` 中 `PixelUnshuffleBackward0::apply()` 直接调用 `pixel_shuffle(...)`

**反向 ATen 依赖**：
- `aten::pixel_shuffle`

**备注**：
- CPU 前向有专用 kernel；CUDA 等无专用 kernel 的前向可能分解为 `reshape -> permute -> clone -> view`，但 `pixel_unshuffle` 自身的 backward 公式依然是显式 `pixel_shuffle`。

---

### 157. torch.nn.functional.rms_norm

**反向来源类型**：CIA 前向分解到 `_fused_rms_norm` 或 composite 基础算子

**Forward 路径**：
- 公共入口：`aten::rms_norm`（CIA）→ `native::rms_norm_symint`
- 常见 CUDA contiguous：
  - `aten::_fused_rms_norm` → CUDA fused kernel
- CPU / fallback：
  - `aten::_fused_rms_norm` 的 `CompositeImplicitAutograd` fallback → `rms_norm_composite`
  - 展开为 `pow -> mean.dim -> add_ -> rsqrt -> mul.Tensor`

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::_fused_rms_norm` 生成 `FusedRmsNormBackward0`
- `Functions.cpp` 中：
```cpp
auto grad_result =
  GradMode::is_enabled() || grads[1].defined()
    ? infinitely_differentiable_native_rms_norm_backward(...)
    : _fused_rms_norm_backward(...)
```

**反向 ATen 依赖**：
- 一阶常见 fused 路径：
  - `aten::_fused_rms_norm_backward`
- fallback / 展开路径：
  - `aten::pow.Tensor_Scalar`
  - `aten::mean.dim`
  - `aten::add_`
  - `aten::rsqrt`
  - `aten::mul.Tensor`

**备注**：
- `aten::rms_norm` 自身不单独生成 backward node，梯度记录发生在内层 `_fused_rms_norm` 或 composite 展开的基础算子上。
- 高阶梯度时，generated node 会绕过 `_fused_rms_norm_backward`，改走 `infinitely_differentiable_native_rms_norm_backward` helper。

---

### 158. torch.nn.functional.rrelu

**反向来源类型**：CIA 前向分解到 `rrelu_with_noise`

**Forward 路径**：
`aten::rrelu`（CIA）→ `native::rrelu` → `aten::rrelu_with_noise`

**derivatives.yaml 条目**：
```yaml
- name: rrelu_with_noise(Tensor self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
  self: rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::rrelu_with_noise` 生成 `RreluWithNoiseBackward0`
- `Functions.cpp` 的 `apply()` 调用 `rrelu_with_noise_backward(...)`

**helper 展开**（`Activation.cpp`）：
- `training=True`：
  - `return noise * grad_output;`
  - 依赖 `aten::mul.Tensor`
- `training=False`：
  - `return at::leaky_relu_backward(grad_output, self_or_result, mid, is_result);`
  - 依赖 `aten::leaky_relu_backward`

**反向 ATen 依赖**：
- `aten::rrelu_with_noise_backward`
- `aten::mul.Tensor`
- `aten::leaky_relu_backward`

---

### 159. torch.nn.functional.soft_margin_loss

**反向来源类型**：helper / backward ATen op

**derivatives.yaml 条目**：
```yaml
- name: soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  self: soft_margin_loss_backward(grad, self, target, reduction)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `SoftMarginLossBackward0`
- `Functions.cpp` 中 `SoftMarginLossBackward0::apply()` 直接调用 `soft_margin_loss_backward(...)`

**反向 ATen 依赖**：
- `aten::soft_margin_loss_backward`

**备注**：
- native backward 实现在 `Loss.cpp`，前向虽然是 `CompositeExplicitAutograd`，但 backward 不再拆回 `neg/mul/exp/log1p` 链。

---

### 160. torch.nn.functional.softmin

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 路径**：
`softmin(input, dim)` = `softmax(-input, dim)`

**反向 ATen 依赖**：
- `aten::_softmax_backward_data`
- `aten::neg`

**备注**：
- `softmin` 没有独立的 `aten::softmin` schema，因此也没有独立 backward node。
- 梯度先由 softmax backward 回到 `-input`，再通过 `neg` 把符号翻回原输入。
