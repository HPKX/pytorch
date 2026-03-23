# torch API 121-125 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 121 | `torch.nn.RNN` | CIA 前向分解，多路径 | cuDNN：`CudnnRnnBackward0`；CPU/native：各子 op 独立 backward | `_cudnn_rnn`: `input, hx, cx, weight: _cudnn_rnn_backward_symint(...)`；native 路径无统一条目 | cuDNN：`aten::_cudnn_rnn_backward`；native：`aten::linear`、`aten::tanh_backward` / `aten::threshold_backward`、`aten::native_dropout_backward`、Packed 分支额外 `aten::cat` | `aten::rnn_tanh.*` / `aten::rnn_relu.*` 外层 schema 无 derivatives 条目 |
| 122 | `torch.nn.RNNCell` | CIA 前向分解，无专属 backward | 无统一 node（`LinearBackward0` + `TanhBackward0` / `ReluBackward0` 等） | 无条目；`rnn_tanh_cell` / `rnn_relu_cell` 都在 `RNN.cpp` 里展开为 `linear + linear + add_ + activation` | `aten::addmm` / `aten::mm` / `aten::t`、`aten::tanh_backward` 或 `aten::threshold_backward`、`aten::add.Tensor` | `RNNCell` 不走 cuDNN 整层 fused RNN 路径 |
| 123 | `torch.nn.RReLU` | 训练态：backward ATen op；推理态：复用 LeakyReLU backward | 训练态：`RreluWithNoiseBackward0`；推理态：`LeakyReluBackward0` | `rrelu_with_noise`: `self: rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)` | 训练态：`aten::rrelu_with_noise_backward`；推理态：`aten::leaky_relu_backward` | `rrelu` 外层是 CIA，训练态落到 `rrelu_with_noise`，eval 则退化成 `leaky_relu.out` |
| 124 | `torch.nn.SoftMarginLoss` | backward ATen op | `SoftMarginLossBackward0` | `self: soft_margin_loss_backward(grad, self, target, reduction)` | `aten::soft_margin_loss_backward` | `soft_margin_loss` 和 `soft_margin_loss_backward` 都是 `CompositeExplicitAutograd` |
| 125 | `torch.nn.Softmax2d` | 外层 CIA → 内层 `_softmax` 的 backward ATen op | `SoftmaxBackward0` | `_softmax`: `self: _softmax_backward_data(grad, result, dim, self.scalar_type())` | `aten::_softmax_backward_data` | `nn.Softmax2d` 只是把 `dim` 固定成 `-3` 的 `softmax.int` 包装 |

## 详细分析

### 121. torch.nn.RNN

**反向来源类型**：CIA 前向分解，多路径

**锁定 schema**：
- `aten::rnn_tanh.input`
- `aten::rnn_tanh.data`
- `aten::rnn_relu.input`
- `aten::rnn_relu.data`

这些外层 schema 都没有 `derivatives.yaml` 条目。

**当前 forward 文档覆盖的两类路径**：

1. **CUDA cuDNN 路径**
   - 内层真实求导 op：`aten::_cudnn_rnn`
   - backward node：`CudnnRnnBackward0`
   - 反向依赖：`aten::_cudnn_rnn_backward`

2. **CPU/native 路径**
   - 分解为 `aten::linear`、`aten::add_`、`aten::tanh` / `aten::relu`
   - 多层训练时还会插入 `aten::dropout`
   - Packed 分支双向时还会有 `aten::cat`

**反向 ATen 依赖**：
- cuDNN：
  - `aten::_cudnn_rnn_backward`
- native：
  - `aten::addmm` / `aten::mm` / `aten::t`（来自 `linear`）
  - `aten::tanh_backward` 或 `aten::threshold_backward`
  - `aten::native_dropout_backward`
  - `aten::cat`

**备注**：
- 外层 `rnn_tanh.*` / `rnn_relu.*` 不会生成统一的 whole-RNN backward node
- 当前文档遵循对应 forward 文档：同时承认 cuDNN 和 native 两类常见路径

---

### 122. torch.nn.RNNCell

**反向来源类型**：CIA 前向分解，无专属 backward

**锁定 schema**：
- `aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor`
- `aten::rnn_relu_cell(...) -> Tensor`

**derivatives.yaml**：
- 两个 schema 都**无条目**

**native 实现**（`RNN.cpp`）：
- `rnn_tanh_cell`：`linear(input, w_ih, b_ih) + linear(hx, w_hh, b_hh)` 再 `tanh`
- `rnn_relu_cell`：同构，只是最终激活是 `relu`

因此 autograd 完全依赖子 op：

**反向 ATen 依赖**：
- `aten::addmm` / `aten::mm` / `aten::t`
- `aten::add.Tensor`
- `aten::tanh_backward`
- `aten::threshold_backward`

**备注**：
- 没有统一的 `RnnCellBackward0`

---

### 123. torch.nn.RReLU

**反向来源类型**：训练态是 backward ATen op；推理态复用 `LeakyReLU` backward

**锁定 schema**：
- 外层：`aten::rrelu(Tensor self, Scalar lower=..., Scalar upper=..., bool training=False, Generator? generator=None) -> Tensor`
- 训练态内层：`aten::rrelu_with_noise(...) -> Tensor`
- backward：`aten::rrelu_with_noise_backward(...) -> Tensor`

**derivatives.yaml 关键条目**：

```yaml
- name: rrelu_with_noise(...) -> Tensor
  self: rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `rrelu_with_noise` 建立 `RreluWithNoiseBackward0`
- `Functions.cpp` 里 `RreluWithNoiseBackward0::apply()` 调用 `rrelu_with_noise_backward(...)`

**反向 ATen 依赖**：
- 训练态：
  - `aten::rrelu_with_noise_backward`
- 推理态：
  - `aten::leaky_relu_backward`

**备注**：
- `rrelu` 自己是 CIA wrapper；`training=False` 时 native 分支直接退化成 `leaky_relu.out`

---

### 124. torch.nn.SoftMarginLoss

**反向来源类型**：backward ATen op

**锁定 schema**：
- forward：`aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor`
- backward：`aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  self: soft_margin_loss_backward(grad, self, target, reduction)
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `soft_margin_loss` 建立 `SoftMarginLossBackward0`
- `Functions.cpp` 中：

```cpp
auto grad_result = any_grad_defined ? (soft_margin_loss_backward(grad, self, target, reduction)) : Tensor();
```

**native helper 实现**（`Loss.cpp`）：
- `soft_margin_loss_backward_out` 内部使用：
  - `aten::exp`
  - `aten::mul_out` / `aten::mul_`
  - `aten::add_`
  - `aten::div_`

**反向 ATen 依赖**：
- 直接依赖：`aten::soft_margin_loss_backward`
- helper 展开后依赖：`aten::exp`、`aten::mul`、`aten::add`、`aten::div`

---

### 125. torch.nn.Softmax2d

**反向来源类型**：外层 CIA → 内层 `_softmax` 的 backward ATen op

**锁定 schema**：
- Python 层实际入口：`aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor`
- 内层真正求导：`aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor`
- backward：`aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  self: _softmax_backward_data(grad, result, dim, self.scalar_type())
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `_softmax` 建立 wrapper
- `Functions.cpp` 的 node 名称是 `SoftmaxBackward0`

**反向 ATen 依赖**：
- `aten::_softmax_backward_data`

**备注**：
- `nn.Softmax2d` 只是把 `dim=-3` 固定下来的模块包装

