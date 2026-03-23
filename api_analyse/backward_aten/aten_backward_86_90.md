# torch API 86-90 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 86 | `torch.nn.AvgPool1d` | CIA 前向分解 → `aten::avg_pool2d` 的 backward ATen op | `AvgPool2DBackward0`（挂在内部 `avg_pool2d` 上） | `self: avg_pool2d_backward(grad, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)` | `aten::avg_pool2d_backward`（structured，CPU/CUDA/MPS backend kernel） | `avg_pool1d` CIA 分解为 `unsqueeze` → `avg_pool2d` → `squeeze`；反向挂在 `avg_pool2d` 上；squeeze/unsqueeze 各自有独立 backward |
| 87 | `torch.nn.CELU` | backward ATen op（经 `aten::celu` → `aten::elu`） | `CeluBackward0`（挂在 `celu` 上） | `self: elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), false, self)` | `aten::elu_backward`（structured，CPU/CUDA/MPS backend kernel） | `celu` CEA 分解为 `elu`；但 derivatives.yaml 在 `celu` 上单独注册了 backward，不走 `elu` 的 backward |
| 88 | `torch.nn.CTCLoss` | CIA 前向分解 → `_ctc_loss` 的 backward ATen op | `CtcLossBackward0`/`CtcLossBackward1`（挂在内部 `_ctc_loss` 上） | `log_probs: _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)` | `aten::_ctc_loss_backward`（CPU: `ctc_loss_backward_cpu`；CUDA: `ctc_loss_backward_gpu`） | `ctc_loss` CIA → `_ctc_loss`；cuDNN/MIOpen 路径有各自独立 backward |
| 89 | `torch.nn.ChannelShuffle` | inline 公式（self-inverse） | `ChannelShuffleBackward0` | `self: channel_shuffle_symint(grad, grad.sym_size(1) / groups)` | `aten::channel_shuffle`（反向再次调用 channel_shuffle 自身，groups 取 C/groups） | channel_shuffle 是自逆的：反向就是用不同 groups 值做一次 channel_shuffle |
| 90 | `torch.nn.ConvTranspose1d` | CIA 前向分解 → `aten::convolution` 的 backward ATen op | `ConvolutionBackward0`（挂在内部 `aten::convolution` 上） | `input, weight, bias: convolution_backward_symint(grad, input, weight, bias->sym_sizes(), stride, padding, dilation, transposed, output_padding, groups, grad_input_mask)` | `aten::convolution_backward`（CEA，内部分发到 backend-specific backward） | `conv_transpose1d` CIA → `convolution(transposed=true)`；backward 挂在 `convolution` 上 |

## 详细分析

### 86. torch.nn.AvgPool1d

**反向来源类型**：CIA 前向分解 → `aten::avg_pool2d` 的 backward ATen op

**Forward 分解路径**：`aten::avg_pool1d` (CIA) → `aten::unsqueeze` → `aten::avg_pool2d` → `aten::squeeze`

`avg_pool1d` 是 `CompositeImplicitAutograd`，将 1D 输入 unsqueeze 为 2D，调用 `avg_pool2d`，再 squeeze 回 1D。autograd 对每个子 op 独立录制，核心梯度来自 `avg_pool2d` 的 backward。

**`avg_pool2d` 的 derivatives.yaml 条目**：
```yaml
- name: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
  self: avg_pool2d_backward(grad, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
  result: auto_linear
```

**Backward Node**：`AvgPool2DBackward0`

**`avg_pool2d_backward`** 是 structured ATen op，有独立的 CPU/CUDA/MPS backend kernel，不再分解为其他 ATen op。

**反向 ATen 依赖（主路径）**：
- `aten::avg_pool2d_backward` — 计算 avg_pool2d 的输入梯度（backend kernel）

**反向 ATen 依赖（view op）**：
- `aten::unsqueeze` 的反向：`aten::squeeze`
- `aten::squeeze` 的反向：`aten::unsqueeze`

---

### 87. torch.nn.CELU

**反向来源类型**：backward ATen op

**Forward 路径**：`aten::celu` (CEA) → `aten::elu` → backend kernel（`elu_stub`）

虽然 `celu` 前向分解为 `elu`，但 derivatives.yaml 在 `celu` 上单独注册了 backward 公式，因此 autograd 在 `celu` 层面而非 `elu` 层面录制。

**derivatives.yaml 条目**：
```yaml
- name: celu(Tensor self, Scalar alpha=1.0) -> Tensor
  self: elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ false, self)
  result: auto_element_wise
```

**Backward Node**：`CeluBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), false, self);
```

**`elu_backward`** 是 structured ATen op，有独立的 CPU/CUDA/MPS backend kernel。

**反向 ATen 依赖**：
- `aten::elu_backward` — 整体作为 backend kernel 执行

---

### 88. torch.nn.CTCLoss

**反向来源类型**：CIA 前向分解 → `_ctc_loss` 的 backward ATen op

**Forward 分解路径**：
- `aten::ctc_loss.IntList` / `aten::ctc_loss.Tensor` (CIA) → `aten::_ctc_loss` → CPU/CUDA backend
- cuDNN 路径：→ `aten::_cudnn_ctc_loss` → cuDNN backend
- MIOpen 路径：→ `aten::miopen_ctc_loss` → MIOpen backend

**`_ctc_loss` 的 derivatives.yaml 条目**：
```yaml
- name: _ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  log_probs: _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)

- name: _ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  log_probs: _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)
```

**Backward Node**：`CtcLossBackward0`（IntList 重载）/ `CtcLossBackward1`（Tensor 重载）

**`_ctc_loss_backward`** 是 ATen op，有 CPU 和 CUDA 独立的 backend kernel：
- CPU: `ctc_loss_backward_cpu`
- CUDA: `ctc_loss_backward_gpu`

cuDNN 路径的 backward 使用 `_miopen_ctc_loss_backward`（MIOpen）或 cuDNN 提供的梯度。

**反向 ATen 依赖**：
- `aten::_ctc_loss_backward` — 整体作为 backend kernel 执行

---

### 89. torch.nn.ChannelShuffle

**反向来源类型**：inline 公式（self-inverse）

**Forward 路径**：`aten::channel_shuffle` → Autograd wrapper → backend dispatch（CPU: `native_channel_shuffle` → 专用 kernel；CUDA: `native_channel_shuffle` → CIA 分解为 `view` → `permute` → `contiguous` → `reshape`）

**derivatives.yaml 条目**：
```yaml
- name: channel_shuffle(Tensor self, SymInt groups) -> Tensor
  self: channel_shuffle_symint(grad, grad.sym_size(1) / groups)
  result: auto_linear
```

**Backward Node**：`ChannelShuffleBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = channel_shuffle_symint(grad, grad.sym_size(1) / groups);
```

channel_shuffle 的反向操作就是再做一次 channel_shuffle，但 groups 参数取 `C / groups`（其中 C 是通道数）。这是因为 channel_shuffle 将通道分成 `groups` 组然后交错排列，逆操作就是以 `C/groups` 为 groups 再做一次。

**反向 ATen 依赖**：
- `aten::channel_shuffle` — 反向再次调用 channel_shuffle 自身（groups 取 `C/groups`）
  - CPU 路径内部：`aten::native_channel_shuffle` → 专用 kernel
  - CUDA 路径内部：`aten::native_channel_shuffle` → `aten::view` → `aten::permute` → `aten::contiguous` → `aten::reshape`

---

### 90. torch.nn.ConvTranspose1d

**反向来源类型**：CIA 前向分解 → `aten::convolution` 的 backward ATen op

**Forward 分解路径**：`aten::conv_transpose1d` (CIA) → `aten::convolution(transposed=true)` → `aten::_convolution` → backend-specific convolution-transpose impl

autograd 在 `aten::convolution` 层面录制 backward。

**`convolution` 的 derivatives.yaml 条目**：
```yaml
- name: convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
  input, weight, bias: "grad.defined() ? convolution_backward_symint(grad, input, weight, bias->sym_sizes(), stride, padding, dilation, transposed, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
```

**Backward Node**：`ConvolutionBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = convolution_backward_symint(grad, input, weight, bias_sym_sizes_opt,
    stride, padding, dilation, transposed, output_padding, groups, grad_input_mask);
```

**`convolution_backward`** 是 CEA ATen op，内部分发到 backend-specific backward：
- CUDA: cuDNN/MIOpen 后端的卷积反向
- CPU: 对应的 CPU 卷积反向
- MPS: `mps_convolution_backward`

**反向 ATen 依赖**：
- `aten::convolution_backward` — 整体作为 CEA op 执行，内部分发到具体后端的卷积反向 kernel
