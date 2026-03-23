# torch API 91-95 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 91 | `torch.nn.ConvTranspose3d` | CIA 前向分解 → `aten::convolution` 的 backward ATen op | `ConvolutionBackward0`（挂在内部 `aten::convolution` 上） | `input, weight, bias: convolution_backward_symint(grad, input, weight, bias->sym_sizes(), stride, padding, dilation, transposed, output_padding, groups, grad_input_mask)` | `aten::convolution_backward`（CEA，内部分发到 backend-specific backward） | 与 `ConvTranspose1d` 相同机制；`conv_transpose3d` CIA → `convolution(transposed=true)` |
| 92 | `torch.nn.Dropout1d` | CIA 前向分解，无专属 backward | 无统一 backward node（子 op `MulBackward0` 等） | 无条目（CIA：`feature_dropout` 分解为 `bernoulli_` → `div_` → `mul`） | `aten::mul.Tensor` 反向：`mul_tensor_backward` → `aten::mul.Tensor`、`aten::conj` | `feature_dropout` CIA 逐元素乘 mask；反向来自 `mul.Tensor` 的 backward |
| 93 | `torch.nn.Dropout2d` | CIA 前向分解，无专属 backward | 无统一 backward node（子 op `MulBackward0` 等） | 无条目（同 Dropout1d） | `aten::mul.Tensor` 反向：`mul_tensor_backward` → `aten::mul.Tensor`、`aten::conj` | 与 Dropout1d 完全相同的反向机制 |
| 94 | `torch.nn.Dropout3d` | CIA 前向分解，无专属 backward | 无统一 backward node（子 op `MulBackward0` 等） | 无条目（同 Dropout1d） | `aten::mul.Tensor` 反向：`mul_tensor_backward` → `aten::mul.Tensor`、`aten::conj` | 与 Dropout1d 完全相同的反向机制 |
| 95 | `torch.nn.GRU` | CIA 前向分解 → 内部 op 各自反向 | CUDA cuDNN：`CudnnRnnBackward0`；CUDA fused：`ThnnFusedGruCellBackward0`；CPU fallback：各子 op 独立 backward | cuDNN：`_cudnn_rnn_backward(...)`；fused：`_thnn_differentiable_gru_cell_backward(...)` / `_thnn_fused_gru_cell_backward(...)`；CPU fallback：各子 op 各自公式 | cuDNN：`aten::_cudnn_rnn_backward`；fused：`aten::_thnn_fused_gru_cell_backward` 或 `aten::_thnn_differentiable_gru_cell_backward`；CPU fallback：`aten::sigmoid_backward`、`aten::tanh_backward` 等 | `gru` CIA 分解为不同路径；cuDNN 不支持 double backward |

## 详细分析

### 91. torch.nn.ConvTranspose3d

**反向来源类型**：CIA 前向分解 → `aten::convolution` 的 backward ATen op

**Forward 分解路径**：`aten::conv_transpose3d.input` (CIA) → `aten::convolution(transposed=true)` → `aten::_convolution` → backend-specific impl（cuDNN/MIOpen/slow_conv_transpose3d 等）

与 `ConvTranspose1d`（接口 90）机制完全相同。autograd 在 `aten::convolution` 层面录制 backward。

**`convolution` 的 derivatives.yaml 条目**：
```yaml
- name: convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
  input, weight, bias: "grad.defined() ? convolution_backward_symint(grad, input, weight, bias->sym_sizes(), stride, padding, dilation, transposed, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
```

**Backward Node**：`ConvolutionBackward0`

**反向 ATen 依赖**：
- `aten::convolution_backward` — CEA op，内部分发到具体后端的卷积反向 kernel

---

### 92. torch.nn.Dropout1d

**反向来源类型**：CIA 前向分解，无专属 backward

**Forward 分解路径**：`aten::feature_dropout` (CIA) → `input.new_empty_symint(...)` 构造 channel mask → `aten::bernoulli_` → `aten::div_` → `aten::mul.Tensor`

`feature_dropout` 是 `CompositeImplicitAutograd`，没有 derivatives.yaml 条目。关键的梯度传递来自 `aten::mul.Tensor`（input 乘以 dropout mask）的 backward。

**`mul.Tensor` 的 derivatives.yaml 条目**：
```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
```

由于 dropout mask 不需要梯度（由 `bernoulli_` 生成，无 autograd 历史），实际只计算 input 的梯度：`grad * mask.conj()`（mask 为实数时 conj 为 no-op）。

**反向 ATen 依赖**：
- `aten::mul.Tensor` — `grad * mask`（来自 `mul_tensor_backward`）
- `aten::conj` — 复数共轭（实数时为 no-op view）

**条件依赖**（非 batched 输入）：
- `aten::unsqueeze` 反向 → `aten::squeeze`
- `aten::squeeze` 反向 → `aten::unsqueeze`

---

### 93. torch.nn.Dropout2d

**反向来源类型**：CIA 前向分解，无专属 backward

与 `Dropout1d`（接口 92）完全相同的机制。`F.dropout2d` 同样调用 `aten::feature_dropout`。

**反向 ATen 依赖**：
- `aten::mul.Tensor` — `grad * mask`
- `aten::conj` — 复数共轭（实数时为 no-op view）

---

### 94. torch.nn.Dropout3d

**反向来源类型**：CIA 前向分解，无专属 backward

与 `Dropout1d`（接口 92）完全相同的机制。`F.dropout3d` 同样调用 `aten::feature_dropout`。

**反向 ATen 依赖**：
- `aten::mul.Tensor` — `grad * mask`
- `aten::conj` — 复数共轭（实数时为 no-op view）

**条件依赖**（非 batched 输入）：
- `aten::unsqueeze` 反向 → `aten::squeeze`
- `aten::squeeze` 反向 → `aten::unsqueeze`

---

### 95. torch.nn.GRU

**反向来源类型**：CIA 前向分解 → 内部 op 各自反向（多路径）

**Forward 分解路径**：`aten::gru.input` / `aten::gru.data` (CIA) → 多个路径：

1. **CUDA cuDNN 路径**：→ `aten::_cudnn_rnn` → backend kernel
2. **CUDA fused 路径**：→ `aten::_thnn_fused_gru_cell` → CUDA kernel
3. **CPU fallback 路径**：→ `aten::linear` → `aten::unsafe_chunk` → `aten::sigmoid_`/`aten::tanh_`/`aten::mul_`/`aten::add_`/`aten::sub`

#### 路径 1：cuDNN

**`_cudnn_rnn` 的 derivatives.yaml 条目**：
```yaml
- name: _cudnn_rnn(...) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  output_differentiability: [True, True, True, False, False]
  input, hx, cx, weight: "_cudnn_rnn_backward_symint(input, weight, weight_stride0, result4, hx, cx, result0, grads[0], grads[1], grads[2], mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, retain_variables ? result3.clone() : result3, grad_input_mask)"
```

**Backward Node**：`CudnnRnnBackward0`

**反向 ATen 依赖**：
- `aten::_cudnn_rnn_backward` — CUDA backend kernel（cuDNN 实现）

注意：cuDNN RNN backward 不支持 double backward（`not_implemented`）。

#### 路径 2：CUDA fused GRU cell

**`_thnn_fused_gru_cell` 的 derivatives.yaml 条目**：
```yaml
- name: _thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
  input_gates, hidden_gates, hx, input_bias, hidden_bias: "grad.defined() ? (GradMode::is_enabled() ? _thnn_differentiable_gru_cell_backward(grad, input_gates, hidden_gates, hx, input_bias, hidden_bias) : _thnn_fused_gru_cell_backward(grad, result1, input_bias.defined())) : std::tuple<...>()"
```

**Backward Node**：`ThnnFusedGruCellBackward0`

根据是否启用 grad mode 选择不同反向实现：
- `GradMode::is_enabled()` → `_thnn_differentiable_gru_cell_backward`（支持 double backward）
- 否则 → `_thnn_fused_gru_cell_backward`（CUDA fused kernel，不支持 double backward）

**`_thnn_differentiable_gru_cell_backward` 实现**（`RNN.cpp:1613`）：
```cpp
// 展开 GRU cell 的反向：
// chunked_input_gates = (in_g + input_bias).unsafe_chunk(3, 1)
// rg = sigmoid(ir + hr), ig = sigmoid(ii + hi), ng = tanh(in + rg*hn)
// grad_hx = grad_hy * ig
// gig = sigmoid_backward(grad_hy * (hx - ng), ig)
// gin = tanh_backward(grad_hy * (1 - ig), ng)
// ghn = gin * rg, grg = sigmoid_backward(gin * hn, rg)
// grad_input_gates = cat({grg, gig, gin}, 1)
// grad_hidden_gates = cat({grg, gig, ghn}, 1)
```

**反向 ATen 依赖（differentiable 路径）**：
- `aten::add.Tensor` — bias 加法
- `aten::unsafe_chunk` — 切分 gates
- `aten::sigmoid` — 重新计算门控
- `aten::tanh` — 重新计算激活
- `aten::sigmoid_backward` — sigmoid 的局部梯度
- `aten::tanh_backward` — tanh 的局部梯度
- `aten::mul.Tensor` — 多次相乘
- `aten::sub.Tensor` — `hx - ng`、`1 - ig`
- `aten::cat` — 拼接梯度
- `aten::sum` — bias 梯度（沿 batch 维求和）

**反向 ATen 依赖（fused 路径）**：
- `aten::_thnn_fused_gru_cell_backward` — CUDA backend kernel

#### 路径 3：CPU fallback

CPU fallback 路径中 GRU 展开为基础 ATen op（`linear`、`sigmoid_`、`tanh_`、`mul_`、`add_`、`sub`），每个 op 有各自的 backward node，不产生统一的 GRU backward。
