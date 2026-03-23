# torch API 106-110 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 106 | `torch.nn.LSTM` | CIA 前向分解，当前区间锁定 cuDNN fast path | `CudnnRnnBackward0`（挂在内层 `aten::_cudnn_rnn` 上） | `_cudnn_rnn`: `input, hx, cx, weight: _cudnn_rnn_backward_symint(...)` | `aten::_cudnn_rnn_backward` | 外层 `aten::lstm.input` / `aten::lstm.data` 无 derivatives 条目；常见 CUDA fast path 的真实 backward 落在 `_cudnn_rnn`；double backward 未实现 |
| 107 | `torch.nn.LSTMCell` | CIA 前向分解 → CUDA fused cell backward ATen op / differentiable helper | `ThnnFusedLstmCellBackward0` | `_thnn_fused_lstm_cell`: `GradMode::is_enabled() ? _thnn_differentiable_lstm_cell_backward(...) : _thnn_fused_lstm_cell_backward(...)` | `aten::_thnn_differentiable_lstm_cell_backward`、`aten::_thnn_fused_lstm_cell_backward`、上游 gate GEMM 的 `aten::matmul`/`aten::t` | 外层 `aten::lstm_cell` 无专属 backward；forward 先做两路 gate GEMM，再进入 fused cell |
| 108 | `torch.nn.LeakyReLU` | backward ATen op | `LeakyReluBackward0` | `self: leaky_relu_backward(grad, self, negative_slope, false)` | `aten::leaky_relu_backward` | 非 inplace 模块路径；inplace 版本才会落到 `LeakyReluBackward1` |
| 109 | `torch.nn.MarginRankingLoss` | CIA 前向分解，无专属 backward | 无统一 node（各基础 op 独立 backward） | 无条目；forward composite 为 `sub -> neg+mul -> add.Scalar -> clamp_min_ / clamp_min -> reduction` | `aten::sub.Tensor`、`aten::neg`、`aten::mul.Tensor`、`aten::clamp_min`、`aten::mean` / `aten::sum` | `aten::margin_ranking_loss` 注册在 `CompositeImplicitAutograd`；`target` 通常不参与梯度 |
| 110 | `torch.nn.MaxPool1d` | CIA 前向分解，无专属 `max_pool1d` backward node | 主要为 `MaxPool2DWithIndicesBackward0` + view backward | `max_pool2d_with_indices`: `self: max_pool2d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)` | `aten::max_pool2d_with_indices_backward`、`as_strided_backward` | `max_pool1d` / `max_pool1d_with_indices` 在 `Pooling.cpp` 分解为 `unsqueeze(-2) -> max_pool2d_with_indices -> squeeze(-2)` |

## 详细分析

### 106. torch.nn.LSTM

**反向来源类型**：CIA 前向分解，当前区间锁定 `torch_api_106_110.md` 里的 CUDA cuDNN fast path

**锁定 schema**：
- Python 入口：`_VF.lstm`
- ATen schema：`aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)`
- 内层真实求导 schema：`aten::_cudnn_rnn(...) -> (Tensor, Tensor, Tensor, Tensor, Tensor)`

**view / inplace 检查**：`lstm.input` / `_cudnn_rnn` 都不是 view / inplace schema。

**derivatives.yaml**：
- `aten::lstm.input` / `aten::lstm.data` **无条目**
- `aten::_cudnn_rnn` 有显式条目：

```yaml
- name: _cudnn_rnn(...) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  output_differentiability: [True, True, True, False, False]
  input, hx, cx, weight: "_cudnn_rnn_backward_symint(...)"
```

**forward dispatch 结论**：
- `aten::lstm.input` 是 `CompositeImplicitAutograd`
- 当前参考前向文档锁定的是 cuDNN fast path，所以 eager autograd 不会在外层 `aten::lstm.input` 结束，而是把梯度记录在真正执行到的 `aten::_cudnn_rnn` 上

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `_cudnn_rnn` 建立 `CudnnRnnBackward0`
- `Functions.cpp` 里 `CudnnRnnBackward0::apply()` 直接调用 `_cudnn_rnn_backward_symint(...)`

**反向 ATen 依赖**：
- `aten::_cudnn_rnn_backward`

**备注**：
- `_cudnn_rnn_backward_symint` 规范化后对应 schema `aten::_cudnn_rnn_backward`
- `tools/autograd/derivatives.yaml` 同时说明 `_cudnn_rnn_backward` 的 double backward 为 `not_implemented`
- 若未来分析 CPU/native 路径，则反向会退化到 `linear`、`sigmoid`、`tanh`、`mul`、`dropout` 等子 op 的组合；本文件按对应 forward 文档只记录 cuDNN 场景

---

### 107. torch.nn.LSTMCell

**反向来源类型**：CIA 前向分解 → CUDA fused cell backward ATen op / differentiable helper

**锁定 schema**：
- 外层 schema：`aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)`
- 内层 fused schema：`aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)`

**view / inplace 检查**：都不是 view / inplace。

**forward 路径**（锁定前向文档里的 CUDA fused 路径）：
`aten::lstm_cell` (CIA) → 两路 gate GEMM：
- `aten::t` + `aten::matmul(input, w_ih.t())`
- `aten::t` + `aten::matmul(hx, w_hh.t())`

然后进入：
- `aten::_thnn_fused_lstm_cell`

**derivatives.yaml 条目**：

```yaml
- name: _thnn_fused_lstm_cell(...) -> (Tensor, Tensor, Tensor)
  input_gates, hidden_gates, cx, input_bias, hidden_bias:
    "GradMode::is_enabled() ? _thnn_differentiable_lstm_cell_backward(...)
                            : _thnn_fused_lstm_cell_backward(...)"
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `_thnn_fused_lstm_cell` 建立 `ThnnFusedLstmCellBackward0`
- `Functions.cpp` 中 `ThnnFusedLstmCellBackward0::apply()` 精确复现上述分支

**立即反向依赖**：
- `aten::_thnn_differentiable_lstm_cell_backward`
- `aten::_thnn_fused_lstm_cell_backward`

**helper 展开后的反向依赖**（`RNN.cpp::_thnn_differentiable_lstm_cell_backward`）：
- `aten::add.Tensor`
- `aten::sigmoid`
- `aten::tanh`
- `aten::sigmoid_backward`
- `aten::tanh_backward`
- `aten::zeros_like`
- `aten::mul.Tensor`
- `aten::cat`
- `aten::sum`

**备注**：
- 上游两路 gate GEMM 的梯度仍由 `aten::matmul` / `aten::t` 各自 backward 负责
- `_thnn_fused_lstm_cell_backward` 是 fused CUDA kernel 路径
- `_thnn_differentiable_lstm_cell_backward` 是支持更高阶求导的 ATen helper 路径

---

### 108. torch.nn.LeakyReLU

**反向来源类型**：backward ATen op

**锁定 schema**：
- forward：`aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor`
- backward：`aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
  self: leaky_relu_backward(grad, self, negative_slope, false)
```

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `aten::leaky_relu` 建立 `LeakyReluBackward0`
- `Functions.cpp` 中：

```cpp
auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, self, negative_slope, false)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::leaky_relu_backward`

**备注**：
- 当前接口是非 inplace 模块 `nn.LeakyReLU(inplace=False)`，因此不会走 `LeakyReluBackward1`
- `leaky_relu_backward` 自身是 structured backward op，CPU/CUDA/MPS 都有 backend 实现

---

### 109. torch.nn.MarginRankingLoss

**反向来源类型**：CIA 前向分解，无专属 backward

**锁定 schema**：
- `aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor`

**view / inplace 检查**：不是 view；forward 内部可能在 composite body 中选 `clamp_min_` 或 `clamp_min`，但对外 schema 不是 inplace。

**derivatives.yaml**：
- `aten::margin_ranking_loss` **无条目**

**native/composite 实现**（`Loss.cpp`）：

```cpp
auto unclamped_output = (-target * (input1 - input2) + margin);
auto output = ... ? unclamped_output.clamp_min(0)
                  : unclamped_output.clamp_min_(0);
return apply_loss_reduction(output, reduction);
```

因此 autograd 记录的是 composite 体内各基础 ATen op，而不是单独的 `MarginRankingLossBackward0`。

**反向 ATen 依赖**：
- `aten::sub.Tensor`
- `aten::neg`
- `aten::mul.Tensor`
- `aten::clamp_min`
- `aten::mean` / `aten::sum`

**备注**：
- `target` 是 label tensor，通常不要求梯度
- 对 `input1` / `input2` 的梯度主链来自 `sub`、`mul` 和 `clamp_min` 的各自 backward

---

### 110. torch.nn.MaxPool1d

**反向来源类型**：CIA 前向分解，无专属 `max_pool1d` backward node

**锁定 schema**：
- 外层：`aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor`
- 内层：`aten::max_pool1d_with_indices(...) -> (Tensor, Tensor)`，进一步分解到 `aten::max_pool2d_with_indices`

**view / inplace 检查**：
- 前向含 `unsqueeze(-2)` / `squeeze(-2)`，因此有 view backward

**derivatives.yaml**：
- `aten::max_pool1d` / `aten::max_pool1d_with_indices` **无条目**
- 真正挂梯度的是 `aten::max_pool2d_with_indices`：

```yaml
- name: max_pool2d_with_indices(...) -> (Tensor, Tensor)
  self: max_pool2d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)
```

**native 实现验证**（`Pooling.cpp`）：

```cpp
auto [output, indices] = at::max_pool2d_with_indices(
    self.unsqueeze(-2),
    {1, kernel_size[0]},
    {1, stride[0]},
    {0, padding[0]},
    {1, dilation[0]},
    ceil_mode);
output = output.squeeze(-2);
indices = indices.squeeze(-2);
```

**反向 ATen 依赖**：
- `aten::max_pool2d_with_indices_backward`
- `as_strided_backward`

**备注**：
- 这里没有统一的 `MaxPool1DBackward0`
- 真实 backward node 来自内层 `max_pool2d_with_indices`，外加两层 view 的 backward

