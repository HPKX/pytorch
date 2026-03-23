# torch API 121-125 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.nn.RNN` | dense 输入，`nonlinearity='tanh'/'relu'` | `nn.RNN.forward -> _VF.rnn_tanh/_VF.rnn_relu -> aten::rnn_tanh.input / aten::rnn_relu.input -> CompositeImplicitAutograd -> at::native::rnn_* -> CUDA 且命中 cuDNN 时进入 aten::_cudnn_rnn；否则走 _rnn_impl_with_concat -> aten::linear -> aten::add_ -> aten::tanh / aten::relu`，多层训练时层间还会插入 `aten::dropout` | `aten::rnn_tanh.input`, `aten::rnn_relu.input`, `aten::_cudnn_rnn`, `aten::linear`, `aten::add_`, `aten::tanh`, `aten::relu`, `aten::dropout` |
| `torch.nn.RNN` | `PackedSequence` 输入 | `nn.RNN.forward(PackedSequence) -> _VF.rnn_tanh/_VF.rnn_relu -> aten::rnn_tanh.data / aten::rnn_relu.data -> CompositeImplicitAutograd -> at::native::rnn_* -> CUDA 且命中 cuDNN 时进入 aten::_cudnn_rnn；否则走 PackedLayer/PackedBidirectionalLayer -> aten::linear -> aten::add_ -> aten::tanh / aten::relu`，双向时用 `aten::cat` 拼回 packed `data` | `aten::rnn_tanh.data`, `aten::rnn_relu.data`, `aten::_cudnn_rnn`, `aten::linear`, `aten::add_`, `aten::tanh`, `aten::relu`, `aten::cat` |
| `torch.nn.RNNCell` | 单步 cell，`nonlinearity='tanh'/'relu'` | `nn.RNNCell.forward -> _VF.rnn_tanh_cell/_VF.rnn_relu_cell -> aten::rnn_tanh_cell / aten::rnn_relu_cell -> CompositeImplicitAutograd -> at::native::rnn_*_cell -> aten::linear(input, w_ih, b_ih) + aten::linear(hx, w_hh, b_hh) -> aten::add_ -> aten::tanh / aten::relu` | `aten::rnn_tanh_cell`, `aten::rnn_relu_cell`, `aten::linear`, `aten::add_`, `aten::tanh`, `aten::relu` |
| `torch.nn.RReLU` | 训练态，`inplace=False` | `nn.RReLU.forward -> F.rrelu -> torch.rrelu -> aten::rrelu -> CompositeImplicitAutograd -> at::native::rrelu -> aten::rrelu_with_noise -> AutogradCPU/CUDA -> CPU/CUDA kernel`，内部生成并保存 `noise` 供 backward 复用 | `aten::rrelu`, `aten::rrelu_with_noise` |
| `torch.nn.RReLU` | 推理态 / `eval()` | `nn.RReLU.forward -> F.rrelu -> torch.rrelu -> aten::rrelu -> CompositeImplicitAutograd -> at::native::rrelu -> aten::rrelu_with_noise -> CPU/CUDA native 分支 -> aten::leaky_relu.out` | `aten::rrelu`, `aten::rrelu_with_noise`, `aten::leaky_relu.out` |
| `torch.nn.SoftMarginLoss` | 默认 `reduction='mean'` | `nn.SoftMarginLoss.forward -> F.soft_margin_loss -> torch._C._nn.soft_margin_loss -> aten::soft_margin_loss -> AutogradCPU/CUDA -> CompositeExplicitAutograd -> at::native::soft_margin_loss -> aten::neg.out -> aten::mul_ -> aten::exp_ -> aten::log1p_ -> aten::mean` | `aten::soft_margin_loss`, `aten::neg.out`, `aten::mul_`, `aten::exp_`, `aten::log1p_`, `aten::mean` |
| `torch.nn.Softmax2d` | `NCHW/CHW` dense 输入，沿通道维 `dim=-3` | `nn.Softmax2d.forward -> F.softmax(input, -3) -> Tensor.softmax(-3) -> aten::softmax.int -> CompositeImplicitAutograd -> at::native::softmax -> aten::_softmax`，若需要梯度则先过 `AutogradCPU/CUDA`，随后进入 `CPU/CUDA` structured kernel（`meta + impl`） | `aten::softmax.int`, `aten::_softmax` |

## 备注

- `RNN`/`RNNCell` 外层 `aten::rnn_*` 系列本身没有单独的标准 backward wrapper；CUDA/cuDNN 路径的梯度记录点在内层 `aten::_cudnn_rnn`，CPU/native 路径则由分解出的 `aten::linear`、`aten::tanh`、`aten::relu`、`aten::dropout` 等 ATen op 自然构图。
- `Softmax2d` 的前向数值计算入口是 `aten::_softmax`；反向细节单独放在对应的 `backward_aten` 文档里。
- `SoftMarginLoss` 若改成 `reduction='sum'` 会把末尾 `aten::mean` 换成 `aten::sum`；`reduction='none'` 则没有最终 reduction。
