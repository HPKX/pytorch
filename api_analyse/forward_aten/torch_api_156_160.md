# torch API 156-160 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.nn.functional.pixel_unshuffle` | CPU 输入 | `F.pixel_unshuffle (= torch.pixel_unshuffle) -> aten::pixel_unshuffle -> AutogradCPU（若求导）-> CPU -> at::native::pixel_unshuffle_cpu -> pixel_unshuffle_kernel stub -> cpu_pixel_unshuffle / cpu_pixel_unshuffle_channels_last` | `aten::pixel_unshuffle` |
| `torch.nn.functional.pixel_unshuffle` | CUDA 等无专用 kernel 的后端 | `F.pixel_unshuffle -> torch.pixel_unshuffle -> aten::pixel_unshuffle -> AutogradCUDA（若求导）-> CompositeExplicitAutogradNonFunctional -> at::native::math_pixel_unshuffle -> aten::reshape -> aten::permute -> aten::clone -> aten::view` | `aten::pixel_unshuffle`, `aten::reshape`, `aten::permute`, `aten::clone`, `aten::view` |
| `torch.nn.functional.rms_norm` | CUDA contiguous dense 输入 | `F.rms_norm -> torch.rms_norm -> aten::rms_norm -> CompositeImplicitAutograd -> at::native::rms_norm_symint -> aten::_fused_rms_norm -> AutogradCUDA（若求导）-> CUDA -> at::native::_fused_rms_norm_cuda` | `aten::rms_norm`, `aten::_fused_rms_norm` |
| `torch.nn.functional.rms_norm` | CPU 普通 dense 输入 | `F.rms_norm -> torch.rms_norm -> aten::rms_norm -> CompositeImplicitAutograd -> at::native::rms_norm_symint -> aten::_fused_rms_norm -> AutogradCPU（若求导）-> CompositeImplicitAutograd -> at::native::rms_norm_composite -> aten::pow.Tensor_Scalar -> aten::mean.dim -> aten::add_ -> aten::rsqrt -> aten::mul.Tensor` | `aten::rms_norm`, `aten::_fused_rms_norm`, `aten::pow.Tensor_Scalar`, `aten::mean.dim`, `aten::add_`, `aten::rsqrt`, `aten::mul.Tensor` |
| `torch.nn.functional.rrelu` | 训练态，`inplace=False` | `F.rrelu -> torch.rrelu -> aten::rrelu -> CompositeImplicitAutograd -> at::native::rrelu -> aten::rrelu_with_noise -> AutogradCPU/CUDA -> CPU/CUDA -> at::native::rrelu_with_noise_{cpu/cuda}` | `aten::rrelu`, `aten::rrelu_with_noise` |
| `torch.nn.functional.rrelu` | 推理态，`training=False` | `F.rrelu -> torch.rrelu -> aten::rrelu -> CompositeImplicitAutograd -> at::native::rrelu -> aten::rrelu_with_noise -> CPU/CUDA native eval 分支 -> aten::leaky_relu.out` | `aten::rrelu`, `aten::rrelu_with_noise`, `aten::leaky_relu.out` |
| `torch.nn.functional.soft_margin_loss` | `reduction='mean'` | `F.soft_margin_loss -> torch._C._nn.soft_margin_loss -> aten::soft_margin_loss -> AutogradCPU/CUDA -> CompositeExplicitAutograd -> at::native::soft_margin_loss -> aten::neg.out -> aten::mul_ -> aten::exp_ -> aten::log1p_ -> aten::mean` | `aten::soft_margin_loss`, `aten::neg.out`, `aten::mul_`, `aten::exp_`, `aten::log1p_`, `aten::mean` |
| `torch.nn.functional.softmin` | 显式指定 `dim`，`dtype=None` | `F.softmin -> (-input).softmax(dim) -> aten::neg -> aten::softmax.int -> CompositeImplicitAutograd -> at::native::softmax -> aten::_softmax -> AutogradCPU/CUDA（若求导）-> CPU/CUDA structured kernel(meta + impl)` | `aten::neg`, `aten::softmax.int`, `aten::_softmax` |

## 备注

- `softmin` 没有独立的 `aten::softmin`，真实链路从 `aten::neg` 开始。
- `rms_norm` 的公共入口 `aten::rms_norm` 本身没有单独 autograd wrapper；梯度记录发生在内层 `aten::_fused_rms_norm` 或 fallback 分解出的基础 ATen 算子上。
- `rms_norm` 的 `channels_last`、complex、`weight.dtype != input.dtype`，以及带梯度的 MPS 情况，会回退到 `at::native::rms_norm_composite` 路线。
