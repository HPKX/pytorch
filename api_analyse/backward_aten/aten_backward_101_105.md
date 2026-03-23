# torch API 101-105 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | `torch.nn.InstanceNorm1d` | CIA 前向分解，无专属 backward | 无统一 node；主要为 `NativeBatchNormBackward0` / `CudnnBatchNormBackward0` / `MiopenBatchNormBackward0` + view backward | `instance_norm` / `batch_norm` 无条目；内部 backend op 继承 `native_batch_norm: native_batch_norm_backward(...)`、`cudnn_batch_norm: cudnn_batch_norm_backward(...)`、`miopen_batch_norm: miopen_batch_norm_backward(...)` | `aten::native_batch_norm_backward`、`aten::cudnn_batch_norm_backward`、`aten::miopen_batch_norm_backward`、`as_strided_backward` | `instance_norm` 是 CIA；前向先 `contiguous` + `view`，再进 `batch_norm` / `_batch_norm_impl_index`；常规训练分支反向挂在选中的 batch-norm backend op 上，最终 `view` 回原形状时还会走 view backward |
| 102 | `torch.nn.InstanceNorm2d` | CIA 前向分解，无专属 backward | 无统一 node；主要为 `NativeBatchNormBackward0` / `CudnnBatchNormBackward0` / `MiopenBatchNormBackward0` + view backward | 同 101 | `aten::native_batch_norm_backward`、`aten::cudnn_batch_norm_backward`、`aten::miopen_batch_norm_backward`、`as_strided_backward` | 与 `InstanceNorm1d` 的 backward 机制相同，只是输入 rank 为 4D |
| 103 | `torch.nn.InstanceNorm3d` | CIA 前向分解，无专属 backward | 无统一 node；主要为 `NativeBatchNormBackward0` / `CudnnBatchNormBackward0` / `MiopenBatchNormBackward0` + view backward | 同 101 | `aten::native_batch_norm_backward`、`aten::cudnn_batch_norm_backward`、`aten::miopen_batch_norm_backward`、`as_strided_backward` | 与 `InstanceNorm1d` 的 backward 机制相同，只是输入 rank 为 5D |
| 104 | `torch.nn.LPPool1d` | Python 组合前向，无专属 backward | 无统一 node；由 `PowBackward0`、`AvgPool2DBackward0`、`AbsBackward0`、`ReluBackward0`、`MulBackward0/1` 等组合 | 无单独条目；继承 `pow.Tensor_Scalar: pow_backward(...)`、`avg_pool2d: avg_pool2d_backward(...)`、`relu: threshold_backward(...)`、`abs: grad * self.sgn()`、`sign: zeros_like(grad)` | `aten::pow.Tensor_Scalar`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj`、`aten::zeros_like`、`aten::avg_pool2d_backward`、`aten::threshold_backward`、`aten::sgn`、`as_strided_backward` | `avg_pool1d` 本身是 CIA，前向分解为 `unsqueeze -> avg_pool2d -> squeeze`，因此比 2D 版本多一层 view backward；`sign` 分支梯度恒为零 |
| 105 | `torch.nn.LPPool2d` | Python 组合前向，无专属 backward | 无统一 node；由 `PowBackward0`、`AvgPool2DBackward0`、`AbsBackward0`、`ReluBackward0`、`MulBackward0/1` 等组合 | 无单独条目；继承 `pow.Tensor_Scalar: pow_backward(...)`、`avg_pool2d: avg_pool2d_backward(...)`、`relu: threshold_backward(...)`、`abs: grad * self.sgn()`、`sign: zeros_like(grad)` | `aten::pow.Tensor_Scalar`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj`、`aten::zeros_like`、`aten::avg_pool2d_backward`、`aten::threshold_backward`、`aten::sgn` | 与 `LPPool1d` 相同，但池化主链直接是 `avg_pool2d`，无额外 `unsqueeze/squeeze` view backward |

## 详细分析

### 101. torch.nn.InstanceNorm1d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 路径**（常规 `(N, C, L)` 输入）：
`nn.InstanceNorm1d.forward` → `F.instance_norm` → `torch.instance_norm` → `aten::instance_norm` → `at::native::instance_norm` → `aten::contiguous` → `aten::view` → `aten::batch_norm` → `aten::_batch_norm_impl_index` → `aten::native_batch_norm / aten::cudnn_batch_norm / aten::miopen_batch_norm` → `aten::view`

`instance_norm` 本身在 `derivatives.yaml` 中**无条目**，`RegisterCompositeImplicitAutogradEverything.cpp` 里注册为 CIA wrapper：
```cpp
at::Tensor wrapper_CompositeImplicitAutograd__instance_norm(...) {
  return at::native::instance_norm(...);
}
```

`native::instance_norm` 的核心实现（`Normalization.cpp:727`）：
```cpp
auto input_reshaped = input.contiguous().view_symint(shape);
auto out = at::batch_norm(input_reshaped, weight_, bias_, running_mean_, running_var_,
                          use_input_stats, momentum, eps, cudnn_enabled);
return out.view_symint(input.sym_sizes());
```

因此 eager autograd 不会为 `aten::instance_norm` 单独生成 backward node，而是把梯度挂在前向真正执行到的子 op 上：

| 子 op | backward 来源 | 反向依赖 |
| --- | --- | --- |
| `aten::view` | view replay | `as_strided_backward` |
| `aten::batch_norm` | CIA 前向分解到 `_batch_norm_impl_index` | 继续落到具体 backend batch-norm backward |
| `aten::native_batch_norm` | `NativeBatchNormBackward0` | `aten::native_batch_norm_backward` |
| `aten::cudnn_batch_norm` | `CudnnBatchNormBackward0` | `aten::cudnn_batch_norm_backward`；`training=False` 时退回 `aten::native_batch_norm_backward` |
| `aten::miopen_batch_norm` | `MiopenBatchNormBackward0` | `aten::miopen_batch_norm_backward`；`training=False` 时退回 `aten::native_batch_norm_backward` |

**反向 ATen 依赖**：
- `aten::native_batch_norm_backward`
- `aten::cudnn_batch_norm_backward`
- `aten::miopen_batch_norm_backward`
- `as_strided_backward`

**备注**：
- `running_mean` / `running_var` 是 buffer，`native::instance_norm` 末尾的 `alias` / `view` / `mean` / `copy_` 回写统计量路径通常不参与参数梯度图。
- 若模块走“无 batch 输入”分支（`input.unsqueeze(0)` 后再 `squeeze(0)`），还会额外引入一对 view backward，本质仍归到 `as_strided_backward`。

---

### 102. torch.nn.InstanceNorm2d

**反向来源类型**：同 101，`CompositeImplicitAutograd` 前向分解

**Forward 路径**（常规 `(N, C, H, W)` 输入）与 `InstanceNorm1d` 完全同构，只是 reshape 后的空间维更多。autograd 仍然不在 `aten::instance_norm` 本身落 node，而是挂到：
- `aten::native_batch_norm_backward`
- `aten::cudnn_batch_norm_backward`
- `aten::miopen_batch_norm_backward`
- `as_strided_backward`

**反向 ATen 依赖**：同 101。

---

### 103. torch.nn.InstanceNorm3d

**反向来源类型**：同 101，`CompositeImplicitAutograd` 前向分解

**Forward 路径**（常规 `(N, C, D, H, W)` 输入）与 `InstanceNorm1d/2d` 同构：先把 `(N, C, ...)` reshape 成 `(1, N*C, ...)` 送入 `batch_norm`，再 view 回原形状。对应 backward 仍由内部 batch-norm backend op 与 view backward 负责。

**反向 ATen 依赖**：
- `aten::native_batch_norm_backward`
- `aten::cudnn_batch_norm_backward`
- `aten::miopen_batch_norm_backward`
- `as_strided_backward`

---

### 104. torch.nn.LPPool1d

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 路径**（`functional.py:1161`）：
```python
out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1.0 / norm_type)
```

其中 `avg_pool1d` 本身又是 CIA 分解（`Pooling.cpp:114`）：
```cpp
auto output = at::avg_pool2d(
    self.unsqueeze(-2),
    {1, kernel_size[0]},
    {1, stride[0]},
    {0, padding[0]},
    ceil_mode,
    count_include_pad);
return output.squeeze(-2);
```

因此 `LPPool1d` 没有统一的 backward node，而是由各内层 op 独立记录：

| 子 op | backward 公式 / 来源 | 反向依赖 |
| --- | --- | --- |
| `aten::pow.Tensor_Scalar` | `pow_backward(grad, self, exponent)` | `aten::pow.Tensor_Scalar`（递归 `self.pow(exp-1)`）、`aten::mul.Scalar`、`aten::mul.Tensor`、`aten::conj`；`exponent==0` 时为 `aten::zeros_like` |
| `aten::avg_pool1d` | CIA 分解到 `unsqueeze -> avg_pool2d -> squeeze` | `aten::avg_pool2d_backward` + `as_strided_backward` |
| `aten::sign` | `zeros_like(grad)` | `aten::zeros_like` |
| `aten::abs` | `grad * self.sgn()` | `aten::sgn`、`aten::mul.Tensor` |
| `aten::relu` | `threshold_backward(grad, result, 0)` | `aten::threshold_backward` |
| `aten::mul.Tensor` / `aten::mul.Scalar` | 标准乘法 backward | `aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj`（Tensor 分支） |

**反向 ATen 依赖**：
- `aten::pow.Tensor_Scalar`
- `aten::mul.Tensor`
- `aten::mul.Scalar`
- `aten::conj`
- `aten::zeros_like`
- `aten::avg_pool2d_backward`
- `aten::threshold_backward`
- `aten::sgn`
- `as_strided_backward`

**条件依赖**：
- `pow_backward` 的 `handle_r_to_c` 在复数梯度 + 实数输入场景会额外调用 `aten::real`。
- `sign(out)` 这条支路的梯度恒为零，实际体现为 `aten::zeros_like`。

---

### 105. torch.nn.LPPool2d

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 路径**（`functional.py:1122`）：
```python
out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1.0 / norm_type)
```

与 `LPPool1d` 相比，差异只在池化主链直接使用 `aten::avg_pool2d`，因此没有额外的 `unsqueeze/squeeze` view。

**各子 op 反向**：
- `aten::pow.Tensor_Scalar` → `pow_backward(...)`
- `aten::avg_pool2d` → `aten::avg_pool2d_backward`
- `aten::sign` → `aten::zeros_like`
- `aten::abs` → `aten::sgn` + `aten::mul.Tensor`
- `aten::relu` → `aten::threshold_backward`
- `aten::mul.Tensor` / `aten::mul.Scalar` → 标准乘法 backward

**反向 ATen 依赖**：
- `aten::pow.Tensor_Scalar`
- `aten::mul.Tensor`
- `aten::mul.Scalar`
- `aten::conj`
- `aten::zeros_like`
- `aten::avg_pool2d_backward`
- `aten::threshold_backward`
- `aten::sgn`

**备注**：
- 与 `LPPool1d` 相同，`pow_backward` 在复数梯度 + 实数输入场景会额外走 `aten::real`。
- 由于没有 `avg_pool1d` 的 view 分解，2D 版本不额外依赖 `as_strided_backward`。
