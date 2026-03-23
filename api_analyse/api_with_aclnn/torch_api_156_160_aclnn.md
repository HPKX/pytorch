# torch API 156-160 aclnn 接入分析

## 接口 156：`torch.nn.functional.pixel_unshuffle`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pixel_unshuffle` | 计算接口 | 否 | - | op_plugin 中无实现 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::clone` | 计算接口 | 是 | `aclnnInplaceCopy` | CloneKernelOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pixel_shuffle` | 计算接口 | 否 | - | op_plugin 中无实现 |

> **注意**：`aten::pixel_unshuffle` 在无专用 kernel 的后端（如 CUDA/NPU）会通过 `CompositeExplicitAutogradNonFunctional` 分解为 `reshape → permute → clone → view`，这些子算子均为 view 或已接入 aclnn。但其 backward 直接调用 `aten::pixel_shuffle`，后者在 NPU 上同样无实现。正反向均缺少独立 aclnn 实现，但前向可通过 CEA 分解执行。由于 backward 的 `pixel_shuffle` 同样会走 CEA 分解（`reshape → permute → clone → view`），实际均可通过 view 操作完成。

---

## 接口 157：`torch.nn.functional.rms_norm`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rms_norm` | 计算接口 | 是 | `aclnnRmsNorm` | RmsNormKernelOpApi.cpp，NPU 有自定义 npu_rms_norm |
| `aten::_fused_rms_norm` | 计算接口 | 否 | - | NPU 上走 rms_norm 自定义算子路径，不走此标准 ATen 接口 |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::mean.dim` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::rsqrt` | 计算接口 | 是 | `aclnnRsqrt` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_fused_rms_norm_backward` | 计算接口 | 否 | - | NPU 上走 npu_rms_norm 自定义反向路径 |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::mean.dim` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::rsqrt` | 计算接口 | 是 | `aclnnRsqrt` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

> **注意**：NPU 上 `rms_norm` 通过自定义 `npu_rms_norm` 算子（`aclnnRmsNorm`）接入，不走标准 `_fused_rms_norm` / `_fused_rms_norm_backward` 路径。fallback 路径分解为 `pow → mean.dim → add_ → rsqrt → mul.Tensor`，这些子算子均已接入 aclnn。因此 `_fused_rms_norm` 和 `_fused_rms_norm_backward` 标记为否不影响 A5 支持。

---

## 接口 158：`torch.nn.functional.rrelu`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rrelu` | 计算接口 | 否 | - | CIA 包装层，委托给 rrelu_with_noise |
| `aten::rrelu_with_noise` | 计算接口 | 是 | `aclnnRReluWithNoise` | opapi/RReluWithNoiseKernelNpuOpApi.cpp |
| `aten::leaky_relu.out` | 计算接口 | 是 | `aclnnLeakyRelu` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rrelu_with_noise_backward` | 计算接口 | 否 | - | op_plugin 中未找到实现 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::leaky_relu_backward` | 计算接口 | 是 | `aclnnLeakyReluBackward` | StructKernelNpuOpApi.cpp |

> **注意**：`aten::rrelu_with_noise_backward` 标记为否，但该 helper 在训练态展开为 `noise * grad_output`（即 `aten::mul.Tensor`），推理态退化为 `aten::leaky_relu_backward`，展开后的子算子均已接入 aclnn。`aten::rrelu` 是 CIA 壳，分解后委托给 `rrelu_with_noise`。

---

## 接口 159：`torch.nn.functional.soft_margin_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::soft_margin_loss` | 计算接口 | 是 | `aclnnSoftMarginLoss` | StructKernelNpuOpApi.cpp |
| `aten::neg.out` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::exp_` | 计算接口 | 是 | `aclnnInplaceExp` | StructKernelNpuOpApi.cpp |
| `aten::log1p_` | 计算接口 | 是 | `aclnnInplaceLog1p` | StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::soft_margin_loss_backward` | 计算接口 | 是 | `aclnnSoftMarginLossBackward` | StructKernelNpuOpApi.cpp |

---

## 接口 160：`torch.nn.functional.softmin`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | StructKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 156 | `torch.nn.functional.pixel_unshuffle` | ✅ | `aclnnInplaceCopy` | - | `aten::pixel_unshuffle`（正向）、`aten::pixel_shuffle`（反向），均可通过 CEA 分解为 view + clone 执行 |
| 157 | `torch.nn.functional.rms_norm` | ✅ | `aclnnRmsNorm`、`aclnnPowTensorScalar`、`aclnnMean`、`aclnnInplaceAdd`、`aclnnRsqrt`、`aclnnMul` | `aclnnPowTensorScalar`、`aclnnMean`、`aclnnInplaceAdd`、`aclnnRsqrt`、`aclnnMul` | `aten::_fused_rms_norm`、`aten::_fused_rms_norm_backward`（NPU 走自定义 npu_rms_norm 路径，不影响支持） |
| 158 | `torch.nn.functional.rrelu` | ✅ | `aclnnRReluWithNoise`、`aclnnLeakyRelu` | `aclnnMul`、`aclnnLeakyReluBackward` | `aten::rrelu`（正向，CIA 壳）、`aten::rrelu_with_noise_backward`（反向，helper 展开后子算子均可支持） |
| 159 | `torch.nn.functional.soft_margin_loss` | ✅ | `aclnnSoftMarginLoss`、`aclnnNeg`、`aclnnInplaceMul`、`aclnnInplaceExp`、`aclnnInplaceLog1p`、`aclnnMean` | `aclnnSoftMarginLossBackward` | - |
| 160 | `torch.nn.functional.softmin` | ✅ | `aclnnNeg`、`aclnnSoftmax` | `aclnnNeg`、`aclnnSoftmaxBackward` | - |
