# torch API 116-120 aclnn 接入分析

## 接口 116：`torch.nn.MultiLabelSoftMarginLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::log_sigmoid` | 计算接口 | 是 | `aclnnLogSigmoidForward` | opapi/LogSigmoidNpuOpApi.cpp，通过 log_sigmoid_forward 实现 |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::log_sigmoid_forward` | 计算接口 | 是 | `aclnnLogSigmoidForward` | opapi/LogSigmoidNpuOpApi.cpp |
| `aten::rsub.Scalar` | 计算接口 | 是 | `aclnnRsubs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` / `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` / `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::log_sigmoid_backward` | 计算接口 | 是 | `aclnnLogSigmoidBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::rsub.Scalar` | 计算接口 | 是 | `aclnnRsubs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sum.dim_IntList` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

---

## 接口 117：`torch.nn.MultiMarginLoss`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multi_margin_loss` | 计算接口 | **否** | - | op_plugin 中无实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multi_margin_loss_backward` | 计算接口 | **否** | - | op_plugin 中无实现 |

---

## 接口 118：`torch.nn.MultiheadAttention`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm` / `aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::scaled_dot_product_attention` | 计算接口 | 是 | `aclnnFlashAttentionScore`（间接） | 通过 npu_fusion_attention 间接调用 |
| `aten::_scaled_dot_product_flash_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_attention_math` | 计算接口 | **否** | - | PyTorch composite 接口 |
| `aten::_native_multi_head_attention` | 计算接口 | **否** | - | 推理 fastpath，训练时不进入 |
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口，分解为 mm/bmm/dot 等 |
| `aten::bmm` | 计算接口 | 是 | `aclnnBatchMatMul` | opapi/BmmKernelNpuOpApi.cpp |
| `aten::_masked_softmax` | 计算接口 | 是 | `aclnnScaledMaskedSoftmax` | opapi/MaskedSoftMaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | opapi/StructKernelNpuOpApi.cpp |
| `aten::dropout` | 计算接口 | 是 | `aclnnDropoutGenMaskV2` / `aclnnDropoutDoMask` | opapi/DropoutKernelNpuOpApi.cpp |
| `aten::_safe_softmax` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现，fallback |
| `aten::baddbmm` | 计算接口 | 是 | `aclnnBaddbmm` | opapi/BaddbmmKernelNpuOpApi.cpp |
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_scaled_dot_product_flash_attention_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention_backward` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_dropout_backward` | 计算接口 | 是 | `aclnnDropoutDoMask` | NativeDropoutKernelNpuOpApi.cpp |
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口，分解为 mm/bmm 等 |
| `aten::bmm` | 计算接口 | 是 | `aclnnBatchMatMul` | opapi/BmmKernelNpuOpApi.cpp |
| `aten::baddbmm` | 计算接口 | 是 | `aclnnBaddbmm` | opapi/BaddbmmKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |

> **备注**：NPU 上 `scaled_dot_product_attention` 通过 `npu_fusion_attention` 走自有 aclnn FlashAttention 接口，不会进入 CUDA 专用的 flash/efficient/cudnn SDPA backend。`_safe_softmax` 和 `matmul` 是 composite 接口，分解到已支持的基础算子。`_native_multi_head_attention` 仅在推理 fastpath 使用，训练时不走此路径。因此整体可在 A5 上支持。

---

## 接口 119：`torch.nn.PixelUnshuffle`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pixel_unshuffle` | 计算接口 | **否** | - | CEA math fallback 可用，分解为 reshape+permute+clone+view |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::clone` | 计算接口 | 是 | `aclnnInplaceCopy` | CloneKernelOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pixel_shuffle` | 计算接口 | **否** | - | CEA math fallback 可用，分解为 reshape+permute+contiguous+view |

> **备注**：`pixel_unshuffle` 和 `pixel_shuffle` 虽然标记为计算接口=否，但它们都有 CompositeExplicitAutograd math fallback（`math_pixel_unshuffle`、`math_pixel_shuffle`），分解为纯 view 操作（reshape+permute+clone/contiguous+view），所有子算子在 NPU 上均可用，因此不影响 A5 支持判定。

---

## 接口 120：`torch.nn.PoissonNLLLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::poisson_nll_loss` | 计算接口 | **否** | - | PyTorch CIA composite 接口，无 NPU 注册 |
| `aten::exp` | 计算接口 | 是 | `aclnnExp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::le` | 计算接口 | 是 | `aclnnLeTensor` | opapi/LeKernelNpuOpApi.cpp |
| `aten::masked_fill` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` / `aclnnInplaceAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::exp` | 计算接口 | 是 | `aclnnExp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::le` | 计算接口 | 是 | `aclnnLeTensor` | opapi/LeKernelNpuOpApi.cpp |
| `aten::masked_fill` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> **备注**：`aten::poisson_nll_loss` 标记为计算接口=否，但它是 CIA composite 函数，实际计算由 `exp`、`mul`、`sub`、`log`、`le`、`masked_fill`、`mean` 等已接入 aclnn 的算子承担，因此不影响 A5 支持判定。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 116 | `torch.nn.MultiLabelSoftMarginLoss` | ✅ | `aclnnLogSigmoidForward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | `aclnnLogSigmoidBackward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | - |
| 117 | `torch.nn.MultiMarginLoss` | ❌ | - | - | **`aten::multi_margin_loss`**、**`aten::multi_margin_loss_backward`** |
| 118 | `torch.nn.MultiheadAttention` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnBatchMatMul`、`aclnnSoftmax`、`aclnnDropoutGenMaskV2`、`aclnnBaddbmm` | `aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnBatchMatMul`、`aclnnAddmm`、`aclnnBaddbmm` | - |
| 119 | `torch.nn.PixelUnshuffle` | ✅ | `aclnnInplaceCopy`（clone） | `aclnnInplaceCopy`（clone） | - |
| 120 | `torch.nn.PoissonNLLLoss` | ✅ | `aclnnExp`、`aclnnMul`、`aclnnSub`、`aclnnAdds`、`aclnnLog`、`aclnnLeTensor`、`aclnnInplaceMaskedFillScalar`、`aclnnMean` | `aclnnExp`、`aclnnMul`、`aclnnSub`、`aclnnAdds`、`aclnnLog`、`aclnnLeTensor`、`aclnnMean`、`aclnnReduceSum` | - |
