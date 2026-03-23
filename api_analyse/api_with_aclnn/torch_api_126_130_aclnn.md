# torch API 126-130 aclnn 接入分析

## 接口 126：`torch.nn.Softmin`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 127：`torch.nn.Softsign`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 128：`torch.nn.Tanhshrink`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tanh` | 计算接口 | 是 | `aclnnTanh` | opapi/TanhKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tanh_backward` | 计算接口 | 是 | `aclnnTanhBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 129：`torch.nn.Transformer`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm` / `aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::scaled_dot_product_attention` | 计算接口 | 是 | `aclnnFlashAttentionScore`（间接） | 通过 npu_fusion_attention 间接调用 |
| `aten::_scaled_dot_product_flash_attention_for_cpu` | 计算接口 | **否** | - | CPU 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_flash_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_attention_math` | 计算接口 | **否** | - | PyTorch composite 接口 |
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口，分解为 mm/bmm 等 |
| `aten::_safe_softmax` | 计算接口 | **否** | - | fallback |
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | opapi/StructKernelNpuOpApi.cpp |
| `aten::dropout` | 计算接口 | 是 | `aclnnDropoutGenMaskV2` / `aclnnDropoutDoMask` | opapi/DropoutKernelNpuOpApi.cpp |
| `aten::layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | opapi/LayerNormKernelNpuOpApi.cpp |
| `aten::native_layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | LayerNormKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::gelu` | 计算接口 | 是 | `aclnnGelu` / `aclnnGeluV2` | opapi/GeluKernelNpuOpApi.cpp |
| `aten::_transformer_encoder_layer_fwd` | 计算接口 | **否** | - | 推理 fastpath，训练时不进入 |
| `aten::_native_multi_head_attention` | 计算接口 | **否** | - | 推理 fastpath，训练时不进入 |
| `aten::_addmm_activation` | 计算接口 | **否** | - | 推理 fastpath 内部使用 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_layer_norm_backward` | 计算接口 | 是 | `aclnnLayerNormBackward` | LayerNormBackwardKernelNpuOpApi.cpp |
| `aten::_scaled_dot_product_flash_attention_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention_backward` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_dropout_backward` | 计算接口 | 是 | `aclnnDropoutDoMask` | NativeDropoutKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::gelu_backward` | 计算接口 | 是 | `aclnnGeluBackward` | opapi/GeluBackwardKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |

> **备注**：NPU 上 `scaled_dot_product_attention` 通过 `npu_fusion_attention` 走自有 aclnn FlashAttention 接口，不会进入 CUDA 专用的 flash/efficient/cudnn SDPA backend。`_transformer_encoder_layer_fwd` 和 `_native_multi_head_attention` 仅在推理 fastpath 使用，训练时不走此路径。`matmul`、`_safe_softmax`、`_scaled_dot_product_attention_math` 是 composite 接口，分解到已支持的基础算子。

---

## 接口 130：`torch.nn.TransformerDecoder`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm` / `aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::scaled_dot_product_attention` | 计算接口 | 是 | `aclnnFlashAttentionScore`（间接） | 通过 npu_fusion_attention 间接调用 |
| `aten::_scaled_dot_product_flash_attention_for_cpu` | 计算接口 | **否** | - | CPU 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_flash_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_attention_math` | 计算接口 | **否** | - | PyTorch composite 接口 |
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口，分解为 mm/bmm 等 |
| `aten::_safe_softmax` | 计算接口 | **否** | - | fallback |
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | opapi/StructKernelNpuOpApi.cpp |
| `aten::dropout` | 计算接口 | 是 | `aclnnDropoutGenMaskV2` / `aclnnDropoutDoMask` | opapi/DropoutKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::gelu` | 计算接口 | 是 | `aclnnGelu` / `aclnnGeluV2` | opapi/GeluKernelNpuOpApi.cpp |
| `aten::layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | opapi/LayerNormKernelNpuOpApi.cpp |
| `aten::native_layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | LayerNormKernelNpuOpApi.cpp |
| `aten::_native_multi_head_attention` | 计算接口 | **否** | - | 推理 fastpath，训练时不进入 |
| `aten::bmm` | 计算接口 | 是 | `aclnnBatchMatMul` | opapi/BmmKernelNpuOpApi.cpp |
| `aten::_masked_softmax` | 计算接口 | 是 | `aclnnScaledMaskedSoftmax` | opapi/MaskedSoftMaxKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_layer_norm_backward` | 计算接口 | 是 | `aclnnLayerNormBackward` | LayerNormBackwardKernelNpuOpApi.cpp |
| `aten::_scaled_dot_product_flash_attention_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention_backward` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_dropout_backward` | 计算接口 | 是 | `aclnnDropoutDoMask` | NativeDropoutKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::gelu_backward` | 计算接口 | 是 | `aclnnGeluBackward` | opapi/GeluBackwardKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |

> **备注**：与 Transformer 相同，NPU 上 SDPA 走自有路径，不进入 CUDA 专用 backend。推理 fastpath 不影响训练态 backward。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 126 | `torch.nn.Softmin` | ✅ | `aclnnNeg`、`aclnnSoftmax` | `aclnnSoftmaxBackward`、`aclnnNeg` | - |
| 127 | `torch.nn.Softsign` | ✅ | `aclnnAbs`、`aclnnAdds`、`aclnnDiv` | `aclnnDiv`、`aclnnAbs`、`aclnnSign`、`aclnnAdds`、`aclnnMul`、`aclnnNeg` | - |
| 128 | `torch.nn.Tanhshrink` | ✅ | `aclnnTanh`、`aclnnSub` | `aclnnTanhBackward`、`aclnnNeg` | - |
| 129 | `torch.nn.Transformer` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnSoftmax`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 130 | `torch.nn.TransformerDecoder` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnSoftmax`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
