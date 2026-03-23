# torch API 131-135 aclnn 接入分析

## 接口 131：`torch.nn.TransformerDecoderLayer`

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
| `aten::dropout` | 计算接口 | 是 | `aclnnDropoutGenMaskV2` / `aclnnDropoutDoMask` | opapi/DropoutKernelNpuOpApi.cpp |
| `aten::layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | opapi/LayerNormKernelNpuOpApi.cpp |
| `aten::native_layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | LayerNormKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::gelu` | 计算接口 | 是 | `aclnnGelu` / `aclnnGeluV2` | opapi/GeluKernelNpuOpApi.cpp |

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
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口，分解为 mm/bmm 等 |

> **备注**：NPU 上 SDPA 走自有 `npu_fusion_attention` 路径，不进入 CUDA 专用 flash/efficient/cudnn backend。`matmul` 和 `_scaled_dot_product_attention_math` 是 composite 接口，分解到已支持的基础算子。

---

## 接口 132：`torch.nn.TransformerEncoder`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_transformer_encoder_layer_fwd` | 计算接口 | **否** | - | 推理 fastpath，训练时不进入 |
| `aten::scaled_dot_product_attention` | 计算接口 | 是 | `aclnnFlashAttentionScore`（间接） | 通过 npu_fusion_attention 间接调用 |
| `aten::layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | opapi/LayerNormKernelNpuOpApi.cpp |
| `aten::native_layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | LayerNormKernelNpuOpApi.cpp |
| `aten::_nested_tensor_from_mask` | 计算接口 | **否** | - | 推理 fastpath NestedTensor 路径，训练时不进入 |
| `aten::to_padded_tensor` | 计算接口 | **否** | - | 推理 fastpath NestedTensor 路径，训练时不进入 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_layer_norm_backward` | 计算接口 | 是 | `aclnnLayerNormBackward` | LayerNormBackwardKernelNpuOpApi.cpp |
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_dropout_backward` | 计算接口 | 是 | `aclnnDropoutDoMask` | NativeDropoutKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::gelu_backward` | 计算接口 | 是 | `aclnnGeluBackward` | opapi/GeluBackwardKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |

> **备注**：`_transformer_encoder_layer_fwd`、`_nested_tensor_from_mask`、`to_padded_tensor` 仅在推理 fastpath 使用，训练时不走此路径，不影响 A5 支持判定。

---

## 接口 133：`torch.nn.TransformerEncoderLayer`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_transformer_encoder_layer_fwd` | 计算接口 | **否** | - | 推理 fastpath，训练时不进入 |
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm` / `aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::scaled_dot_product_attention` | 计算接口 | 是 | `aclnnFlashAttentionScore`（间接） | 通过 npu_fusion_attention 间接调用 |
| `aten::_scaled_dot_product_flash_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_efficient_attention` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_cudnn_attention` | 计算接口 | **否** | - | CUDA/cuDNN 专用接口，NPU 未实现 |
| `aten::_scaled_dot_product_attention_math` | 计算接口 | **否** | - | PyTorch composite 接口 |
| `aten::dropout` | 计算接口 | 是 | `aclnnDropoutGenMaskV2` / `aclnnDropoutDoMask` | opapi/DropoutKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | opapi/LayerNormKernelNpuOpApi.cpp |
| `aten::native_layer_norm` | 计算接口 | 是 | `aclnnLayerNorm` / `aclnnFastLayerNorm` | LayerNormKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::gelu` | 计算接口 | 是 | `aclnnGelu` / `aclnnGeluV2` | opapi/GeluKernelNpuOpApi.cpp |
| `aten::_native_multi_head_attention` | 计算接口 | **否** | - | 推理 fastpath 内部使用 |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::_addmm_activation` | 计算接口 | **否** | - | 推理 fastpath 内部使用 |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_layer_norm_backward` | 计算接口 | 是 | `aclnnLayerNormBackward` | LayerNormBackwardKernelNpuOpApi.cpp |
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_dropout_backward` | 计算接口 | 是 | `aclnnDropoutDoMask` | NativeDropoutKernelNpuOpApi.cpp |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::gelu_backward` | 计算接口 | 是 | `aclnnGeluBackward` | opapi/GeluBackwardKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |

> **备注**：推理 fastpath 的 `_transformer_encoder_layer_fwd`、`_native_multi_head_attention`、`_addmm_activation` 仅在 `eval/no_grad` 时进入，训练态不走此路径。NPU 上 SDPA 走自有路径。

---

## 接口 134：`torch.nn.TripletMarginLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::triplet_margin_loss` | 计算接口 | **否** | - | CIA composite 接口，分解为 pairwise_distance + clamp_min + reduction |
| `aten::pairwise_distance` | 计算接口 | **否** | - | CIA composite 接口，分解为 sub + norm 等 |
| `aten::norm` | 计算接口 | 是 | `aclnnNorm` | NormKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | opapi/StructKernelNpuOpApi.cpp |
| `aten::minimum` | 计算接口 | 是 | `aclnnMinimum` | MinKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pairwise_distance` | 计算接口 | **否** | - | CIA composite 接口，分解为 sub + norm 等 |
| `aten::norm` | 计算接口 | 是 | `aclnnNorm` | NormKernelNpuOpApi.cpp |
| `aten::minimum` | 计算接口 | 是 | `aclnnMinimum` | MinKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> **备注**：`aten::triplet_margin_loss` 和 `aten::pairwise_distance` 均为 CIA composite 接口，实际计算由 `norm`、`sub`、`clamp_min` 等已接入 aclnn 的算子承担，因此不影响 A5 支持判定。

---

## 接口 135：`torch.nn.functional.affine_grid`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::affine_grid_generator` | 计算接口 | 是 | `aclnnAffineGrid` | opapi/AffineGridGeneratorKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::linspace` | 计算接口 | 是 | `aclnnLinspace` | LinspaceKernelNpuOpApi.cpp |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |
| `aten::fill_` | 计算接口 | 是 | `aclnnInplaceFillScalar` / `aclnnInplaceFillTensor` | opapi/FillKernelNpuOpApi.cpp |
| `aten::transpose` | view | N/A | - | view 类接口 |
| `aten::bmm` | 计算接口 | 是 | `aclnnBatchMatMul` | opapi/BmmKernelNpuOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::affine_grid_generator_backward` | 计算接口 | 是 | `aclnnBatchMatMul` | opapi/AffineGridGeneratorBackwardKernelNpuOpApi.cpp（通过 BatchMatMul 实现） |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 131 | `torch.nn.TransformerDecoderLayer` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 132 | `torch.nn.TransformerEncoder` | ✅ | `aclnnFlashAttentionScore`、`aclnnLayerNorm` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 133 | `torch.nn.TransformerEncoderLayer` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 134 | `torch.nn.TripletMarginLoss` | ✅ | `aclnnNorm`、`aclnnClampMin`、`aclnnMinimum`、`aclnnMean`、`aclnnReduceSum` | `aclnnNorm`、`aclnnMinimum`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | - |
| 135 | `torch.nn.functional.affine_grid` | ✅ | `aclnnAffineGrid` | `aclnnBatchMatMul` | - |
