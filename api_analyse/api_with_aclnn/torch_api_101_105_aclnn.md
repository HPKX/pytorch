# torch API 101-105 aclnn 接入分析

## 接口 101：`torch.nn.InstanceNorm1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::instance_norm` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::batch_norm` | 计算接口 | 是 | `aclnnBatchNorm` | 通过 native_batch_norm 路径 |
| `aten::_batch_norm_impl_index` | 计算接口 | 否 | - | 仅 aclops 实现，未注册 opapi |
| `aten::native_batch_norm` | 计算接口 | 是 | `aclnnBatchNorm` | StructKernelNpuOpApi.cpp |
| `aten::cudnn_batch_norm` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_batch_norm` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_batch_norm_backward` | 计算接口 | 是 | `aclnnBatchNormBackward` | BatchNormBackwardKernelNpuOpApi.cpp |
| `aten::cudnn_batch_norm_backward` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_batch_norm_backward` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |
| `as_strided_backward` | view | N/A | - | view backward |

> `aten::instance_norm` 虽标记为否，但它是 CIA 分解函数，前向实际分解为 `view`（view）+ `batch_norm` → `native_batch_norm`（aclnn=是）。`aten::_batch_norm_impl_index` 是内部 dispatch 选择函数，NPU 上实际走 `native_batch_norm` 路径。`aten::cudnn_batch_norm` 和 `aten::miopen_batch_norm` 是 CUDA/MIOpen 专用路径，NPU 不会调度到。

---

## 接口 102：`torch.nn.InstanceNorm2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::instance_norm` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::batch_norm` | 计算接口 | 是 | `aclnnBatchNorm` | 通过 native_batch_norm 路径 |
| `aten::_batch_norm_impl_index` | 计算接口 | 否 | - | 仅 aclops 实现，未注册 opapi |
| `aten::native_batch_norm` | 计算接口 | 是 | `aclnnBatchNorm` | StructKernelNpuOpApi.cpp |
| `aten::cudnn_batch_norm` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_batch_norm` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_batch_norm_backward` | 计算接口 | 是 | `aclnnBatchNormBackward` | BatchNormBackwardKernelNpuOpApi.cpp |
| `aten::cudnn_batch_norm_backward` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_batch_norm_backward` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |
| `as_strided_backward` | view | N/A | - | view backward |

> 与 InstanceNorm1d 机制完全一致，NPU 走 `native_batch_norm` → `aclnnBatchNorm` 路径。

---

## 接口 103：`torch.nn.InstanceNorm3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::instance_norm` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::batch_norm` | 计算接口 | 是 | `aclnnBatchNorm` | 通过 native_batch_norm 路径 |
| `aten::_batch_norm_impl_index` | 计算接口 | 否 | - | 仅 aclops 实现，未注册 opapi |
| `aten::native_batch_norm` | 计算接口 | 是 | `aclnnBatchNorm` | StructKernelNpuOpApi.cpp |
| `aten::cudnn_batch_norm` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_batch_norm` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::native_batch_norm_backward` | 计算接口 | 是 | `aclnnBatchNormBackward` | BatchNormBackwardKernelNpuOpApi.cpp |
| `aten::cudnn_batch_norm_backward` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_batch_norm_backward` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |
| `as_strided_backward` | view | N/A | - | view backward |

> 与 InstanceNorm1d/2d 机制完全一致。

---

## 接口 104：`torch.nn.LPPool1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::avg_pool1d` | 计算接口 | 否 | - | 无 op_plugin 实现，通过 avg_pool2d 路径实现 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::avg_pool2d` | 计算接口 | 是 | `aclnnAvgPool2d` | opapi/AvgPool2dKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口 |
| `aten::sign` | 计算接口 | 是 | `aclnnSign` | opapi/StructKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::avg_pool2d_backward` | 计算接口 | 是 | `aclnnAvgPool2dBackward` | opapi/AvgPool2dBackwardKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `as_strided_backward` | view | N/A | - | view backward |

> `aten::avg_pool1d` 虽标记为否，但它是 CIA 分解函数，实际通过 `unsqueeze` + `avg_pool2d`（aclnn=是）+ `squeeze` 实现。

---

## 接口 105：`torch.nn.LPPool2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::avg_pool2d` | 计算接口 | 是 | `aclnnAvgPool2d` | opapi/AvgPool2dKernelNpuOpApi.cpp |
| `aten::sign` | 计算接口 | 是 | `aclnnSign` | opapi/StructKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::avg_pool2d_backward` | 计算接口 | 是 | `aclnnAvgPool2dBackward` | opapi/AvgPool2dBackwardKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 101 | `torch.nn.InstanceNorm1d` | ✅ | `aclnnBatchNorm` | `aclnnBatchNormBackward` | - |
| 102 | `torch.nn.InstanceNorm2d` | ✅ | `aclnnBatchNorm` | `aclnnBatchNormBackward` | - |
| 103 | `torch.nn.InstanceNorm3d` | ✅ | `aclnnBatchNorm` | `aclnnBatchNormBackward` | - |
| 104 | `torch.nn.LPPool1d` | ✅ | `aclnnPowTensorScalar`, `aclnnAvgPool2d`, `aclnnSign`, `aclnnAbs`, `aclnnRelu`, `aclnnMul`, `aclnnMuls` | `aclnnPowTensorScalar`, `aclnnMul`, `aclnnMuls`, `aclnnInplaceZero`, `aclnnAvgPool2dBackward`, `aclnnThresholdBackward`, `aclnnSign` | - |
| 105 | `torch.nn.LPPool2d` | ✅ | `aclnnPowTensorScalar`, `aclnnAvgPool2d`, `aclnnSign`, `aclnnAbs`, `aclnnRelu`, `aclnnMul`, `aclnnMuls` | `aclnnPowTensorScalar`, `aclnnMul`, `aclnnMuls`, `aclnnInplaceZero`, `aclnnAvgPool2dBackward`, `aclnnThresholdBackward`, `aclnnSign` | - |
