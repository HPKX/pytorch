# torch API 86-90 aclnn 接入分析

## 接口 86：`torch.nn.AvgPool1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::avg_pool1d` | 计算接口 | **否** | - | 无 op_plugin 实现，CIA 分解走 avg_pool2d 路径 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::avg_pool2d` | 计算接口 | 是 | `aclnnAvgPool2d` | opapi/AvgPool2dKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::avg_pool2d_backward` | 计算接口 | 是 | `aclnnAvgPool2dBackward` | opapi/AvgPool2dBackwardKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |

> **备注**：`aten::avg_pool1d` 标记为计算接口=否，但它是 CIA composite 函数，实际计算由 `aten::avg_pool2d`（已接入 aclnn）承担，因此不影响 A5 支持判定。

---

## 接口 87：`torch.nn.CELU`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::celu` | 计算接口 | 是 | `aclnnCelu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::elu` | 计算接口 | 是 | `aclnnElu` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::elu_backward` | 计算接口 | 是 | `aclnnEluBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 接口 88：`torch.nn.CTCLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ctc_loss.IntList` | 计算接口 | 是 | `aclnnCtcLoss` | opapi/CtcLossKernelNpuOpApi.cpp |
| `aten::ctc_loss.Tensor` | 计算接口 | 是 | `aclnnCtcLoss` | opapi/CtcLossKernelNpuOpApi.cpp |
| `aten::_ctc_loss` | 计算接口 | 是 | `aclnnCtcLoss` | opapi/CtcLossKernelNpuOpApi.cpp |
| `aten::_use_cudnn_ctc_loss` | 计算接口 | **否** | - | CUDA 专用接口，NPU 上返回 false，不进入此路径 |
| `aten::_use_miopen_ctc_loss` | 计算接口 | **否** | - | CUDA/MIOpen 专用接口，NPU 上返回 false，不进入此路径 |
| `aten::_cudnn_ctc_loss` | 计算接口 | **否** | - | CUDA 专用接口，NPU 上不触发 |
| `aten::miopen_ctc_loss` | 计算接口 | **否** | - | MIOpen 专用接口，NPU 上不触发 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_ctc_loss_backward` | 计算接口 | 是 | `aclnnCtcLossBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::_cudnn_ctc_loss_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 上不触发 |
| `aten::_miopen_ctc_loss_backward` | 计算接口 | **否** | - | MIOpen 专用接口，NPU 上不触发 |

> **备注**：cuDNN/MIOpen 路径仅在 CUDA/ROCm 设备上触发，NPU 设备走 `_ctc_loss` + `_ctc_loss_backward` 主路径，均已接入 aclnn，不影响 A5 支持。

---

## 接口 89：`torch.nn.ChannelShuffle`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | StructKernelNpuOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::reshape` | view | N/A | - | composite_view |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | self-inverse，反向再次调用自身 |
| `aten::native_channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | StructKernelNpuOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::reshape` | view | N/A | - | composite_view |

---

## 接口 90：`torch.nn.ConvTranspose1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::conv_transpose1d` | 计算接口 | 是 | `aclnnConvolution` | 通过 convolution 路径（transposed=true） |
| `aten::convolution` | 计算接口 | 是 | `aclnnConvolution` | opapi/ConvolutionKernelNpuOpApi.cpp |
| `aten::_convolution` | 计算接口 | 是 | `aclnnConvolution` | opapi/ConvolutionKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::convolution_backward` | 计算接口 | 是 | `aclnnConvolutionBackward` | opapi/ConvolutionBackwardKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 86 | `torch.nn.AvgPool1d` | ✅ | `aclnnAvgPool2d` | `aclnnAvgPool2dBackward` | - |
| 87 | `torch.nn.CELU` | ✅ | `aclnnCelu`、`aclnnElu` | `aclnnEluBackward` | - |
| 88 | `torch.nn.CTCLoss` | ✅ | `aclnnCtcLoss` | `aclnnCtcLossBackward` | - |
| 89 | `torch.nn.ChannelShuffle` | ✅ | `aclnnChannelShuffle` | `aclnnChannelShuffle` | - |
| 90 | `torch.nn.ConvTranspose1d` | ✅ | `aclnnConvolution` | `aclnnConvolutionBackward` | - |
