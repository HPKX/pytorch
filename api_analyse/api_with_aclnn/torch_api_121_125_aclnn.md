# torch API 121-125 aclnn 接入分析

## 接口 121：`torch.nn.RNN`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rnn_tanh.input` | 计算接口 | **否** | - | CIA composite，分解为 linear + add_ + tanh |
| `aten::rnn_relu.input` | 计算接口 | **否** | - | CIA composite，分解为 linear + add_ + relu |
| `aten::rnn_tanh.data` | 计算接口 | **否** | - | CIA composite（PackedSequence 变体） |
| `aten::rnn_relu.data` | 计算接口 | **否** | - | CIA composite（PackedSequence 变体） |
| `aten::_cudnn_rnn` | 计算接口 | **否** | - | CUDA 专用接口，NPU 上不触发 |
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm` / `aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::tanh` | 计算接口 | 是 | `aclnnTanh` | opapi/TanhKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::dropout` | 计算接口 | 是 | `aclnnDropoutGenMaskV2` / `aclnnDropoutDoMask` | opapi/DropoutKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_cudnn_rnn_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 上不触发 |
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::t` | view | N/A | - | view 类接口 |
| `aten::tanh_backward` | 计算接口 | 是 | `aclnnTanhBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::native_dropout_backward` | 计算接口 | 是 | `aclnnDropoutDoMask` | NativeDropoutKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

> **备注**：`aten::rnn_tanh.*` 和 `aten::rnn_relu.*` 外层接口均为 CIA composite 函数，NPU 上 cuDNN 条件不满足，自动退回 native 路径（`linear + add_ + tanh/relu`），所有 native 路径子算子均已接入 aclnn。cuDNN 路径仅在 CUDA 设备上触发，不影响 A5 支持判定。

---

## 接口 122：`torch.nn.RNNCell`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rnn_tanh_cell` | 计算接口 | **否** | - | op_plugin 中无实现 |
| `aten::rnn_relu_cell` | 计算接口 | **否** | - | op_plugin 中无实现 |
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm` / `aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::tanh` | 计算接口 | 是 | `aclnnTanh` | opapi/TanhKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::addmm` | 计算接口 | 是 | `aclnnAddmm` | opapi/AddmmKernelNpuOpApi.cpp |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::t` | view | N/A | - | view 类接口 |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::tanh_backward` | 计算接口 | 是 | `aclnnTanhBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | opapi/StructKernelNpuOpApi.cpp |

> **备注**：`aten::rnn_tanh_cell` 和 `aten::rnn_relu_cell` 标记为计算接口=否，但它们是 CIA composite 函数（在 `RNN.cpp` 中展开为 `linear + linear + add_ + tanh/relu`），实际计算由已接入 aclnn 的子算子承担，因此不影响 A5 支持判定。

---

## 接口 123：`torch.nn.RReLU`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rrelu` | 计算接口 | 否 | - | CIA 壳，委托给 rrelu_with_noise |
| `aten::rrelu_with_noise` | 计算接口 | 是 | `aclnnRReluWithNoise` | opapi/RReluWithNoiseKernelNpuOpApi.cpp |
| `aten::leaky_relu.out` | 计算接口 | 是 | `aclnnLeakyRelu` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rrelu_with_noise_backward` | 计算接口 | 否 | - | CEA composite，训练态分解为 noise*grad（mul.Tensor），推理态为 leaky_relu_backward |
| `aten::leaky_relu_backward` | 计算接口 | 是 | `aclnnLeakyReluBackward` | opapi/StructKernelNpuOpApi.cpp |

> **备注**：`aten::rrelu_with_noise_backward` 是 CEA composite，训练态展开为 `noise * grad_output`（即 `mul.Tensor`，aclnn=是），推理态退化为 `leaky_relu_backward`（aclnn=是）。与 API 158（`torch.nn.functional.rrelu`）结论一致。

---

## 接口 124：`torch.nn.SoftMarginLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::soft_margin_loss` | 计算接口 | 是 | `aclnnSoftMarginLoss` | opapi/StructKernelNpuOpApi.cpp |
| `aten::neg.out` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` / `aclnnInplaceMuls` | MulKernelNpuOpApi.cpp |
| `aten::exp_` | 计算接口 | 是 | `aclnnInplaceExp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::log1p_` | 计算接口 | 是 | `aclnnInplaceLog1p` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::soft_margin_loss_backward` | 计算接口 | 是 | `aclnnSoftMarginLossBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 接口 125：`torch.nn.Softmax2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 121 | `torch.nn.RNN` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnInplaceAdd`、`aclnnTanh`、`aclnnRelu`、`aclnnDropoutGenMaskV2`、`aclnnCat` | `aclnnAddmm`、`aclnnMm`、`aclnnTanhBackward`、`aclnnThresholdBackward`、`aclnnDropoutDoMask`、`aclnnCat` | - |
| 122 | `torch.nn.RNNCell` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnInplaceAdd`、`aclnnTanh`、`aclnnRelu` | `aclnnAddmm`、`aclnnMm`、`aclnnAdd`、`aclnnTanhBackward`、`aclnnThresholdBackward` | - |
| 123 | `torch.nn.RReLU` | ✅ | `aclnnRReluWithNoise` | `aclnnMul`、`aclnnLeakyReluBackward` | - |
| 124 | `torch.nn.SoftMarginLoss` | ✅ | `aclnnSoftMarginLoss` | `aclnnSoftMarginLossBackward` | - |
| 125 | `torch.nn.Softmax2d` | ✅ | `aclnnSoftmax` | `aclnnSoftmaxBackward` | - |
