# torch API 91-95 aclnn 接入分析

## 接口 91：`torch.nn.ConvTranspose3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::conv_transpose3d.input` | 计算接口 | 是 | `aclnnConvolution` | 通过 convolution 路径（transposed=true） |
| `aten::convolution` | 计算接口 | 是 | `aclnnConvolution` | opapi/ConvolutionKernelNpuOpApi.cpp |
| `aten::cudnn_convolution_transpose` | 计算接口 | 否 | - | CUDA 专用接口，NPU 不走此路径 |
| `aten::miopen_convolution_transpose` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 不走此路径 |
| `aten::slow_conv_transpose3d` | 计算接口 | 否 | - | op_plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::convolution_backward` | 计算接口 | 是 | `aclnnConvolutionBackward` | opapi/ConvolutionBackwardKernelNpuOpApi.cpp |

> `aten::cudnn_convolution_transpose`、`aten::miopen_convolution_transpose`、`aten::slow_conv_transpose3d` 是 CUDA/MIOpen/CPU 专用 fallback 路径，NPU 上不会被调度到，NPU 走 `aten::convolution` → `aclnnConvolution` 路径。

---

## 接口 92：`torch.nn.Dropout1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::feature_dropout` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::new_empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::bernoulli_` | 计算接口 | 是 | `aclnnInplaceBernoulli` | opapi/BernoulliKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::squeeze` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::squeeze` | view | N/A | - | view 类接口 |

> `aten::feature_dropout` 虽标记为计算接口/否，但它是 CIA 分解函数，前向分解为 `bernoulli_`（aclnn=是）+ `div_`（aclnn=是）+ `mul.Tensor`（aclnn=是），实际计算由 aclnn 支持的子 op 完成。

---

## 接口 93：`torch.nn.Dropout2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::feature_dropout` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::new_empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::bernoulli_` | 计算接口 | 是 | `aclnnInplaceBernoulli` | opapi/BernoulliKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

> `aten::feature_dropout` 虽标记为计算接口/否，但同 Dropout1d 分析，实际计算由 aclnn 支持的子 op 完成。

---

## 接口 94：`torch.nn.Dropout3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::feature_dropout` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::new_empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::bernoulli_` | 计算接口 | 是 | `aclnnInplaceBernoulli` | opapi/BernoulliKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::squeeze` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::squeeze` | view | N/A | - | view 类接口 |

> `aten::feature_dropout` 虽标记为计算接口/否，但同 Dropout1d 分析，实际计算由 aclnn 支持的子 op 完成。

---

## 接口 95：`torch.nn.GRU`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gru.input` | 计算接口 | 否 | - | aclops/GruKernelNpu.cpp，无 aclnn，使用旧 ACL 算子 |
| `aten::gru.data` | 计算接口 | 否 | - | aclops/GruKernelNpu.cpp，无 aclnn，使用旧 ACL 算子 |
| `aten::_cudnn_rnn` | 计算接口 | 否 | - | CUDA 专用接口，NPU 未实现 |
| `aten::miopen_rnn` | 计算接口 | 否 | - | MIOpen 专用接口，NPU 未实现 |
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm / aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::_thnn_fused_gru_cell` | 计算接口 | 否 | - | composite 操作组合实现，无直接 aclnn 调用 |
| `aten::sigmoid_` | 计算接口 | 是 | `aclnnInplaceSigmoid` | opapi/StructKernelNpuOpApi.cpp |
| `aten::tanh_` | 计算接口 | 是 | `aclnnInplaceTanh` | opapi/TanhKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_cudnn_rnn_backward` | 计算接口 | 否 | - | CUDA 专用接口，NPU 未实现 |
| `aten::_thnn_fused_gru_cell_backward` | 计算接口 | 否 | - | op_plugin 中无实现，fallback |
| `aten::_thnn_differentiable_gru_cell_backward` | 计算接口 | 否 | - | op_plugin 中无实现，fallback |
| `aten::sigmoid_backward` | 计算接口 | 是 | `aclnnSigmoidBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::tanh_backward` | 计算接口 | 是 | `aclnnTanhBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> `aten::gru.input` 和 `aten::gru.data` 是 GRU 的核心前向接口，仅有旧 ACL 实现（aclops），未迁移到 aclnn。虽然存在 CPU fallback 和 fused cell 路径，但 GRU 主路径的前向计算无法在 A5 上通过 aclnn 完成。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 91 | `torch.nn.ConvTranspose3d` | ✅ | `aclnnConvolution` | `aclnnConvolutionBackward` | - |
| 92 | `torch.nn.Dropout1d` | ✅ | `aclnnInplaceBernoulli`, `aclnnInplaceDiv`, `aclnnMul` | `aclnnMul` | - |
| 93 | `torch.nn.Dropout2d` | ✅ | `aclnnInplaceBernoulli`, `aclnnInplaceDiv`, `aclnnMul` | `aclnnMul` | - |
| 94 | `torch.nn.Dropout3d` | ✅ | `aclnnInplaceBernoulli`, `aclnnInplaceDiv`, `aclnnMul` | `aclnnMul` | - |
| 95 | `torch.nn.GRU` | ❌ | `aclnnAddmm`, `aclnnInplaceSigmoid`, `aclnnInplaceTanh`, `aclnnInplaceMul`, `aclnnInplaceAdd` | `aclnnSigmoidBackward`, `aclnnTanhBackward`, `aclnnMul`, `aclnnSub`, `aclnnCat`, `aclnnReduceSum` | **`aten::gru.input`（正向）**, **`aten::gru.data`（正向）**, **`aten::_thnn_fused_gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell_backward`（反向）**, **`aten::_thnn_differentiable_gru_cell_backward`（反向）** |
