# torch API 136-140 aclnn 接入分析

## 接口 136：`torch.nn.functional.celu`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::celu` | 计算接口 | 是 | `aclnnCelu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::celu_` | 计算接口 | 是 | `aclnnInplaceCelu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::elu` | 计算接口 | 是 | `aclnnElu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::elu_` | 计算接口 | 是 | `aclnnInplaceElu` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::elu_backward` | 计算接口 | 是 | `aclnnEluBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 接口 137：`torch.nn.functional.cosine_similarity`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::cosine_similarity` | 计算接口 | **否** | - | CIA composite 接口，分解为 norm/div/mul/sum 等 |
| `aten::to` | 计算接口 | 是 | `aclnnInplaceCopy` | ToKernelNpu.cpp（间接） |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::linalg_vector_norm` | 计算接口 | 是 | `aclnnLinalgVectorNorm` | opapi/LinalgNormKernelNpuOpApi.cpp |
| `aten::clone` | 计算接口 | 是 | `aclnnInplaceCopy` | CloneKernelOpApi.cpp |
| `aten::scalar_tensor` | 框架接口 | N/A | - | 标量转张量，框架接口 |
| `aten::clamp_min_` | 计算接口 | 是 | `aclnnClampMin` | 通过 clamp_min_out 路径 |
| `aten::div` | 计算接口 | 是 | `aclnnDiv` / `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` / `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_vector_norm_backward` | 计算接口 | **否** | - | autograd helper（非独立 ATen 接口），CIA 分解为 div/mul/pow 等已接入 aclnn 的基础算子，不阻塞 |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sum.dim_IntList` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::expand` | view | N/A | - | view 类接口 |

> **备注**：`aten::cosine_similarity` 是 CIA composite 函数，实际计算由 `linalg_vector_norm`、`div`、`mul`、`sum` 等承担。`aten::linalg_vector_norm_backward` 标记为计算接口=否，但它是 `linalg_vector_norm` 的 derivatives.yaml helper，其内部实现会分解为 `div`、`mul`、`pow` 等已接入 aclnn 的基础算子（CompositeImplicitAutograd），因此不影响 A5 支持判定。

---

## 接口 138：`torch.nn.functional.ctc_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
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

## 接口 139：`torch.nn.functional.dropout1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unsqueeze` | view | N/A | - | view 类接口（仅 2D 输入时） |
| `aten::feature_dropout` | 计算接口 | **否** | - | CIA composite，分解为 new_empty + bernoulli_ + div_ + mul |
| `aten::feature_dropout_` | 计算接口 | **否** | - | CIA composite，inplace 变体 |
| `aten::new_empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::bernoulli_` | 计算接口 | 是 | `aclnnInplaceBernoulli` | opapi/BernoulliKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` / `aclnnInplaceDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` / `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` / `aclnnInplaceMuls` | MulKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口（仅 2D 输入时） |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

> **备注**：`aten::feature_dropout` 标记为计算接口=否，但它是 CIA composite 函数，训练态实际为 `input * mask`，计算由 `mul`（已接入 aclnn）承担。mask 生成链（`new_empty` + `bernoulli_` + `div_`）中 `bernoulli_` 和 `div_` 均已接入 aclnn。`training=False` 时为 identity。

---

## 接口 140：`torch.nn.functional.dropout2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::feature_dropout` | 计算接口 | **否** | - | CIA composite，分解为 new_empty + bernoulli_ + div_ + mul |
| `aten::feature_dropout_` | 计算接口 | **否** | - | CIA composite，inplace 变体 |
| `aten::new_empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::bernoulli_` | 计算接口 | 是 | `aclnnInplaceBernoulli` | opapi/BernoulliKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` / `aclnnInplaceDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` / `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` / `aclnnInplaceMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

> **备注**：与 `dropout1d` 相同，`aten::feature_dropout` 是 CIA composite 函数，实际计算由已接入 aclnn 的算子承担，不影响 A5 支持判定。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 136 | `torch.nn.functional.celu` | ✅ | `aclnnCelu`、`aclnnElu` | `aclnnEluBackward` | - |
| 137 | `torch.nn.functional.cosine_similarity` | ✅ | `aclnnLinalgVectorNorm`、`aclnnDiv`、`aclnnMul`、`aclnnReduceSum`、`aclnnClampMin` | `aclnnDiv`、`aclnnMul`、`aclnnReduceSum` | - |
| 138 | `torch.nn.functional.ctc_loss` | ✅ | `aclnnCtcLoss` | `aclnnCtcLossBackward` | - |
| 139 | `torch.nn.functional.dropout1d` | ✅ | `aclnnInplaceBernoulli`、`aclnnInplaceDiv`、`aclnnMul` | `aclnnMul` | - |
| 140 | `torch.nn.functional.dropout2d` | ✅ | `aclnnInplaceBernoulli`、`aclnnInplaceDiv`、`aclnnMul` | `aclnnMul` | - |
