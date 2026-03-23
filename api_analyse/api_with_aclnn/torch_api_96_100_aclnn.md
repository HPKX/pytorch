# torch API 96-100 aclnn 接入分析

## 接口 96：`torch.nn.GRUCell`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gru_cell` | 计算接口 | 否 | - | _thnn_fused_gru_cell 为组合实现，无直接 aclnn |
| `aten::linear` | 计算接口 | 是 | `aclnnAddmm / aclnnMm` | opapi/LinearKernelNpuOpApi.cpp |
| `aten::unsafe_chunk` | view | N/A | - | CIA composite（无 dispatch key），等价于不检查边界的 chunk，纯框架层分块操作 |
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sigmoid_` | 计算接口 | 是 | `aclnnInplaceSigmoid` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::tanh_` | 计算接口 | 是 | `aclnnInplaceTanh` | opapi/TanhKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::t` | view | N/A | - | view 类接口 |
| `aten::matmul` | 计算接口 | 否 | - | PyTorch composite 接口，分解为 mm/bmm/dot 等 |
| `aten::_thnn_fused_gru_cell` | 计算接口 | 否 | - | composite 操作组合实现，无直接 aclnn 调用 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_thnn_fused_gru_cell_backward` | 计算接口 | 否 | - | op_plugin 中无实现，fallback |
| `aten::_thnn_differentiable_gru_cell_backward` | 计算接口 | 否 | - | op_plugin 中无实现，fallback |
| `aten::sigmoid_backward` | 计算接口 | 是 | `aclnnSigmoidBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::tanh_backward` | 计算接口 | 是 | `aclnnTanhBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> `aten::gru_cell` 和 `aten::_thnn_fused_gru_cell` 是 GRUCell 的核心前向接口，均无 aclnn 实现。虽然 CPU fallback 路径的基础算子多数有 aclnn，但 fused cell 路径的 backward（`_thnn_fused_gru_cell_backward` 和 `_thnn_differentiable_gru_cell_backward`）无 aclnn 实现，无法在 A5 上完整支持。

---

## 接口 97：`torch.nn.GaussianNLLLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::lt.Scalar` | 计算接口 | 是 | `aclnnLtScalar` | opapi/LtKernelNpuOpApi.cpp |
| `aten::any` | 计算接口 | 是 | `aclnnAny` | opapi/AnyKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::clone` | 计算接口 | 是 | `aclnnInplaceCopy` | CloneKernelOpApi.cpp |
| `aten::clamp_` | 计算接口 | 是 | `aclnnClamp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |

---

## 接口 98：`torch.nn.HardTanh`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hardtanh` | 计算接口 | 是 | `aclnnHardtanh` | opapi/StructKernelNpuOpApi.cpp |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::hardtanh.out` | 计算接口 | 是 | `aclnnHardtanh` | opapi/StructKernelNpuOpApi.cpp |
| `aten::clamp.out` | 计算接口 | 是 | `aclnnClamp` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hardtanh_backward` | 计算接口 | 是 | `aclnnHardtanhBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 接口 99：`torch.nn.HingeEmbeddingLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hinge_embedding_loss` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::rsub.Scalar` | 计算接口 | 是 | `aclnnRsubs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::clamp_min_` | 计算接口 | 是 | `aclnnClampMin` | opapi/StructKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | opapi/StructKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::ge.Scalar` | 计算接口 | 是 | `aclnnGeScalar` | opapi/GeKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |

> `aten::hinge_embedding_loss` 虽标记为计算接口/否，但它是 Python composite 函数，前向分解为 `zeros_like`、`rsub`、`clamp_min`、`where`、`add`、`mean` 等 aclnn 已支持的子 op。

---

## 接口 100：`torch.nn.HuberLoss`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::broadcast_tensors` | view | N/A | - | CompositeImplicitAutograd，expand（view） |
| `aten::huber_loss` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::huber_loss_backward` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 96 | `torch.nn.GRUCell` | ❌ | `aclnnAddmm`, `aclnnInplaceAdd`, `aclnnInplaceSigmoid`, `aclnnInplaceMul`, `aclnnInplaceTanh`, `aclnnSub` | `aclnnSigmoidBackward`, `aclnnTanhBackward`, `aclnnMul`, `aclnnSub`, `aclnnCat`, `aclnnReduceSum` | **`aten::gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell_backward`（反向）**, **`aten::_thnn_differentiable_gru_cell_backward`（反向）** |
| 97 | `torch.nn.GaussianNLLLoss` | ✅ | `aclnnLtScalar`, `aclnnAny`, `aclnnInplaceCopy`, `aclnnClamp`, `aclnnSub`, `aclnnPowTensorScalar`, `aclnnDiv`, `aclnnLog`, `aclnnAdd`, `aclnnMuls`, `aclnnMean` | `aclnnNeg`, `aclnnMul`, `aclnnMuls`, `aclnnDiv`, `aclnnPowTensorScalar`, `aclnnDivs` | - |
| 98 | `torch.nn.HardTanh` | ✅ | `aclnnHardtanh`, `aclnnClamp` | `aclnnHardtanhBackward` | - |
| 99 | `torch.nn.HingeEmbeddingLoss` | ✅ | `aclnnInplaceZero`, `aclnnRsubs`, `aclnnClampMin`, `aclnnSWhere`, `aclnnAdd`, `aclnnMean` | `aclnnNeg`, `aclnnGeScalar`, `aclnnSWhere`, `aclnnDivs` | - |
| 100 | `torch.nn.HuberLoss` | ❌ | `aclnnMean` | - | **`aten::huber_loss`（正向）**, **`aten::huber_loss_backward`（反向）** |
