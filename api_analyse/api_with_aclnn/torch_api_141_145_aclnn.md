# torch API 141-145 aclnn 接入分析

## 接口 141：`torch.nn.functional.dropout3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::feature_dropout` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::new_empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::bernoulli_` | 计算接口 | 是 | `aclnnInplaceBernoulli` | opapi/BernoulliKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `as_strided_backward` | view | N/A | - | view backward（4D 无 batch 输入时） |

> **注意**：`aten::feature_dropout` 是 `CompositeImplicitAutograd` 接口，其前向分解为 `new_empty` + `bernoulli_` + `div_` + `mul`，这些子算子均已接入 aclnn 或为框架/view 接口。但 `feature_dropout` 本身作为 composite 入口在 NPU 上无独立注册，需要依赖 PyTorch 的 composite 分解机制。由于分解后的子算子均可支持，实际可在 A5 上运行。

---

## 接口 142：`torch.nn.functional.gaussian_nll_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::clone` | 计算接口 | 是 | `aclnnInplaceCopy` | CloneKernelOpApi.cpp |
| `aten::clamp_` | 计算接口 | 是 | `aclnnClamp` | StructKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `as_strided_backward` | view | N/A | - | view backward |

---

## 接口 143：`torch.nn.functional.gumbel_softmax`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::exponential_` | 计算接口 | 否 | - | 基于 uniform_/sub_/mul_/log_ 组合实现，未直接使用 aclnn |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::softmax.int` | 计算接口 | 是 | `aclnnSoftmax` | opapi/SoftmaxKernelNpuOpApi.cpp |
| `aten::_softmax` | 计算接口 | 是 | `aclnnSoftmax` | opapi/StructKernelNpuOpApi.cpp |
| `aten::max.dim` | 计算接口 | 是 | `aclnnMaxDim` | MaxKernelNpuOpApi.cpp |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::scatter_.value` | 计算接口 | 是 | `aclnnScatterValue` | opapi/ScatterKernelNpuOpApi.cpp |
| `aten::detach` | view | N/A | - | composite_view |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_softmax_backward_data` | 计算接口 | 是 | `aclnnSoftmaxBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

> **注意**：`aten::exponential_` 虽标记为 aclnn=否，但其 NPU 实现在 opapi/ExponentialKernelNpuOpApi.cpp 中通过 uniform_/sub_/mul_/log_ 组合实现，这些底层算子均已接入 aclnn。因此 `exponential_` 在 NPU 上可以正常执行，不影响 A5 支持。

---

## 接口 144：`torch.nn.functional.hardtanh`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hardtanh` | 计算接口 | 是 | `aclnnHardtanh` | opapi/StructKernelNpuOpApi.cpp |
| `aten::clamp.out` | 计算接口 | 是 | `aclnnClamp` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hardtanh_backward` | 计算接口 | 是 | `aclnnHardtanhBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 接口 145：`torch.nn.functional.hinge_embedding_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hinge_embedding_loss` | 计算接口 | 否 | - | CompositeImplicitAutograd，分解为下列子算子 |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::sub.Scalar` | 计算接口 | 是 | `aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::clamp_min_` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> **注意**：`aten::hinge_embedding_loss` 本身是 CompositeImplicitAutograd 接口，在 NPU 上无独立注册（aclnn=否），但其分解后的所有子算子均已接入 aclnn，因此通过 composite 分解机制可正常运行。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 141 | `torch.nn.functional.dropout3d` | ✅ | `aclnnInplaceBernoulli`、`aclnnInplaceDiv`、`aclnnMul` | `aclnnMul` | `aten::feature_dropout`（正向，CIA 分解后子算子均可支持） |
| 142 | `torch.nn.functional.gaussian_nll_loss` | ✅ | `aclnnInplaceCopy`、`aclnnClamp`、`aclnnLog`、`aclnnSub`、`aclnnPowTensorScalar`、`aclnnDiv`、`aclnnAdd`、`aclnnMuls`、`aclnnMean` | `aclnnLog`、`aclnnSub`、`aclnnPowTensorScalar`、`aclnnDiv`、`aclnnAdd`、`aclnnMuls`、`aclnnMean`、`aclnnReduceSum` | - |
| 143 | `torch.nn.functional.gumbel_softmax` | ✅ | `aclnnLog`、`aclnnNeg`、`aclnnAdd`、`aclnnDivs`、`aclnnSoftmax`、`aclnnMaxDim`、`aclnnInplaceZero`、`aclnnScatterValue`、`aclnnSub` | `aclnnSoftmaxBackward`、`aclnnDivs`、`aclnnAdd`、`aclnnNeg`、`aclnnSub` | `aten::exponential_`（正向，NPU 通过已有 aclnn 算子组合实现） |
| 144 | `torch.nn.functional.hardtanh` | ✅ | `aclnnHardtanh`、`aclnnClamp` | `aclnnHardtanhBackward` | - |
| 145 | `torch.nn.functional.hinge_embedding_loss` | ✅ | `aclnnInplaceZero`、`aclnnSubs`、`aclnnClampMin`、`aclnnSWhere`、`aclnnAdd`、`aclnnMean`、`aclnnReduceSum` | `aclnnSWhere`、`aclnnClampMin`、`aclnnAdd`、`aclnnMean`、`aclnnReduceSum` | `aten::hinge_embedding_loss`（正向，CIA 分解后子算子均可支持） |
