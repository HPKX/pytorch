# torch API 146-150 aclnn 接入分析

## 接口 146：`torch.nn.functional.huber_loss`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::huber_loss` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::huber_loss_backward` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

---

## 接口 147：`torch.nn.functional.lp_pool1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::avg_pool1d` | 计算接口 | 否 | - | CIA 分解为 unsqueeze → avg_pool2d → squeeze |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::avg_pool2d` | 计算接口 | 是 | `aclnnAvgPool2d` | opapi/AvgPool2dKernelNpuOpApi.cpp |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::sign` | 计算接口 | 是 | `aclnnSign` | StructKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | StructKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::avg_pool2d_backward` | 计算接口 | 是 | `aclnnAvgPool2dBackward` | opapi/AvgPool2dBackwardKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `as_strided_backward` | view | N/A | - | view backward |

> **注意**：`aten::avg_pool1d` 标记为 aclnn=否，但它是 CIA 接口，分解为 `unsqueeze → avg_pool2d → squeeze`，这些子算子均已接入 aclnn 或为 view 接口，不影响 A5 支持。

---

## 接口 148：`torch.nn.functional.lp_pool2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::avg_pool2d` | 计算接口 | 是 | `aclnnAvgPool2d` | opapi/AvgPool2dKernelNpuOpApi.cpp |
| `aten::sign` | 计算接口 | 是 | `aclnnSign` | StructKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | StructKernelNpuOpApi.cpp |
| `aten::relu` | 计算接口 | 是 | `aclnnRelu` | StructKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | StructKernelNpuOpApi.cpp |
| `aten::avg_pool2d_backward` | 计算接口 | 是 | `aclnnAvgPool2dBackward` | opapi/AvgPool2dBackwardKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::threshold_backward` | 计算接口 | 是 | `aclnnThresholdBackward` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |

---

## 接口 149：`torch.nn.functional.margin_ranking_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::margin_ranking_loss` | 计算接口 | 否 | - | CompositeImplicitAutograd，分解为下列子算子 |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> **注意**：`aten::margin_ranking_loss` 本身是 CIA 接口（aclnn=否），但分解后的所有子算子均已接入 aclnn。

---

## 接口 150：`torch.nn.functional.max_pool3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool3d` | 计算接口 | 否 | - | CIA 包装层，委托给 max_pool3d_with_indices |
| `aten::max_pool3d_with_indices` | 计算接口 | 是 | `aclnnMaxPool3dWithArgmax` | MaxPool3dWithIndicesKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool3d_with_indices_backward` | 计算接口 | 是 | `aclnnMaxPool3dWithArgmaxBackward` | MaxPool3dWithIndicesBackwardKernelNpuOpApi.cpp |

> **注意**：`aten::max_pool3d` 是 CIA 包装层（aclnn=否），直接委托给 `max_pool3d_with_indices`，后者已接入 aclnn。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 146 | `torch.nn.functional.huber_loss` | ❌ | `aclnnMean`、`aclnnReduceSum`、`aclnnMul` | `aclnnMul`、`aclnnMean`、`aclnnReduceSum` | **`aten::huber_loss`（正向）、`aten::huber_loss_backward`（反向）** |
| 147 | `torch.nn.functional.lp_pool1d` | ✅ | `aclnnPowTensorScalar`、`aclnnAvgPool2d`、`aclnnSign`、`aclnnAbs`、`aclnnRelu`、`aclnnMuls` | `aclnnPowTensorScalar`、`aclnnAvgPool2dBackward`、`aclnnSign`、`aclnnThresholdBackward`、`aclnnMul`、`aclnnMuls`、`aclnnInplaceZero` | `aten::avg_pool1d`（正向，CIA 分解后子算子均可支持） |
| 148 | `torch.nn.functional.lp_pool2d` | ✅ | `aclnnPowTensorScalar`、`aclnnAvgPool2d`、`aclnnSign`、`aclnnAbs`、`aclnnRelu`、`aclnnMuls` | `aclnnPowTensorScalar`、`aclnnAvgPool2dBackward`、`aclnnSign`、`aclnnThresholdBackward`、`aclnnMul`、`aclnnMuls`、`aclnnInplaceZero` | - |
| 149 | `torch.nn.functional.margin_ranking_loss` | ✅ | `aclnnSub`、`aclnnMul`、`aclnnNeg`、`aclnnAdds`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | `aclnnSub`、`aclnnMul`、`aclnnNeg`、`aclnnAdds`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | `aten::margin_ranking_loss`（正向，CIA 分解后子算子均可支持） |
| 150 | `torch.nn.functional.max_pool3d` | ✅ | `aclnnMaxPool3dWithArgmax` | `aclnnMaxPool3dWithArgmaxBackward` | `aten::max_pool3d`（正向，CIA 包装层，委托给已支持的 with_indices 变体） |
