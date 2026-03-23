# torch API 81-85 aclnn 接入分析

## 接口 81：`torch.mvlgamma`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mvlgamma` | 计算接口 | **否** | - | op_plugin 中无实现 |
| `aten::arange.start_step` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::lgamma_` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::sum.dim_IntList` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::add_.Scalar` | 计算接口 | 是 | `aclnnInplaceAdds` | opapi/AddKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::arange` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::digamma_` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

---

## 接口 82：`torch.nanmean`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::nanmean` | 计算接口 | 否 | - | CIA composite，分解为 isnan+nansum+div 等（均已支持） |
| `aten::detach` | view | N/A | - | composite_view |
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::logical_not_` | 计算接口 | 是 | `aclnnInplaceLogicalNot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sum.dim_IntList` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::nansum` | 计算接口 | 是 | `aclnnReduceNansum` | NansumKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::logical_not` | 计算接口 | 是 | `aclnnLogicalNot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 83：`torch.nanmedian`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::nanmedian` | 计算接口 | 是 | `aclnnNanMedian` | StructKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::nanmedian.dim` | 计算接口 | 是 | `aclnnNanMedianDim` | StructKernelNpuOpApi.cpp |
| `aten::nanmedian.dim_values` | 计算接口 | 是 | `aclnnNanMedianDim` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::logical_and_` | 计算接口 | 是 | `aclnnInplaceLogicalAnd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::logical_or_` | 计算接口 | 是 | `aclnnInplaceLogicalOr` | opapi/LogicalOrKernelNpuOpApi.cpp |
| `aten::eq.Tensor` | 计算接口 | 是 | `aclnnEqTensor` | opapi/EqKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::new_zeros` | 计算接口 | 是 | `aclnnInplaceZero` | composite: empty + zero_() |
| `aten::masked_fill_` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::scatter_` | 计算接口 | 是 | `aclnnInplaceScatter` | opapi/ScatterKernelNpuOpApi.cpp |

---

## 接口 84：`torch.nextafter`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::nextafter` | 计算接口 | **否** | - | op_plugin 中无实现 |

### 反向依赖

无（`not_implemented`，反向会抛异常）

---

## 接口 85：`torch.nn.AdaptiveMaxPool3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::adaptive_max_pool3d` | 计算接口 | 是 | `aclnnAdaptiveMaxPool3d` | opapi/AdaptiveMaxPool3dKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::adaptive_max_pool3d_backward` | 计算接口 | 是 | `aclnnAdaptiveMaxPool3dBackward` | opapi/AdaptiveMaxPool3dBackwardKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 81 | `torch.mvlgamma` | ❌ | `aclnnArange`、`aclnnAdd`、`aclnnReduceSum`、`aclnnInplaceAdds` | `aclnnArange`、`aclnnAdd`、`aclnnReduceSum`、`aclnnMul` | 正向：`aten::mvlgamma`、`aten::lgamma_`；反向：`aten::digamma_` |
| 82 | `torch.nanmean` | ✅ | `aclnnInplaceLogicalNot`、`aclnnReduceSum`、`aclnnReduceNansum`、`aclnnDiv` | `aclnnLogicalNot`、`aclnnMul`、`aclnnDiv`、`aclnnNeg` | - |
| 83 | `torch.nanmedian` | ✅ | `aclnnNanMedian`、`aclnnNanMedianDim` | `aclnnInplaceLogicalAnd`、`aclnnInplaceLogicalOr`、`aclnnEqTensor`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMul`、`aclnnInplaceZero`、`aclnnInplaceMaskedFillScalar`、`aclnnInplaceScatter` | - |
| 84 | `torch.nextafter` | ❌ | - | 无（不可微） | 正向：`aten::nextafter` |
| 85 | `torch.nn.AdaptiveMaxPool3d` | ✅ | `aclnnAdaptiveMaxPool3d` | `aclnnAdaptiveMaxPool3dBackward` | - |
