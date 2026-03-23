# torch API 171-175 aclnn 接入分析

## 接口 171：`torch.optim.Rprop`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sign` | 计算接口 | 是 | `aclnnSign` | StructKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::clamp_` | 计算接口 | 是 | `aclnnClamp` | StructKernelNpuOpApi.cpp |
| `aten::addcmul_` | 计算接口 | 是 | `aclnnInplaceAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |
| `aten::_foreach_add_.Scalar` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_mul` | 计算接口 | 是 | `aclnnForeachMulList` | opapi/ForeachMulKernelNpuOpApi.cpp |
| `aten::_foreach_neg_` | 计算接口 | 是 | `aclnnForeachNeg` | opapi/ForeachNegKernelNpuOpApi.cpp |
| `aten::_foreach_copy_` | 计算接口 | 是 | `aclnnForeachCopy` | opapi/ForeachCopyKernelOpApi.cpp |
| `aten::_foreach_sign_` | 计算接口 | 是 | `aclnnForeachSign` | opapi/ForeachSignKernelNpuOpApi.cpp |
| `aten::_foreach_mul_` | 计算接口 | 是 | `aclnnForeachMulList` | opapi/ForeachMulKernelNpuOpApi.cpp |
| `aten::_foreach_addcmul_` | 计算接口 | 是 | `aclnnForeachAddcmulScalar` | opapi/ForeachAddcmulScalarKernelNpuOpApi.cpp |

### 反向依赖

无（优化器，默认 `no_grad`，不可微）。

---

## 接口 172：`torch.optim.adadelta.Adadelta`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::addcmul_` | 计算接口 | 是 | `aclnnInplaceAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::sqrt_` | 计算接口 | 是 | `aclnnInplaceSqrt` | StructKernelNpuOpApi.cpp |
| `aten::div_` | 计算接口 | 是 | `aclnnInplaceDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::_foreach_add_.Scalar` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_neg` | 计算接口 | 是 | `aclnnForeachNeg` | opapi/ForeachNegKernelNpuOpApi.cpp |
| `aten::_foreach_add` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_add_` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_mul_` | 计算接口 | 是 | `aclnnForeachMulList` | opapi/ForeachMulKernelNpuOpApi.cpp |
| `aten::_foreach_addcmul_` | 计算接口 | 是 | `aclnnForeachAddcmulScalar` | opapi/ForeachAddcmulScalarKernelNpuOpApi.cpp |
| `aten::_foreach_sqrt_` | 计算接口 | 是 | `aclnnForeachSqrt` | opapi/ForeachSqrtKernelNpuOpApi.cpp |
| `aten::_foreach_div_` | 计算接口 | 是 | `aclnnForeachDivScalar` | opapi/ForeachDivKernelNpuOpApi.cpp |

### 反向依赖

无（优化器，默认 `no_grad`，不可微）。

---

## 接口 173：`torch.optim.adagrad.Adagrad`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::addcmul_` | 计算接口 | 是 | `aclnnInplaceAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::sqrt` | 计算接口 | 是 | `aclnnSqrt` | StructKernelNpuOpApi.cpp |
| `aten::sqrt_` | 计算接口 | 是 | `aclnnInplaceSqrt` | StructKernelNpuOpApi.cpp |
| `aten::addcdiv_` | 计算接口 | 是 | `aclnnInplaceAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |
| `aten::pow` | 计算接口 | 是 | `aclnnPowTensorTensor` | StructKernelNpuOpApi.cpp |
| `aten::sparse_mask` | 计算接口 | 否 | - | op_plugin 中未找到实现 |
| `aten::_foreach_add_.Scalar` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_neg` | 计算接口 | 是 | `aclnnForeachNeg` | opapi/ForeachNegKernelNpuOpApi.cpp |
| `aten::_foreach_add` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_add_` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_addcmul_` | 计算接口 | 是 | `aclnnForeachAddcmulScalar` | opapi/ForeachAddcmulScalarKernelNpuOpApi.cpp |
| `aten::_foreach_sqrt` | 计算接口 | 是 | `aclnnForeachSqrt` | opapi/ForeachSqrtKernelNpuOpApi.cpp |
| `aten::_foreach_mul` | 计算接口 | 是 | `aclnnForeachMulList` | opapi/ForeachMulKernelNpuOpApi.cpp |
| `aten::_foreach_mul_` | 计算接口 | 是 | `aclnnForeachMulList` | opapi/ForeachMulKernelNpuOpApi.cpp |
| `aten::_foreach_addcdiv_` | 计算接口 | 是 | `aclnnForeachAddcdivScalar` | opapi/ForeachAddcdivScalarKernelNpuOpApi.cpp |
| `aten::_fused_adagrad_` | 计算接口 | 否 | - | op_plugin 及 PTA 中均未找到实现，fallback |

### 反向依赖

无（优化器，默认 `no_grad`，不可微）。

> **注意**：
> - `aten::sparse_mask` 仅在 sparse grad 路径使用，dense grad 路径不依赖此接口。NPU 上 sparse tensor 支持有限，通常使用 dense grad 路径。
> - `aten::_fused_adagrad_` 仅在 `fused=True`（仅 CPU）路径使用，NPU 上通常走 `foreach=True` 路径，不走 fused 路径。
> - 常规 NPU 使用场景（dense grad + foreach=True）下所有算子均已接入 aclnn。

---

## 接口 174：`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

无 ATen dispatch 依赖（纯 Python 调度逻辑，使用 `math.cos` 等标量运算计算学习率）。

### 反向依赖

无（不可微）。

---

## 接口 175：`torch.orgqr`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::orgqr` | 计算接口 | 否 | - | op_plugin 中无实现 |
| `aten::linalg_householder_product` | 计算接口 | 否 | - | op_plugin 中无实现 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::linalg_householder_product.out` | 计算接口 | 否 | - | op_plugin 中无实现 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_householder_product` | 计算接口 | 否 | - | op_plugin 中无实现 |
| `aten::tril` | 计算接口 | 是 | `aclnnTril` | StructKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::fill_` | 计算接口 | 是 | `aclnnInplaceFillScalar` | opapi/FillKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::matmul` | 计算接口 | 否 | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 171 | `torch.optim.Rprop` | ✅ | `aclnnNeg`、`aclnnMul`、`aclnnSign`、`aclnnInplaceMul`、`aclnnClamp`、`aclnnInplaceAddcmul`、`aclnnInplaceCopy`、`aclnnForeachAddScalar`、`aclnnForeachMulList`、`aclnnForeachNeg`、`aclnnForeachCopy`、`aclnnForeachSign`、`aclnnForeachAddcmulScalar` | 无（优化器，不可微） | - |
| 172 | `torch.optim.adadelta.Adadelta` | ✅ | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnInplaceAddcmul`、`aclnnInplaceSqrt`、`aclnnInplaceDiv`、`aclnnForeachAddScalar`、`aclnnForeachNeg`、`aclnnForeachAddScalar`、`aclnnForeachMulList`、`aclnnForeachAddcmulScalar`、`aclnnForeachSqrt`、`aclnnForeachDivScalar` | 无（优化器，不可微） | - |
| 173 | `torch.optim.adagrad.Adagrad` | ✅ | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceAddcmul`、`aclnnSqrt`、`aclnnInplaceSqrt`、`aclnnInplaceAddcdiv`、`aclnnPowTensorTensor`、`aclnnForeachAddScalar`、`aclnnForeachNeg`、`aclnnForeachAddcmulScalar`、`aclnnForeachSqrt`、`aclnnForeachMulList`、`aclnnForeachAddcdivScalar` | 无（优化器，不可微） | `aten::sparse_mask`（仅 sparse grad 路径）、`aten::_fused_adagrad_`（仅 CPU fused 路径），常规 NPU 路径不依赖 |
| 174 | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` | ✅ | 无 ATen 依赖 | 无（不可微） | - |
| 175 | `torch.orgqr` | ❌ | `aclnnInplaceCopy` | `aclnnTril`、`aclnnInplaceFillScalar`、`aclnnReduceSum`、`aclnnInplaceZero`、`aclnnCat`、`aclnnInplaceCopy` | **`aten::orgqr`（正向）、`aten::linalg_householder_product`（正反向）、`aten::linalg_householder_product.out`（正向）** |
