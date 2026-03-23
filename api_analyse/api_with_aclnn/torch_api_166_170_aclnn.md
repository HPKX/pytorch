# torch API 166-170 aclnn 接入分析

## 接口 166：`torch.nn.modules.flatten.Unflatten`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unflatten.int` | view | N/A | - | 底层 reshape/view，返回共享存储的 view |
| `aten::unflatten.Dimname` | view | N/A | - | 同 unflatten.int，底层 reshape/view |
| `aten::view` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::view` | view | N/A | - | view 类接口 |

---

## 接口 167：`torch.numel`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

无 ATen dispatch 依赖（eager 路径直接读元信息）。

### 反向依赖

无（不可微）。

---

## 接口 168：`torch.optim.ASGD`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |
| `aten::_foreach_add_.Scalar` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_neg` | 计算接口 | 是 | `aclnnForeachNeg` | opapi/ForeachNegKernelNpuOpApi.cpp |
| `aten::_foreach_add.List` | 计算接口 | 是 | `aclnnForeachAddList` | opapi/ForeachAddKernelNpuOpApi.cpp |
| `aten::_foreach_add_.List` | 计算接口 | 是 | `aclnnForeachAddList` | opapi/ForeachAddKernelNpuOpApi.cpp |
| `aten::_foreach_addcmul_.Scalar` | 计算接口 | 是 | `aclnnForeachAddcmulScalar` | opapi/ForeachAddcmulScalarKernelNpuOpApi.cpp |
| `aten::_foreach_sub.List` | 计算接口 | 是 | `aclnnForeachSubList` | opapi/ForeachSubKernelNpuOpApi.cpp |
| `aten::_foreach_copy_` | 计算接口 | 是 | `aclnnForeachCopy` | opapi/ForeachCopyKernelOpApi.cpp |
| `aten::_foreach_maximum_.Scalar` | 计算接口 | 是 | `aclnnForeachMaximumScalar` | opapi/ForeachMaximumKernelNpuOpApi.cpp |
| `aten::_foreach_pow_.Scalar` | 计算接口 | 是 | `aclnnForeachPowScalar` | opapi/ForeachPowScalarKernelNpuOpApi.cpp |
| `aten::_foreach_reciprocal_` | 计算接口 | 是 | `aclnnForeachReciprocal` | opapi/ForeachReciprocalKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

---

## 接口 169：`torch.optim.Adamax`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::lerp_` | 计算接口 | 是 | `aclnnInplaceLerps` | StructKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | StructKernelNpuOpApi.cpp |
| `aten::maximum.out` | 计算接口 | 是 | `aclnnMaximum` | MaxKernelNpuOpApi.cpp |
| `aten::addcdiv_` | 计算接口 | 是 | `aclnnInplaceAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |
| `aten::_foreach_add_.Scalar` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_add.List` | 计算接口 | 是 | `aclnnForeachAddList` | opapi/ForeachAddKernelNpuOpApi.cpp |
| `aten::_foreach_add_.List` | 计算接口 | 是 | `aclnnForeachAddList` | opapi/ForeachAddKernelNpuOpApi.cpp |
| `aten::_foreach_lerp_.Scalar` | 计算接口 | 是 | `aclnnForeachLerpScalar` | opapi/ForeachLerpScalarKernelNpuOpApi.cpp |
| `aten::_foreach_mul_.Scalar` | 计算接口 | 是 | `aclnnForeachMulScalar` | opapi/ForeachMulScalarKernelOpApi.cpp |
| `aten::_foreach_abs` | 计算接口 | 是 | `aclnnForeachAbs` | opapi/ForeachAbsKernelOpApi.cpp |
| `aten::_foreach_abs_` | 计算接口 | 是 | `aclnnForeachAbs` | opapi/ForeachAbsKernelOpApi.cpp |
| `aten::_foreach_maximum_.List` | 计算接口 | 是 | `aclnnForeachMaximumList` | opapi/ForeachMaximumKernelNpuOpApi.cpp |
| `aten::_foreach_addcdiv_.ScalarList` | 计算接口 | 否 | - | fallback 到 slow 实现 |
| `aten::_foreach_pow` | 计算接口 | 是 | `aclnnForeachPowList` | opapi/ForeachPowKernelNpuOpApi.cpp |
| `aten::_foreach_div_` | 计算接口 | 是 | `aclnnForeachDivScalar` | opapi/ForeachDivKernelNpuOpApi.cpp |
| `aten::_foreach_mul` | 计算接口 | 是 | `aclnnForeachMulList` | opapi/ForeachMulKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::amax` | 计算接口 | 是 | `aclnnAmax` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::lerp_` | 计算接口 | 是 | `aclnnInplaceLerps` | StructKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | StructKernelNpuOpApi.cpp |
| `aten::maximum.out` | 计算接口 | 是 | `aclnnMaximum` | MaxKernelNpuOpApi.cpp |
| `aten::addcdiv_` | 计算接口 | 是 | `aclnnInplaceAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |

> **注意**：`aten::_foreach_addcdiv_.ScalarList` 正向为否，但该接口仅在 `foreach=True` 路径使用且 fallback 到 slow（逐张量）实现，slow 路径最终调用已有 aclnn 的 `addcdiv_`，不影响功能正确性。默认 `no_grad` 下优化器步骤不参与 autograd，因此不影响 A5 支持判定。

---

## 接口 170：`torch.optim.RMSprop`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::addcmul_` | 计算接口 | 是 | `aclnnInplaceAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::sqrt` | 计算接口 | 是 | `aclnnSqrt` | StructKernelNpuOpApi.cpp |
| `aten::addcdiv_` | 计算接口 | 是 | `aclnnInplaceAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |
| `aten::lerp_` | 计算接口 | 是 | `aclnnInplaceLerps` | StructKernelNpuOpApi.cpp |
| `aten::addcmul` | 计算接口 | 是 | `aclnnAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::sqrt_` | 计算接口 | 是 | `aclnnInplaceSqrt` | StructKernelNpuOpApi.cpp |
| `aten::_foreach_add_.Scalar` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |
| `aten::_foreach_add.List` | 计算接口 | 是 | `aclnnForeachAddList` | opapi/ForeachAddKernelNpuOpApi.cpp |
| `aten::_foreach_mul_.Scalar` | 计算接口 | 是 | `aclnnForeachMulScalar` | opapi/ForeachMulScalarKernelOpApi.cpp |
| `aten::_foreach_addcmul_.Scalar` | 计算接口 | 是 | `aclnnForeachAddcmulScalar` | opapi/ForeachAddcmulScalarKernelNpuOpApi.cpp |
| `aten::_foreach_lerp_.Scalar` | 计算接口 | 是 | `aclnnForeachLerpScalar` | opapi/ForeachLerpScalarKernelNpuOpApi.cpp |
| `aten::_foreach_addcmul.Scalar` | 计算接口 | 是 | `aclnnForeachAddcmulScalar` | opapi/ForeachAddcmulScalarKernelNpuOpApi.cpp |
| `aten::_foreach_sqrt_` | 计算接口 | 是 | `aclnnForeachSqrt` | opapi/ForeachSqrtKernelNpuOpApi.cpp |
| `aten::_foreach_sqrt` | 计算接口 | 是 | `aclnnForeachSqrt` | opapi/ForeachSqrtKernelNpuOpApi.cpp |
| `aten::_foreach_addcdiv_.Scalar` | 计算接口 | 是 | `aclnnForeachAddcdivScalar` | opapi/ForeachAddcdivScalarKernelNpuOpApi.cpp |
| `aten::_foreach_add_` | 计算接口 | 是 | `aclnnForeachAddScalar` | opapi/ForeachAddScalarKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::add_` | 计算接口 | 是 | `aclnnInplaceAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul_` | 计算接口 | 是 | `aclnnInplaceMul` | MulKernelNpuOpApi.cpp |
| `aten::addcmul_` | 计算接口 | 是 | `aclnnInplaceAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::lerp_` | 计算接口 | 是 | `aclnnInplaceLerps` | StructKernelNpuOpApi.cpp |
| `aten::sqrt` | 计算接口 | 是 | `aclnnSqrt` | StructKernelNpuOpApi.cpp |
| `aten::sqrt_` | 计算接口 | 是 | `aclnnInplaceSqrt` | StructKernelNpuOpApi.cpp |
| `aten::addcdiv_` | 计算接口 | 是 | `aclnnInplaceAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 166 | `torch.nn.modules.flatten.Unflatten` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 167 | `torch.numel` | ✅ | 无 ATen 依赖 | 无（不可微） | - |
| 168 | `torch.optim.ASGD` | ✅ | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnSub`、`aclnnInplaceCopy`、`aclnnForeachAddScalar`、`aclnnForeachNeg`、`aclnnForeachAddList`、`aclnnForeachAddcmulScalar`、`aclnnForeachSubList`、`aclnnForeachCopy`、`aclnnForeachMaximumScalar`、`aclnnForeachPowScalar`、`aclnnForeachReciprocal` | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnSub`、`aclnnInplaceCopy` | - |
| 169 | `torch.optim.Adamax` | ✅ | `aclnnInplaceAdd`、`aclnnAdd`、`aclnnInplaceLerps`、`aclnnInplaceMul`、`aclnnAbs`、`aclnnMaximum`、`aclnnInplaceAddcdiv`、`aclnnForeachAddScalar`、`aclnnForeachAddList`、`aclnnForeachLerpScalar`、`aclnnForeachMulScalar`、`aclnnForeachAbs`、`aclnnForeachMaximumList`、`aclnnForeachPowList`、`aclnnForeachDivScalar`、`aclnnForeachMulList`、`aclnnCat`、`aclnnAmax` | `aclnnInplaceAdd`、`aclnnAdd`、`aclnnInplaceLerps`、`aclnnInplaceMul`、`aclnnAbs`、`aclnnMaximum`、`aclnnInplaceAddcdiv` | `aten::_foreach_addcdiv_.ScalarList`（正向，foreach 路径 fallback，不影响功能） |
| 170 | `torch.optim.RMSprop` | ✅ | `aclnnInplaceAdd`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnInplaceAddcmul`、`aclnnSqrt`、`aclnnInplaceAddcdiv`、`aclnnInplaceLerps`、`aclnnAddcmul`、`aclnnInplaceSqrt`、`aclnnForeachAddScalar`、`aclnnForeachAddList`、`aclnnForeachMulScalar`、`aclnnForeachAddcmulScalar`、`aclnnForeachLerpScalar`、`aclnnForeachSqrt`、`aclnnForeachAddcdivScalar` | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnInplaceAddcmul`、`aclnnInplaceLerps`、`aclnnSqrt`、`aclnnInplaceSqrt`、`aclnnInplaceAddcdiv` | - |
