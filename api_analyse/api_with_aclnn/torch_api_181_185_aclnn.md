# torch API 181-185 aclnn 接入分析

## 接口 181：`torch.rad2deg`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rad2deg` | 计算接口 | **否** | - | CEA composite，源码直接调用 `mul_out(self, 180/pi)` |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::rad2deg.out` | 计算接口 | **否** | - | CEA composite，实际由 `mul.out` 承担计算 |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul / aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

> `aten::rad2deg` / `aten::rad2deg.out` 在 `native_functions.yaml` 中均为 `CompositeExplicitAutograd`。源码直接分解为常数乘法，前向由 `mul.out`、反向由 `mul.Scalar` 完成，子 op 均已支持。

---

## 接口 182：`torch.range`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::range.step` | 计算接口 | 是 | `aclnnRange` | RangeKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::range.out` | 计算接口 | 是 | `aclnnRange` | RangeKernelNpuOpApi.cpp |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 183：`torch.renorm`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::renorm` | 计算接口 | 是 | `aclnnRenorm` | RenormKernelNpuOpApi.cpp |
| `aten::renorm.out` | 计算接口 | 是 | `aclnnRenorm` | RenormKernelNpuOpApi.cpp |
| `aten::linalg_vector_norm` | 计算接口 | 是 | `aclnnLinalgVectorNorm` | opapi/LinalgNormKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul / aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_vector_norm` | 计算接口 | 是 | `aclnnLinalgVectorNorm` | opapi/LinalgNormKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::real` | view | N/A | - | composite_view |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::masked_fill_` | 计算接口 | 是 | `aclnnInplaceMaskedFillTensor / aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::eq.Scalar` | 计算接口 | 是 | `aclnnEqScalar` | opapi/EqKernelNpuOpApi.cpp |
| `aten::eq.Tensor` | 计算接口 | 是 | `aclnnEqTensor` | opapi/EqKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::logical_or` | 计算接口 | 是 | `aclnnLogicalOr` | opapi/LogicalOrKernelNpuOpApi.cpp |
| `aten::reciprocal` | 计算接口 | 是 | `aclnnReciprocal` | StructKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::gt.Scalar` | 计算接口 | 是 | `aclnnGtScalar` | opapi/GtKernelNpuOpApi.cpp |

> `aten::isnan` 在 NPU PTA 层实现为 `self != self` → `ne.Tensor`（aclnn=是），不阻塞 A5 支持。正反向所有其他接口均已接入 aclnn 或为 view/框架接口。

---

## 接口 184：`torch.rot90`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rot90` | 计算接口 | 否 | - | CEA composite，分解为 flip+transpose_+clone（均已支持） |
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |
| `aten::transpose_` | view | N/A | - | inplace_view 类接口 |
| `aten::clone` | 计算接口 | 是 | `aclnnInplaceCopy` | CloneKernelOpApi.cpp → copy_ → aclnnInplaceCopy |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rot90` | 计算接口 | 否 | - | CEA composite，backward 递归调用 rot90(-k)，同样走 composite 分解 |

> `aten::rot90` 是 CompositeExplicitAutograd，分解为 flip（aclnn=是）+ transpose_（view）+ clone（aclnn=是）。backward 递归调用 `rot90(-k, dims)` 也同样走 composite 分解路径，所有子 op 均已支持。

---

## 接口 185：`torch.row_stack`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::row_stack` | 计算接口 | **否** | - | op_plugin 中未找到实现，vstack 的别名 |
| `aten::vstack` | 计算接口 | **否** | - | op_plugin 中未找到实现，cat(dim=0) 的封装 |
| `aten::atleast_2d.Sequence` | view | N/A | - | composite_view |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::reshape` | view | N/A | - | composite_view |

> `aten::row_stack` 和 `aten::vstack` 是 CIA composite 函数，前向实际分解为 atleast_2d（view）、unsqueeze（view）、reshape（view）、cat（aclnn=是）。composite 函数不需要独立 aclnn 实现，实际计算由 cat 承担。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 181 | `torch.rad2deg` | ✅ | `aclnnMul` | `aclnnMuls` | - |
| 182 | `torch.range` | ✅ | `aclnnRange` | - | - |
| 183 | `torch.renorm` | ✅ | `aclnnRenorm`, `aclnnLinalgVectorNorm`, `aclnnMul` | `aclnnLinalgVectorNorm`, `aclnnMul`, `aclnnReduceSum`, `aclnnSign`, `aclnnDiv`, `aclnnInplaceMaskedFillScalar`, `aclnnEqScalar`, `aclnnAbs`, `aclnnLogicalOr`, `aclnnReciprocal`, `aclnnAdds`, `aclnnMuls`, `aclnnSub`, `aclnnSWhere`, `aclnnGtScalar` | - |
| 184 | `torch.rot90` | ✅ | `aclnnFlip`, `aclnnInplaceCopy` | `aclnnFlip`, `aclnnInplaceCopy` | - |
| 185 | `torch.row_stack` | ✅ | `aclnnCat` | 全为 view，无需 aclnn | - |
