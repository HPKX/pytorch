# torch API 21-25 aclnn 接入分析

## 接口 21：`torch.column_stack`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::column_stack` | 计算接口 | 否 | - | PyTorch composite 函数，无 op_plugin 实现 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::hstack` | 计算接口 | 否 | - | op-plugin 中未找到实现，通常为 composite 接口 |
| `aten::atleast_1d.Sequence` | view | N/A | - | 同 atleast_1d，序列版本 |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |

> `aten::column_stack` 和 `aten::hstack` 均为 PyTorch composite 函数，前向最终分解为 reshape（view）、atleast_1d.Sequence（view）、cat（aclnn=是）等子 op。composite 函数不需要独立 aclnn 实现，实际计算由子 op 完成。

---

## 接口 22：`torch.combinations`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::combinations` | 计算接口 | 否 | - | CIA composite，分解为 meshgrid/arange/masked_select/stack（均已支持） |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::meshgrid` | view | N/A | - | CompositeImplicitAutograd，view + expand |
| `aten::arange` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::full` | 计算接口 | 是 | `aclnnInplaceFillScalar` | FullKernelNpu.cpp |
| `aten::masked_select` | 计算接口 | 是 | `aclnnMaskedSelect` | MaskedSelectKernelNpuOpApi.cpp |
| `aten::stack` | 计算接口 | 是 | `aclnnStack` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::masked_select_backward` | 计算接口 | 否 | - | CIA composite，分解为 zeros_like+expand+masked_scatter_（均有 aclnn） |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::masked_scatter_` | 计算接口 | 是 | `aclnnInplaceMaskedScatter` | MaskedScatterKernelNpuOpApi.cpp |
| `aten::select.int` | view | N/A | - | view 类接口 |

> `aten::combinations` 和 `aten::masked_select_backward` 均为 CIA composite 函数，分解后的所有子算子均已接入 aclnn 或为 view/框架接口。

---

## 接口 23：`torch.cond`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

无（不是传统 ATen 算子，属于 Python 层 HigherOrderOperator）

### 反向依赖

无（正反向依赖取决于 true_fn / false_fn 内部算子）

> `torch.cond` 是 Python 层的 HigherOrderOperator，不依赖传统 ATen 算子，其支持性取决于 true_fn / false_fn 中使用的具体算子。

---

## 接口 24：`torch.conj`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::conj` | view | N/A | - | PyTorch composite 函数，底层调用 _conj |
| `aten::conj_physical` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::_conj` | view | N/A | - | metadata_change 类 view 接口 |
| `aten::_conj_physical` | 计算接口 | 否 | - | 仅在 Sparse/SparseCsr dispatch 中有 native 实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::conj` | view | N/A | - | PyTorch composite 函数 |

> `torch.conj` 是 CIA 分解。对于复数张量，走 `_conj` 路径（view，N/A）。`conj_physical` 和 `_conj_physical` 虽标记为计算接口/否，但它们仅在特定分支（非复数或 Sparse 场景）才会被触发，复数张量主路径为 `_conj`（view）。conj 的 composite 分解使其在 A5 上可通过 view 路径支持。

---

## 接口 25：`torch.copysign`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::copysign.Tensor` | 计算接口 | **否** | - | 无 op_plugin 实现 |
| `aten::copysign.Scalar` | 计算接口 | **否** | - | 无 op_plugin 实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::eq.Scalar` | 计算接口 | 是 | `aclnnEqScalar` | opapi/EqKernelNpuOpApi.cpp |
| `aten::masked_fill_.Scalar` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 21 | `torch.column_stack` | ✅ | `aclnnCat` | - | - |
| 22 | `torch.combinations` | ✅ | `aclnnArange`, `aclnnInplaceFillScalar`, `aclnnMaskedSelect`, `aclnnStack` | `aclnnInplaceZero`, `aclnnInplaceMaskedScatter` | - |
| 23 | `torch.cond` | ✅ | - | - | - |
| 24 | `torch.conj` | ✅ | - | - | - |
| 25 | `torch.copysign` | ❌ | - | `aclnnDiv`, `aclnnEqScalar`, `aclnnInplaceMaskedFillScalar`, `aclnnMul`, `aclnnInplaceZero` | 正向 `aten::copysign.Tensor`（否）、`aten::copysign.Scalar`（否） |
