# torch API 76-80 aclnn 接入分析

## 接口 76：`torch.lu_unpack`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::lu_unpack` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::triu.out` | 计算接口 | 是 | `aclnnTriu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::tril.out` | 计算接口 | 是 | `aclnnTril` | opapi/StructKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::fill_.Scalar` | 计算接口 | 是 | `aclnnInplaceFillScalar` | opapi/FillKernelNpuOpApi.cpp |
| `aten::arange` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::zero_` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::scatter_.value` | 计算接口 | 是 | `aclnnScatterValue` | opapi/ScatterKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tril` | 计算接口 | 是 | `aclnnTril` | opapi/StructKernelNpuOpApi.cpp |
| `aten::triu` | 计算接口 | 是 | `aclnnTriu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |

---

## 接口 77：`torch.moveaxis`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::moveaxis` | view | N/A | - | composite_view，等价于 movedim |
| `aten::movedim` | view | N/A | - | composite_view |
| `aten::permute` | view | N/A | - | view 操作 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::permute` | view | N/A | - | view 操作 |

---

## 接口 78：`torch.movedim`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::movedim` | view | N/A | - | composite_view |
| `aten::permute` | view | N/A | - | view 操作 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::permute` | view | N/A | - | view 操作 |

---

## 接口 79：`torch.msort`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::msort` | 计算接口 | **否** | - | PyTorch composite 接口，委托给 sort；NPU 不直接注册该接口，走 composite 分解 |
| `aten::sort` | 计算接口 | 是 | `aclnnSort` | opapi/SortKernelNpuOpApi.cpp |
| `aten::sort.stable` | 计算接口 | 是 | `aclnnSort` | opapi/SortKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::scatter_` | 计算接口 | 是 | `aclnnInplaceScatter` | opapi/ScatterKernelNpuOpApi.cpp |
| `aten::scatter` | 计算接口 | 是 | `aclnnScatter` | opapi/ScatterKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |

> **备注**：`aten::msort` 本身标记为计算接口=否，但它是 PyTorch composite 函数，直接分解为 `sort(self, 0, false)`，实际计算由 `aten::sort`（已接入 aclnn）承担，因此不影响 A5 支持判定。

---

## 接口 80：`torch.multiply`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multiply.Tensor` | 计算接口 | 是 | `aclnnMul` | mul 的别名，MulKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::multiply.Scalar` | 计算接口 | 是 | `aclnnMuls` | mul 的别名，MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 76 | `torch.lu_unpack` | ❌ | `aclnnTriu`、`aclnnTril`、`aclnnInplaceFillScalar`、`aclnnArange`、`aclnnInplaceZero`、`aclnnScatterValue` | `aclnnTril`、`aclnnTriu`、`aclnnAdd`、`aclnnCat`、`aclnnInplaceZero` | 正向：`aten::lu_unpack` |
| 77 | `torch.moveaxis` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 78 | `torch.movedim` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 79 | `torch.msort` | ✅ | `aclnnSort` | `aclnnInplaceZero`、`aclnnInplaceScatter`、`aclnnScatter` | - |
| 80 | `torch.multiply` | ✅ | `aclnnMul`、`aclnnMuls` | `aclnnMul`、`aclnnMuls` | - |
