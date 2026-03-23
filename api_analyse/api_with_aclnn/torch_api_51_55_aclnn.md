# torch API 51-55 aclnn 接入分析

## 接口 51：`torch.hsplit`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hsplit.int` | view | N/A | - | CompositeImplicitAutograd，tensor_split → narrow（view） |
| `aten::hsplit.array` | view | N/A | - | 同上 |
| `aten::tensor_split.sections` | view | N/A | - | composite_view 类接口 |
| `aten::tensor_split.indices` | view | N/A | - | composite_view 类接口 |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

---

## 接口 52：`torch.hstack`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hstack` | 计算接口 | 否 | - | op-plugin 中未找到实现，composite 接口 |
| `aten::hstack.out` | 计算接口 | 否 | - | op-plugin 中未找到实现，composite 接口 |
| `aten::atleast_1d.Sequence` | view | N/A | - | 同 atleast_1d，对序列逐一执行，均为 alias 或 view |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::cat.out` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::real` | view | N/A | - | composite_view |

> `aten::hstack` 和 `aten::hstack.out` 虽标记为计算接口/否，但它们是 PyTorch composite 函数，前向分解为 atleast_1d（view）+ cat（aclnn=是），实际计算由 aclnn 支持的子 op 完成。

---

## 接口 53：`torch.hypot`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hypot` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |

---

## 接口 54：`torch.i0`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::i0` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::special_i1` | 计算接口 | 否 | - | op_plugin 中未找到实现 |

---

## 接口 55：`torch.igamma`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::igamma` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sub.Scalar` | 计算接口 | 是 | `aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::lgamma` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::exp` | 计算接口 | 是 | `aclnnExp` | opapi/StructKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 51 | `torch.hsplit` | ✅ | - | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 52 | `torch.hstack` | ✅ | `aclnnCat` | `aclnnInplaceZero` | - |
| 53 | `torch.hypot` | ❌ | - | `aclnnMul`, `aclnnDiv` | **`aten::hypot`（正向）** |
| 54 | `torch.i0` | ❌ | - | `aclnnMul` | **`aten::i0`（正向）**, **`aten::special_i1`（反向）** |
| 55 | `torch.igamma` | ❌ | - | `aclnnSubs`, `aclnnLog`, `aclnnMul`, `aclnnSub`, `aclnnExp` | **`aten::igamma`（正向）**, **`aten::lgamma`（反向）** |
