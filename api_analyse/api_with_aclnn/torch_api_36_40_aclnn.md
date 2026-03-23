# torch API 36-40 aclnn 接入分析

## 接口 36：`torch.distribution.uniform.Uniform`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::rand` | 计算接口 | 是 | `aclnnInplaceRandom` | RandomKernelNpuOpApi.cpp |
| `aten::uniform_` | 计算接口 | 是 | `aclnnInplaceUniform` | opapi/UniformKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::le.Tensor` | 计算接口 | 是 | `aclnnLeTensor` | opapi/LeKernelNpuOpApi.cpp |
| `aten::gt.Tensor` | 计算接口 | 是 | `aclnnGtTensor` | opapi/GtKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::clamp` | 计算接口 | 是 | `aclnnClamp` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |

---

## 接口 37：`torch.dsplit`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::dsplit.int` | view | N/A | - | CompositeImplicitAutograd，tensor_split → narrow（view） |
| `aten::dsplit.array` | view | N/A | - | 同上 |
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

## 接口 38：`torch.dstack`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::dstack` | 计算接口 | 否 | - | PyTorch composite 函数，无 op_plugin 实现 |
| `aten::dstack.out` | 计算接口 | 否 | - | PyTorch composite 函数，无 op_plugin 实现 |
| `aten::atleast_3d` | view | N/A | - | CompositeImplicitAutograd，reshape 插入维度 |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::cat.out` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::real` | view | N/A | - | composite_view |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::reshape` | view | N/A | - | composite_view |

> `aten::dstack` 和 `aten::dstack.out` 虽标记为计算接口/否，但它们是 PyTorch composite 函数，前向分解为 atleast_3d（view）+ cat（aclnn=是），实际计算由 aclnn 支持的子 op 完成。

---

## 接口 39：`torch.fliplr`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::fliplr` | 计算接口 | 否 | - | PyTorch composite 函数，无 op_plugin 实现 |
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |

> `aten::fliplr` 虽标记为计算接口/否，但它是 PyTorch composite 函数，前向直接分解为 `flip`（aclnn=是）。

---

## 接口 40：`torch.flipud`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::flipud` | 计算接口 | 否 | - | PyTorch composite 函数，无 op_plugin 实现 |
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |

> `aten::flipud` 虽标记为计算接口/否，但它是 PyTorch composite 函数，前向直接分解为 `flip`（aclnn=是）。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 36 | `torch.distribution.uniform.Uniform` | ✅ | `aclnnInplaceRandom`, `aclnnInplaceUniform`, `aclnnMul`, `aclnnSub`, `aclnnAdd`, `aclnnLeTensor`, `aclnnGtTensor`, `aclnnLog`, `aclnnDiv`, `aclnnClamp` | `aclnnMul`, `aclnnNeg`, `aclnnDiv` | - |
| 37 | `torch.dsplit` | ✅ | - | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 38 | `torch.dstack` | ✅ | `aclnnCat` | `aclnnInplaceZero` | - |
| 39 | `torch.fliplr` | ✅ | `aclnnFlip` | `aclnnFlip` | - |
| 40 | `torch.flipud` | ✅ | `aclnnFlip` | `aclnnFlip` | - |
