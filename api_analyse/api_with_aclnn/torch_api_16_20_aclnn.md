# torch API 16-20 aclnn 接入分析

## 接口 16：`torch.block_diag`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::block_diag` | 计算接口 | 否 | - | 无 op_plugin 实现，PyTorch composite 函数 |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::real` | view | N/A | - | composite_view |

> `aten::block_diag` 本身标记为计算接口/否，但它是 PyTorch composite 函数，前向实际分解为 expand（view）、zeros、slice.Tensor（view）、copy_ 等子 op，这些子 op 均已有 aclnn 接入或为 view/框架接口。composite 函数不需要独立 aclnn 实现。

---

## 接口 17：`torch.bucketize`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::bucketize.Tensor` | 计算接口 | 是 | `aclnnSearchSorted` | opapi/BucketizeKernelNpuOpApi.cpp 转调 searchsorted |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |

### 反向依赖

无（不可微）

---

## 接口 18：`torch.cholesky`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::cholesky` | 计算接口 | 是 | `aclnnLinalgCholesky` | opapi/LinalgCholeskyKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::_linalg_check_errors` | 计算接口 | 否 | - | CEA composite，仅调用 any()+item()，均已支持 |
| `aten::tril_` | 计算接口 | 是 | `aclnnInplaceTril` | opapi/StructKernelNpuOpApi.cpp |
| `aten::triu_` | 计算接口 | 是 | `aclnnInplaceTriu` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mH` | view | N/A | - | composite_view |
| `aten::matmul` | 计算接口 | 否 | - | CIA composite，分解为 mm/bmm/dot 等（均有 aclnn） |
| `aten::tril` | 计算接口 | 是 | `aclnnTril` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::linalg_solve_triangular` | 计算接口 | 是 | `aclnnTriangularSolve` | opapi/LinalgSolveTriangularKernelNpuOpApi.cpp |

> `aten::_linalg_check_errors` 是 CEA composite，仅调用 `any()` + `item()`（均已支持），不涉及实际计算。`aten::matmul` 是 CIA composite，分解为 mm/bmm/dot 等已有 aclnn 的子 op。两者均通过 composite 分解机制在 A5 上可正常执行。

---

## 接口 19：`torch.cholesky_solve`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::cholesky_solve` | 计算接口 | **否** | - | 无 op_plugin 实现 |
| `aten::_cholesky_solve_helper` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现，fallback |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::cholesky_solve` | 计算接口 | **否** | - | 无 op_plugin 实现 |
| `aten::matmul` | 计算接口 | **否** | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 20：`torch.clip`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::clip` | 计算接口 | 是 | `aclnnClamp` | clip 是 clamp 的别名 |
| `aten::clip.Tensor` | 计算接口 | 是 | `aclnnClampTensor` | clip 是 clamp 的别名 |
| `aten::clamp` | 计算接口 | 是 | `aclnnClamp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::clamp.Tensor` | 计算接口 | 是 | `aclnnClampTensor` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::scalar_tensor` | 框架接口 | N/A | - | 标量转张量 |
| `aten::ge.Scalar` | 计算接口 | 是 | `aclnnGeScalar` | opapi/GeKernelNpuOpApi.cpp |
| `aten::le.Scalar` | 计算接口 | 是 | `aclnnLeScalar` | opapi/LeKernelNpuOpApi.cpp |
| `aten::ge.Tensor` | 计算接口 | 是 | `aclnnGeTensor` | opapi/GeKernelNpuOpApi.cpp |
| `aten::le.Tensor` | 计算接口 | 是 | `aclnnLeTensor` | opapi/LeKernelNpuOpApi.cpp |
| `aten::logical_and_` | 计算接口 | 是 | `aclnnInplaceLogicalAnd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 16 | `torch.block_diag` | ✅ | `aclnnInplaceZero`, `aclnnInplaceCopy` | `aclnnInplaceZero` | - |
| 17 | `torch.bucketize` | ✅ | `aclnnSearchSorted` | - | - |
| 18 | `torch.cholesky` | ✅ | `aclnnLinalgCholesky`, `aclnnInplaceTril`, `aclnnInplaceTriu` | `aclnnTril`, `aclnnMuls`, `aclnnAdd`, `aclnnTriangularSolve` | - |
| 19 | `torch.cholesky_solve` | ❌ | - | `aclnnAdd`, `aclnnNeg` | 正向 `aten::cholesky_solve`（否）、`aten::_cholesky_solve_helper`（否），反向 `aten::cholesky_solve`（否） |
| 20 | `torch.clip` | ✅ | `aclnnClamp`, `aclnnClampTensor` | `aclnnGeScalar`, `aclnnLeScalar`, `aclnnGeTensor`, `aclnnLeTensor`, `aclnnInplaceLogicalAnd`, `aclnnSWhere` | - |
