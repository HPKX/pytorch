# torch API 71-75 aclnn 接入分析

## 接口 71：`torch.logcumsumexp`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::logcumsumexp` | 计算接口 | **否** | - | CEA composite（`CompositeExplicitAutograd`），分解为 `_logcumsumexp`（CPU/CUDA/MPS only），不可穿透 |
| `aten::_logcumsumexp` | 计算接口 | **否** | - | 结构化 kernel（CPU/CUDA/MPS dispatch），NPU 无实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |
| `aten::logcumsumexp` | 计算接口 | **否** | - | CEA composite → `_logcumsumexp`（无 NPU 实现），阻塞 |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::scalar_tensor` | 框架接口 | N/A | - | 标量转张量，框架接口 |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::exp` | 计算接口 | 是 | `aclnnExp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | 复数共轭 view |
| `aten::gt.Scalar` | 计算接口 | 是 | `aclnnGtScalar` | opapi/GtKernelNpuOpApi.cpp |
| `aten::lt.Scalar` | 计算接口 | 是 | `aclnnLtScalar` | opapi/LtKernelNpuOpApi.cpp |

---

## 接口 72：`torch.logdet`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::logdet` | 计算接口 | **否** | - | CIA composite（无 dispatch key），分解为 `linalg_slogdet`（aclnnSlogdet=是），正向不阻塞 |
| `aten::linalg_slogdet` | 计算接口 | 是 | `aclnnSlogdet` | opapi/SlogdetKernelNpuOpApi.cpp |
| `aten::_linalg_slogdet` | 计算接口 | 是 | `aclnnSlogdet` | opapi/SlogdetKernelNpuOpApi.cpp |
| `aten::linalg_lu_factor_ex.out` | 计算接口 | **否** | - | CPU/CUDA 内部 `_linalg_slogdet` 实现依赖；NPU 直接走 `aclnnSlogdet`，不经此路径，正向不阻塞 |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::prod` | 计算接口 | 是 | `aclnnProd` | ProdKernelNpuOpApi.cpp |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::log_` | 计算接口 | 是 | `aclnnInplaceLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::sum.IntList_out` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::imag` | view | N/A | - | composite_view |
| `aten::conj` | view | N/A | - | 复数共轭 view |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::diag_embed` | 计算接口 | **否** | - | CEA composite，分解为 zeros + copy_，不阻塞 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::expand_as` | view | N/A | - | composite_view |
| `aten::mT` | view | N/A | - | composite_view |
| `aten::linalg_lu_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

---

## 接口 73：`torch.logit`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::logit` | 计算接口 | 是 | `aclnnLogit` | opapi/LogitKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::logit_backward` | 计算接口 | 是 | `aclnnLogitGrad` | opapi/LogitBackwardKernelOpApi.cpp |
| `aten::logical_and` | 计算接口 | 是 | `aclnnLogicalAnd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::ge.Scalar` | 计算接口 | 是 | `aclnnGeScalar` | opapi/GeKernelNpuOpApi.cpp |
| `aten::le.Scalar` | 计算接口 | 是 | `aclnnLeScalar` | opapi/LeKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Scalar` | 计算接口 | 是 | `aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::fill_` | 计算接口 | 是 | `aclnnInplaceFillScalar` | opapi/FillKernelNpuOpApi.cpp |

---

## 接口 74：`torch.logspace`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::logspace` | 计算接口 | 是 | `aclnnLogSpace` | LogSpaceKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::logspace.out` | 计算接口 | 是 | `aclnnLogSpace` | LogSpaceKernelNpuOpApi.cpp |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 75：`torch.lu_solve`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::lu_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_lu_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_lu_solve.out` | 计算接口 | **否** | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_lu_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::lu_unpack` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::matmul` | 计算接口 | **否** | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::mT` | view | N/A | - | composite_view |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::linalg_solve_triangular` | 计算接口 | 是 | `aclnnTriangularSolve` | opapi/LinalgSolveTriangularKernelNpuOpApi.cpp |
| `aten::tril` | 计算接口 | 是 | `aclnnTril` | opapi/StructKernelNpuOpApi.cpp |
| `aten::triu` | 计算接口 | 是 | `aclnnTriu` | opapi/StructKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 71 | `torch.logcumsumexp` | ❌ | - | `aclnnFlip`、`aclnnAbs`、`aclnnLog`、`aclnnSWhere`、`aclnnSub`、`aclnnAdd`、`aclnnExp`、`aclnnGtScalar`、`aclnnLtScalar` | 正向：`aten::logcumsumexp`、`aten::_logcumsumexp`；反向：`aten::logcumsumexp` |
| 72 | `torch.logdet` | ❌ | `aclnnSlogdet`、`aclnnSign`、`aclnnProd`、`aclnnMul`、`aclnnAbs`、`aclnnInplaceLog`、`aclnnReduceSum`、`aclnnSWhere`、`aclnnLog`、`aclnnAdd` | `aclnnMul`、`aclnnSub` | 正向：`aten::logdet`、`aten::linalg_lu_factor_ex.out`；反向：`aten::linalg_lu_solve`、`aten::linalg_solve` |
| 73 | `torch.logit` | ✅ | `aclnnLogit` | `aclnnLogitGrad`、`aclnnLogicalAnd`、`aclnnGeScalar`、`aclnnLeScalar`、`aclnnSWhere`、`aclnnDiv`、`aclnnMul`、`aclnnSubs`、`aclnnInplaceZero`、`aclnnInplaceFillScalar` | - |
| 74 | `torch.logspace` | ✅ | `aclnnLogSpace` | 无（工厂函数） | - |
| 75 | `torch.lu_solve` | ❌ | - | `aclnnNeg`、`aclnnTriangularSolve`、`aclnnTril`、`aclnnTriu`、`aclnnAdd` | 正向：`aten::lu_solve`、`aten::linalg_lu_solve`（全部重载）；反向：`aten::linalg_lu_solve`、`aten::lu_unpack` |
