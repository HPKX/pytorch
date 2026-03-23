# torch API 186-190 aclnn 接入分析

## 接口 186：`torch.select_scatter`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::select_scatter` | 计算接口 | 否 | - | CEA-NonFunctional composite，分解为 clone+select(view)+copy_ |
| `aten::select.int` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::select_scatter` | 计算接口 | 否 | - | CEA-NonFunctional composite，同正向分解路径 |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::select.int` | view | N/A | - | view 类接口 |

> `aten::select_scatter` 是 CompositeExplicitAutogradNonFunctional，分解为 clone（aclnn=是）+ select（view）+ copy_（aclnn=是），所有子算子均已支持。

---

## 接口 187：`torch.sgn`

**A5 支持结论：✅ 可在 A5 上支持（实数 tensor；复数场景受限，见备注）**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数梯度路径 |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::masked_fill_.Scalar` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::eq.Scalar` | 计算接口 | 是 | `aclnnEqScalar` | opapi/EqKernelNpuOpApi.cpp |

> **复数场景说明**：`torch.sgn` 对复数定义为 z/|z|，其反向梯度公式含 `conj()`。`aten::conj` 仅翻转 `conj_bit`（框架层 view），实数 tensor 完全无影响；复数 tensor 场景下若下游 NPU kernel 无法原生处理 `conj_bit`，将触发 `aten::_conj_physical`（无 NPU 实现），导致执行失败。详见 API 5 `torch.adjoint` 分析。

---

## 接口 188：`torch.signbit`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::signbit` | 计算接口 | 是 | `aclnnSignbit` | opapi/StructKernelNpuOpApi.cpp |
| `aten::fill_.Scalar` | 计算接口 | 是 | `aclnnInplaceFillScalar` | opapi/FillKernelNpuOpApi.cpp |

### 反向依赖

无（布尔输出，不可微）

---

## 接口 189：`torch.slogdet`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::slogdet` | 计算接口 | 是 | `aclnnSlogdet` | opapi/SlogdetKernelNpuOpApi.cpp |
| `aten::linalg_slogdet` | 计算接口 | 是 | `aclnnSlogdet` | opapi/SlogdetKernelNpuOpApi.cpp |
| `aten::_linalg_slogdet` | 计算接口 | 是 | `aclnnSvd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::linalg_lu_factor_ex.out` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::prod` | 计算接口 | 是 | `aclnnProd / aclnnProdDim` | ProdKernelNpuOpApi.cpp |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul / aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::log_` | 计算接口 | 是 | `aclnnInplaceLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::sum.IntList_out` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::imag` | view | N/A | - | composite_view |
| `aten::conj` | view | N/A | - | composite → `_conj`（view） |
| `aten::diag_embed` | 计算接口 | **否** | - | CEA composite，分解为 zeros + copy_，不阻塞 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::expand_as` | view | N/A | - | composite_view |
| `aten::mT` | view | N/A | - | composite_view |
| `aten::linalg_lu_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::mH` | view | N/A | - | composite_view |

---

## 接口 190：`torch.special.bessel_j0`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_bessel_j0` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

无（不可微）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 186 | `torch.select_scatter` | ✅ | `aclnnInplaceCopy` | `aclnnInplaceZero` | - |
| 187 | `torch.sgn` | ✅ | `aclnnSign` | `aclnnAbs`, `aclnnMul`, `aclnnDiv`, `aclnnInplaceMaskedFillScalar`, `aclnnEqScalar` | - |
| 188 | `torch.signbit` | ✅ | `aclnnSignbit`, `aclnnInplaceFillScalar` | - | - |
| 189 | `torch.slogdet` | ❌ | `aclnnSlogdet`, `aclnnSvd`, `aclnnSign`, `aclnnProd`, `aclnnMul`, `aclnnAbs`, `aclnnInplaceLog`, `aclnnReduceSum` | - | 正向 **`aten::linalg_lu_factor_ex.out`（否）**，反向 **`aten::linalg_lu_solve`（否）**、**`aten::linalg_solve`（否）** |
| 190 | `torch.special.bessel_j0` | ❌ | - | - | 正向 **`aten::special_bessel_j0`（否）** |
