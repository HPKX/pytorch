# torch API 196-200 aclnn 接入分析

## 接口 196：`torch.special.i1e`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_i1e` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_i0e` | 计算接口 | **否** | - | op_plugin 中未找到实现 |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::reciprocal` | 计算接口 | 是 | `aclnnReciprocal` | StructKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

---

## 接口 197：`torch.special.zeta`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_zeta` | 计算接口 | **否** | - | op_plugin 中未找到实现 |
| `aten::special_zeta.self_scalar` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_zeta` | 计算接口 | **否** | - | op_plugin 中未找到实现 |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 198：`torch.subtract`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::subtract.Tensor` | 计算接口 | 是 | `aclnnSub` | subtract 为 sub 的别名，opapi/SubKernelNpuOpApi.cpp |
| `aten::subtract.Scalar` | 计算接口 | 是 | `aclnnSubs` | subtract 为 sub 的别名，opapi/SubKernelNpuOpApi.cpp |
| `aten::subtract.out` | 计算接口 | 是 | `aclnnSub / aclnnSubs` | subtract 为 sub 的别名，opapi/SubKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::sub.Scalar` | 计算接口 | 是 | `aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::sub.out` | 计算接口 | 是 | `aclnnSub / aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

---

## 接口 199：`torch.svd`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::svd` | 计算接口 | 是 | `aclnnSvd` | opapi/StructKernelNpuOpApi.cpp（通过 _linalg_svd_out） |
| `aten::linalg_svd` | 计算接口 | 是 | `aclnnSvd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::linalg_svdvals` | 计算接口 | **否** | - | CIA composite（无 dispatch key），分解为 `_linalg_svd`（aclnnSvd=是）；但 NPU 已直接注册 `svd`/`linalg_svd` → `aclnnSvd`，不走此路径，不阻塞 |
| `aten::_linalg_svd` | 计算接口 | 是 | `aclnnSvd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mH` | view | N/A | - | composite_view |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::matmul` | 计算接口 | **否** | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::transpose.int` | view | N/A | - | view 类接口 |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::diag_embed` | 计算接口 | **否** | - | CEA composite，分解为 zeros + copy_，不阻塞 |
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

---

## 接口 200：`torch.swapdims`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::swapdims` | view | N/A | composite_view 类接口 |
| `aten::transpose.int` | view | N/A | view 类接口 |
| `aten::as_strided` | view | N/A | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::transpose.int` | view | N/A | view 类接口 |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 196 | `torch.special.i1e` | ❌ | - | `aclnnSign`, `aclnnReciprocal`, `aclnnSWhere`, `aclnnAbs`, `aclnnMul`, `aclnnSub` | 正向 **`aten::special_i1e`（否）**，反向 **`aten::special_i0e`（否）** |
| 197 | `torch.special.zeta` | ❌ | - | `aclnnAdds`, `aclnnMul`, `aclnnNeg` | 正向 **`aten::special_zeta`（否）**、**`aten::special_zeta.self_scalar`（否）**，反向 **`aten::special_zeta`（否）** |
| 198 | `torch.subtract` | ✅ | `aclnnSub`, `aclnnSubs` | `aclnnNeg`, `aclnnMuls` | - |
| 199 | `torch.svd` | ✅ | `aclnnSvd` | `aclnnDiv`, `aclnnAdd`, `aclnnSub`, `aclnnMul` | - |
| 200 | `torch.swapdims` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
