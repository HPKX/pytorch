# torch API 41-45 aclnn 接入分析

## 接口 41：`torch.fmax`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::fmax` | 计算接口 | 否 | - | 无 op_plugin 实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ge.Tensor` | 计算接口 | 是 | `aclnnGeTensor` | opapi/GeKernelNpuOpApi.cpp |
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::logical_or_` | 计算接口 | 是 | `aclnnInplaceLogicalOr` | opapi/LogicalOrKernelNpuOpApi.cpp |
| `aten::logical_not_` | 计算接口 | 是 | `aclnnInplaceLogicalNot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::masked_fill_.Scalar` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |

> `aten::fmax` 本身无 aclnn 实现，且不是 composite 分解到已支持的子 op，因此无法在 A5 上支持。

---

## 接口 42：`torch.fmod`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::fmod.Scalar` | 计算接口 | 是 | `aclnnFmodScalar` | opapi/StructKernelNpuOpApi.cpp |
| `aten::fmod.Tensor` | 计算接口 | 是 | `aclnnFmodTensor` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::div.Tensor_mode` | 计算接口 | 是 | `aclnnDivMod` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 43：`torch.gather`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gather` | 计算接口 | 是 | `aclnnGather` | opapi/GatherKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gather_backward` | 计算接口 | 否 | - | 通过 scatter_add 组合实现，无直接 aclnn 调用 |
| `aten::new_zeros` | 计算接口 | 是 | `aclnnInplaceZero` | composite: empty + zero_() |
| `aten::scatter_add_` | 计算接口 | 是 | `aclnnScatterAdd` | opapi/ScatterAddKernelNpuOpApi.cpp |
| `aten::scatter_add` | 计算接口 | 是 | `aclnnScatterAdd` | opapi/ScatterAddKernelNpuOpApi.cpp |
| `aten::_gather_sparse_backward` | 计算接口 | 否 | - | op_plugin 中无实现，fallback |

> `aten::gather_backward` 虽标记为否，但它通过 `scatter_add`（aclnn=是）组合实现；`_gather_sparse_backward` 仅在 `sparse_grad=True` 时使用，默认路径通过 `scatter_add_` 完成，不影响主路径支持。

---

## 接口 44：`torch.gcd`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gcd` | 计算接口 | 是 | `aclnnGcd` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

无（整数专用，不可微）

---

## 接口 45：`torch.ge`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ge.Tensor` | 计算接口 | 是 | `aclnnGeTensor` | opapi/GeKernelNpuOpApi.cpp |
| `aten::ge.Scalar` | 计算接口 | 是 | `aclnnGeScalar` | opapi/GeKernelNpuOpApi.cpp |

### 反向依赖

无（`output_differentiability: [False]`，结果为 bool 张量，不可微）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 41 | `torch.fmax` | ❌ | - | `aclnnGeTensor`, `aclnnInplaceLogicalOr`, `aclnnInplaceLogicalNot`, `aclnnInplaceMaskedFillScalar` | **`aten::fmax`（正向）** |
| 42 | `torch.fmod` | ✅ | `aclnnFmodScalar`, `aclnnFmodTensor` | `aclnnDivMod`, `aclnnMul`, `aclnnNeg` | - |
| 43 | `torch.gather` | ✅ | `aclnnGather` | `aclnnInplaceZero`, `aclnnScatterAdd` | - |
| 44 | `torch.gcd` | ✅ | `aclnnGcd` | - | - |
| 45 | `torch.ge` | ✅ | `aclnnGeTensor`, `aclnnGeScalar` | - | - |
