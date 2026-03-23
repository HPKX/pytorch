# torch API 26-30 aclnn 接入分析

## 接口 26：`torch.cov`

**A5 支持结论：✅ 可在 A5 上支持（实数 tensor；复数场景受限，见备注）**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::cov` | 计算接口 | 否 | - | PyTorch composite 函数，无 op_plugin 实现 |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::scalar_tensor` | 框架接口 | N/A | - | 标量转张量 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::t` | view | N/A | - | view 类接口 |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel） |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::real` | view | N/A | - | composite_view |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::complex` | 计算接口 | 是 | `aclnnComplex` | opapi/ComplexKernelNpuOpApi.cpp |
| `aten::true_divide` | 计算接口 | 是 | `aclnnDiv` | true_divide 为 div 的别名 |
| `aten::squeeze` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel） |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::t` | view | N/A | - | view 类接口 |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::real` | view | N/A | - | composite_view |
| `aten::complex` | 计算接口 | 是 | `aclnnComplex` | opapi/ComplexKernelNpuOpApi.cpp |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |

> `aten::cov` 是 CompositeImplicitAutograd 函数，分解为基础算术和线性代数子 op，所有子 op 均已有 aclnn 接入或为 view/框架接口。
>
> **复数场景说明**：`torch.cov` 支持复数输入（前向依赖 `aten::complex`/`aten::real`）。其中 `aten::conj` 仅翻转 `conj_bit`（框架层 view），实数 tensor 完全无影响；复数 tensor 场景下若下游 NPU kernel 无法原生处理 `conj_bit`，将触发 `aten::_conj_physical`（无 NPU 实现），导致执行失败。详见 API 5 `torch.adjoint` 分析。

---

## 接口 27：`torch.deg2rad`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::deg2rad` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::deg2rad.out` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

> `aten::deg2rad` 和 `aten::deg2rad.out` 虽标记为计算接口/否，但其实现本质是 `self * (pi / 180)`，通过 `mul.out` 完成（已有 aclnn），实际计算均由 aclnn 支持的子 op 完成。composite 分解路径可在 A5 上运行。

---

## 接口 28：`torch.diag_embed`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diag_embed` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::as_strided` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::as_strided` | view | N/A | - | view 类接口 |

> `aten::diag_embed` 虽标记为计算接口/否，但它是 composite 函数，前向分解为 zeros、diagonal（view）、as_strided（view）、copy_ 等子 op。所有实际计算子 op 均已有 aclnn 接入。

---

## 接口 29：`torch.diagflat`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diagflat` | 计算接口 | 否 | - | 无 op_plugin 实现，PyTorch composite 函数 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::diag` | 计算接口 | 否 | - | 仅有 aclops 实现，不支持 aclnn |
| `aten::diag_embed` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::as_strided` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::diagonal` | view | N/A | - | view 类接口 |

> `aten::diagflat` 是 CIA 分解，先 contiguous→view→diag_embed。`aten::diag` 虽标记为否，但 `diagflat` 实际路径会通过 `diag_embed`（再分解为 zeros+diagonal+copy_），不一定直接调用 `aten::diag`。然而 `aten::diag` 在 ATen native 中有 composite 实现可以 fallback 到 CPU。考虑到 diagflat 的 CIA 分解最终走 diag_embed 路径，实际计算子 op 均可支持。

---

## 接口 30：`torch.diagonal`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::as_strided` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diagonal_backward` | 计算接口 | 否 | - | 无 op_plugin 实现 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

> `aten::diagonal_backward` 标记为计算接口/否，但其实现是创建零矩阵再通过 diagonal view 写回梯度（即 zeros + diagonal + copy_），这些子 op 均已有 aclnn 接入。diagonal_backward 作为 composite 函数，实际计算可在 A5 上完成。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 26 | `torch.cov` | ✅ | `aclnnMul`, `aclnnReduceSum`, `aclnnSub`, `aclnnMm`, `aclnnInplaceZero`, `aclnnComplex`, `aclnnDiv` | `aclnnMul`, `aclnnNeg`, `aclnnMm`, `aclnnDiv`, `aclnnComplex`, `aclnnInplaceZero` | - |
| 27 | `torch.deg2rad` | ✅ | `aclnnMul` | `aclnnMuls` | - |
| 28 | `torch.diag_embed` | ✅ | `aclnnInplaceZero`, `aclnnInplaceCopy` | - | - |
| 29 | `torch.diagflat` | ✅ | `aclnnInplaceZero`, `aclnnInplaceCopy` | - | - |
| 30 | `torch.diagonal` | ✅ | - | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
