# torch API 66-70 aclnn 接入分析

## 接口 66：`torch.lcm`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::lcm` | 计算接口 | **否** | - | op-plugin 中未找到实现 |

### 反向依赖

无（整数专用 op，不可微）

---

## 接口 67：`torch.ldexp`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ldexp.Tensor` | 计算接口 | **否** | - | CEA composite（`CompositeExplicitAutograd`），分解为 `self * 2^other` → pow + mul（均有 aclnn），不阻塞 |
| `aten::pow` | 计算接口 | 是 | `aclnnPowTensorTensor` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pow.Scalar` | 计算接口 | 是 | `aclnnPowScalarTensor` | opapi/StructKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | 复数共轭 view |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

---

## 接口 68：`torch.linalg.eigvals`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_eigvals` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::_linalg_eigvals` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_eig` | 计算接口 | **否** | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_solve` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::matmul` | 计算接口 | **否** | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::conj` | view | N/A | - | 复数共轭 view |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |
| `aten::real` | view | N/A | - | composite_view |

---

## 接口 69：`torch.linalg.pinv`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_pinv` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_pinv.atol_rtol_float` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::linalg_pinv.atol_rtol_tensor` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::svd` | 计算接口 | 是 | `aclnnSvd` | opapi/StructKernelNpuOpApi.cpp |
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::matmul` | 计算接口 | **否** | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::linalg_eigh` | 计算接口 | **否** | - | op-plugin 中未找到实现 |
| `aten::amax` | 计算接口 | 是 | `aclnnAmax` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mH` | view | N/A | - | composite_view |
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口 |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

---

## 接口 70：`torch.linalg.vecdot`

**A5 支持结论：✅ 可在 A5 上支持（实数 tensor；复数场景受限，见备注）**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::linalg_vecdot` | 计算接口 | **否** | - | CIA composite，1D 情况走 `vdot`，高维情况走 `conj + mul + sum` |
| `aten::vdot` | 计算接口 | 是 | `aclnnDot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel） |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel） |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::expand` | view | N/A | - | view 类接口 |

> `aten::linalg_vecdot` 在 `native_functions.yaml` 中无独立 backend dispatch，等价于 CIA composite。源码中 1D 路径直接调用 `vdot`，其他路径调用 `x.conj().mul(y).sum(dim)`，子 op 均已接入 aclnn 或为 view。
>
> **复数场景说明**：`torch.linalg.vecdot` 设计上支持复数内积（正向显式调用 `x.conj()`）。`aten::conj` 仅翻转 `conj_bit`（框架层 view），实数 tensor 完全无影响；复数 tensor 场景下若下游 `aclnnMul` 无法原生处理 `conj_bit`，将触发 `aten::_conj_physical`（无 NPU 实现），导致执行失败。详见 API 5 `torch.adjoint` 分析。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 66 | `torch.lcm` | ❌ | - | 无（不可微） | 正向：`aten::lcm` |
| 67 | `torch.ldexp` | ✅ | `aclnnPowTensorTensor`、`aclnnMul` | `aclnnPowScalarTensor`、`aclnnMul`、`aclnnMuls` | - |
| 68 | `torch.linalg.eigvals` | ❌ | - | `aclnnDiv`、`aclnnInplaceCopy` | 正向：`aten::linalg_eigvals`、`aten::_linalg_eigvals`、`aten::linalg_eig`；反向：`aten::linalg_solve` |
| 69 | `torch.linalg.pinv` | ❌ | `aclnnSvd`、`aclnnSWhere`、`aclnnAmax` | `aclnnNeg`、`aclnnAdd`、`aclnnSub` | 正向：`aten::linalg_pinv`（全部重载）、`aten::linalg_eigh` |
| 70 | `torch.linalg.vecdot` | ✅ | `aclnnDot`、`aclnnMul`、`aclnnReduceSum` | `aclnnMul` | - |
