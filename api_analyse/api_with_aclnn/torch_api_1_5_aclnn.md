# torch API 1-5 aclnn 接入分析

## 接口 1：`torch.absolute`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::absolute` | 计算接口 | 是 | `aclnnAbs` | `absolute` 是 `abs` 的别名，底层复用 `aclnnAbs` |
| `aten::absolute.out` | 计算接口 | 是 | `aclnnAbs` | `absolute.out` 是 `abs.out` 的别名，底层复用 `aclnnAbs` |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::abs.out` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | `grad * self.sgn()` |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |

---

## 接口 2：`torch.addcdiv`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::addcdiv` | 计算接口 | 是 | `aclnnAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |
| `aten::addcdiv.out` | 计算接口 | 是 | `aclnnAddcdiv` | opapi/AddcdivKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | `grad * (value / tensor2).conj()` 等 |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | `value * tensor1` |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | `value / tensor2` |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | tensor2 梯度取负 |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

---

## 接口 3：`torch.addcmul`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::addcmul` | 计算接口 | 是 | `aclnnAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |
| `aten::addcmul.out` | 计算接口 | 是 | `aclnnAddcmul` | opapi/AddcmulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | `grad * (tensor2 * value).conj()` 等 |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | `tensor2 * value`、`tensor1 * value` |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

---

## 接口 4：`torch.addr`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::addr` | 计算接口 | 是 | `aclnnAddr` | opapi/StructKernelNpuOpApi.cpp；NPU 主路径 |
| `aten::addr.out` | 计算接口 | 是 | `aclnnAddr` | out= 重载 |
| `aten::outer` | 计算接口 | 否 | - | 仅 CEA fallback 路径，NPU 已有 native addr 实现，不触发 |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | 仅 CEA fallback 路径，NPU 不触发 |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | 仅 CEA fallback 路径，NPU 不触发 |
| `aten::to` | 计算接口 | 是 | `aclnnInplaceCopy` | 仅 CEA fallback 路径，NPU 不触发 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | `maybe_multiply`（beta/alpha ≠ 1 时） |
| `aten::mv` | 计算接口 | 是 | `aclnnMv` | `grad.mv(vec2.conj())`、`grad.t().mv(vec1.conj())` |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |
| `aten::t` | view | N/A | - | 矩阵转置 view |

---

## 接口 5：`torch.adjoint`

**A5 支持结论：✅ 可在 A5 上支持（实数 tensor；复数场景受限，见备注）**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::adjoint` | view | N/A | CIA composite，分解为 transpose + conj |
| `aten::transpose.int` | view | N/A | 转置最后两维，纯 stride/shape 改写 |
| `aten::as_strided` | view | N/A | transpose 内部分解 |
| `aten::conj` | view | N/A | CIA composite → `_conj`（仅翻 conj_bit，不调用 backend kernel） |
| `aten::_conj` | view | N/A | metadata_change 类 view，置位 conj_bit |
| `aten::alias` | view | N/A | `_conj` 内部返回别名 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::transpose` | view | N/A | `transpose.int` 的反向 |
| `aten::conj` | view | N/A | `_conj` 的反向（仅复数路径） |
| `as_strided_backward` | helper | N/A | 内部使用 `aten::zeros`（框架接口）+ `aten::as_strided`（view） |

> **复数场景说明**：`aten::conj` 本身是框架层操作（仅在 tensor metadata 中翻转 `conj_bit`，不调用任何 backend kernel），因此被分类为 view。但当下游 NPU kernel 接收到带 `conj_bit` 的 tensor 且自身无法处理该标志时，PyTorch 框架会调用 `aten::_conj_physical` 来物化共轭计算；而 `_conj_physical` 在 NPU 上无 aclnn 实现（分类为 计算接口/否）。对于**实数 tensor**，`conj_bit` 不会被置位，`adjoint` 等价于纯 transpose，完全支持。对于**复数 tensor**，是否可运行取决于下游各 NPU kernel 能否原生处理 `conj_bit`；若不能，则会在执行阶段报错。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 1 | `torch.absolute` | ✅ | `aclnnAbs` | `aclnnMul`、`aclnnSign` | - |
| 2 | `torch.addcdiv` | ✅ | `aclnnAddcdiv` | `aclnnMul`、`aclnnMuls`、`aclnnDiv`、`aclnnNeg` | - |
| 3 | `torch.addcmul` | ✅ | `aclnnAddcmul` | `aclnnMul`、`aclnnMuls` | - |
| 4 | `torch.addr` | ✅ | `aclnnAddr` | `aclnnMuls`、`aclnnMv` | - |
| 5 | `torch.adjoint` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
