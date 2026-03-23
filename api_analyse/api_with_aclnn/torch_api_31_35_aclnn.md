# torch API 31-35 aclnn 接入分析

## 接口 31：`torch.diagonal_scatter`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diagonal_scatter` | 计算接口 | 否 | - | CEA-NonFunctional composite，分解为 clone+diagonal(view)+copy_ |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::diagonal_scatter` | 计算接口 | 否 | - | CEA-NonFunctional composite，同正向分解路径 |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |

> `aten::diagonal_scatter` 是 CompositeExplicitAutogradNonFunctional，分解为 clone（aclnn=是）+ diagonal（view）+ copy_（aclnn=是），所有子算子均已支持。

---

## 接口 32：`torch.digamma`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::digamma` | 计算接口 | **否** | - | 无 op_plugin 实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::polygamma` | 计算接口 | **否** | - | op_plugin 中无实现 |

---

## 接口 33：`torch.dist`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::dist` | 计算接口 | 否 | - | CEA composite，分解为 sub+norm（均有 aclnn） |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::norm` | 计算接口 | 是 | `aclnnNorm` | NormKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::masked_fill_` | 计算接口 | 是 | `aclnnInplaceMaskedFillScalar` | MaskedFillKernelNpuOpApi.cpp |
| `aten::eq.Scalar` | 计算接口 | 是 | `aclnnEqScalar` | opapi/EqKernelNpuOpApi.cpp |
| `aten::eq.Tensor` | 计算接口 | 是 | `aclnnEqTensor` | opapi/EqKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::logical_or` | 计算接口 | 是 | `aclnnLogicalOr` | opapi/LogicalOrKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

---

## 接口 34：`torch.distribution.gamma.Gamma`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::broadcast_tensors` | view | N/A | - | CompositeImplicitAutograd，expand |
| `aten::expand` | view | N/A | - | view 类接口 |
| `aten::_standard_gamma` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现 |
| `aten::div` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::xlogy` | 计算接口 | 是 | `aclnnXLogYTensor` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::lgamma` | 计算接口 | **否** | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::_standard_gamma_grad` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现 |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |

---

## 接口 35：`torch.distribution.laplace.Laplace`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::broadcast_tensors` | view | N/A | - | CompositeImplicitAutograd，expand |
| `aten::uniform_` | 计算接口 | 是 | `aclnnInplaceUniform` | opapi/UniformKernelNpuOpApi.cpp |
| `aten::rand` | 计算接口 | 是 | `aclnnInplaceRandom` | RandomKernelNpuOpApi.cpp |
| `aten::sign` | 计算接口 | 是 | `aclnnSign` | opapi/StructKernelNpuOpApi.cpp |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::clamp` | 计算接口 | 是 | `aclnnClamp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::log1p` | 计算接口 | 是 | `aclnnLog1p` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::div` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 31 | `torch.diagonal_scatter` | ✅ | `aclnnInplaceCopy` | `aclnnInplaceZero` | - |
| 32 | `torch.digamma` | ❌ | - | `aclnnMul` | 正向 `aten::digamma`（否），反向 `aten::polygamma`（否） |
| 33 | `torch.dist` | ✅ | `aclnnSub`, `aclnnNorm` | `aclnnSub`, `aclnnNeg`, `aclnnSign`, `aclnnMul`, `aclnnDiv`, `aclnnInplaceMaskedFillScalar`, `aclnnEqScalar`, `aclnnEqTensor`, `aclnnAbs`, `aclnnLogicalOr`, `aclnnReduceSum` | - |
| 34 | `torch.distribution.gamma.Gamma` | ❌ | `aclnnDiv`, `aclnnXLogYTensor`, `aclnnSub`, `aclnnMul` | `aclnnMul`, `aclnnDiv`, `aclnnNeg` | 正向 `aten::_standard_gamma`（否）、`aten::lgamma`（否），反向 `aten::_standard_gamma_grad`（否） |
| 35 | `torch.distribution.laplace.Laplace` | ✅ | `aclnnInplaceUniform`, `aclnnInplaceRandom`, `aclnnSign`, `aclnnAbs`, `aclnnClamp`, `aclnnNeg`, `aclnnLog1p`, `aclnnMul`, `aclnnSub`, `aclnnLog`, `aclnnDiv` | `aclnnSign`, `aclnnMul`, `aclnnNeg`, `aclnnAdds`, `aclnnDiv`, `aclnnMuls` | - |
