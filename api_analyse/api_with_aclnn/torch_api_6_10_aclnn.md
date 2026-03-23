# torch API 6-10 aclnn 接入分析

## 接口 6：`torch.aminmax`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::aminmax` | 计算接口 | 是 | `aclnnAminmax` | opapi/AminmaxKernelNpuOpApi.cpp |

### 反向依赖

无（不可微，`autogradNotImplementedFallback`）

---

## 接口 7：`torch.angle`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::angle` | 计算接口 | 是 | `aclnnAngleV2` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | 复数分支：条件选择 |
| `aten::eq.Scalar` | 计算接口 | 是 | `aclnnEqScalar` | 复数分支：`self == 0.0` |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | 复数分支：零标量 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | 复数分支：`grad * self` |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | 复数分支：乘以复数标量 `{0, 1}` |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | 复数分支：`… / self.abs().pow(2)` |
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | 复数分支：`self.abs()` |
| `aten::pow.Tensor_Scalar` | 计算接口 | 是 | `aclnnPowTensorScalar` | 复数分支：`.pow(2)` |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | 实数分支：返回全零梯度 |

---

## 接口 8：`torch.argwhere`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::argwhere` | 计算接口 | 否 | - | CIA 函数，分解为 `nonzero`，NPU 上通过 CIA 分解执行，不需要直接实现 |
| `aten::nonzero` | 计算接口 | 是 | `aclnnNonzero` | NonzeroKernelNpuOpApi.cpp；CIA 分解后的实际计算接口 |

### 反向依赖

无（不可微，`nonzero` 标记 `output_differentiability: [False]`）

---

## 接口 9：`torch.atleast_1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::atleast_1d` | view | N/A | - | CIA，≥1d 返回 alias，0d 调用 reshape |
| `aten::reshape` | view | N/A | - | dim==0 时：`self.reshape({1})` |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | dim==0 时 grad reshape 回原始 shape |

---

## 接口 10：`torch.atleast_2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::atleast_2d` | view | N/A | - | CIA，≥2d 返回 alias，否则 reshape/unsqueeze |
| `aten::reshape` | view | N/A | - | dim==0 时：`self.reshape({1,1})` |
| `aten::unsqueeze` | view | N/A | - | dim==1 时：`self.unsqueeze(0)` |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | dim==0 时 grad reshape 回原始 shape |
| `aten::squeeze.dim` | view | N/A | - | dim==1 时：`grad.squeeze(0)` |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 6 | `torch.aminmax` | ✅ | `aclnnAminmax` | 无（不可微） | - |
| 7 | `torch.angle` | ✅ | `aclnnAngleV2` | `aclnnSWhere`、`aclnnEqScalar`、`aclnnInplaceZero`、`aclnnMul`、`aclnnMuls`、`aclnnDiv`、`aclnnAbs`、`aclnnPowTensorScalar` | - |
| 8 | `torch.argwhere` | ✅ | `aclnnNonzero` | 无（不可微） | - |
| 9 | `torch.atleast_1d` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 10 | `torch.atleast_2d` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
