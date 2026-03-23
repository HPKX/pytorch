# torch API 56-60 aclnn 接入分析

## 接口 56：`torch.igammac`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::igammac` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::igammac.out` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::exp` | 计算接口 | 是 | `aclnnExp` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sub.Scalar` | 计算接口 | 是 | `aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::log` | 计算接口 | 是 | `aclnnLog` | opapi/LogKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::lgamma` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

---

## 接口 57：`torch.inner`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::inner` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::tensordot` | 计算接口 | 否 | - | op_plugin 中未找到实现 |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::dot` | 计算接口 | 是 | `aclnnDot` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::dot` | 计算接口 | 是 | `aclnnDot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::reshape` | view | N/A | - | composite_view |

> `aten::inner` 和 `aten::tensordot` 虽标记为计算接口/否，但它们是 CIA 分解链，前向实际分解为 `permute`（view）+ `reshape`（view）+ `mm`/`dot`（aclnn=是）+ `mul.Tensor`（aclnn=是），实际计算由 aclnn 支持的子 op 完成。

---

## 接口 58：`torch.is_complex`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::is_complex` | 框架接口 | N/A | - | 类型查询接口 |

### 反向依赖

无（返回 `bool`，不可微）

---

## 接口 59：`torch.is_floating_point`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::is_floating_point` | 框架接口 | N/A | - | 类型查询接口 |

### 反向依赖

无（返回 `bool`，不可微）

---

## 接口 60：`torch.is_nonzero`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::is_nonzero` | 框架接口 | N/A | - | 类型查询接口 |
| `aten::item` | 框架接口 | N/A | - | 取值接口 |
| `aten::_local_scalar_dense` | 框架接口 | N/A | - | 标量取值接口 |

### 反向依赖

无（返回 `bool`，不可微）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 56 | `torch.igammac` | ❌ | - | `aclnnNeg`, `aclnnMul`, `aclnnExp`, `aclnnSubs`, `aclnnLog`, `aclnnSub` | **`aten::igammac`（正向）**, **`aten::igammac.out`（正向）**, **`aten::lgamma`（反向）** |
| 57 | `torch.inner` | ✅ | `aclnnMul`, `aclnnMm`, `aclnnDot` | `aclnnMul`, `aclnnMm`, `aclnnDot` | - |
| 58 | `torch.is_complex` | ✅ | - | - | - |
| 59 | `torch.is_floating_point` | ✅ | - | - | - |
| 60 | `torch.is_nonzero` | ✅ | - | - | - |
