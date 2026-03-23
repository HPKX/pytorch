# torch API 191-195 aclnn 接入分析

## 接口 191：`torch.special.bessel_j1`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_bessel_j1` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

无（不可微）

---

## 接口 192：`torch.special.bessel_y0`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_bessel_y0` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

无（不可微）

---

## 接口 193：`torch.special.bessel_y1`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_bessel_y1` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

无（不可微）

---

## 接口 194：`torch.special.i0`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_i0` | 计算接口 | **否** | - | op_plugin 中未找到实现 |
| `aten::i0` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::special_i1` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

---

## 接口 195：`torch.special.i1`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::special_i1` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | opapi/StructKernelNpuOpApi.cpp |
| `aten::where` | 计算接口 | 是 | `aclnnSWhere` | opapi/WhereKernelNpuOpApi.cpp |
| `aten::scalar_tensor` | 框架接口 | N/A | - | 标量转张量 |
| `aten::i0` | 计算接口 | **否** | - | op_plugin 中未找到实现 |
| `aten::reciprocal` | 计算接口 | 是 | `aclnnReciprocal` | StructKernelNpuOpApi.cpp |
| `aten::sub` | 计算接口 | 是 | `aclnnSub / aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 191 | `torch.special.bessel_j1` | ❌ | - | - | 正向 **`aten::special_bessel_j1`（否）** |
| 192 | `torch.special.bessel_y0` | ❌ | - | - | 正向 **`aten::special_bessel_y0`（否）** |
| 193 | `torch.special.bessel_y1` | ❌ | - | - | 正向 **`aten::special_bessel_y1`（否）** |
| 194 | `torch.special.i0` | ❌ | - | `aclnnMul` | 正向 **`aten::special_i0`（否）**、**`aten::i0`（否）**，反向 **`aten::special_i1`（否）** |
| 195 | `torch.special.i1` | ❌ | - | `aclnnAbs`, `aclnnSWhere`, `aclnnReciprocal`, `aclnnSub`, `aclnnMul` | 正向 **`aten::special_i1`（否）**，反向 **`aten::i0`（否）** |
