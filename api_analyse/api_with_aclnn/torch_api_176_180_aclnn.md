# torch API 176-180 aclnn 接入分析

## 接口 176：`torch.ormqr`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ormqr` | 计算接口 | **否** | - | op_plugin 中无实现 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::resize_as_` | view | N/A | - | inplace_view 类接口 |
| `aten::transpose_` | view | N/A | - | inplace_view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ormqr` | 计算接口 | **否** | - | op_plugin 中无实现 |
| `aten::tril` | 计算接口 | 是 | `aclnnTril` | opapi/StructKernelNpuOpApi.cpp |
| `aten::diagonal` | view | N/A | - | view 类接口 |
| `aten::fill_` | 计算接口 | 是 | `aclnnInplaceFillScalar / aclnnInplaceFillTensor` | opapi/FillKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::matmul` | 计算接口 | **否** | - | CIA composite，分解为 mm/bmm/dot 等（均已支持），不阻塞 |
| `aten::mH` | view | N/A | - | composite_view |
| `aten::narrow` | view | N/A | - | composite_view |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

---

## 接口 177：`torch.pdist`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::pdist` | 计算接口 | 是 | `aclnnPdist` | PdistKernelNpuOpApi.cpp |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::_pdist_forward` | 计算接口 | 是 | `aclnnPdist` | opapi/PdistKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_pdist_backward` | 计算接口 | **否** | - | op_plugin 中仅有 _pdist_forward 的 opapi 实现，backward 无实现 |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::sum.IntList_out` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

---

## 接口 178：`torch.poisson`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::poisson` | 计算接口 | **否** | - | op_plugin 中无实现 |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |

---

## 接口 179：`torch.polygamma`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::polygamma` | 计算接口 | **否** | - | op_plugin 中无实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::polygamma` | 计算接口 | **否** | - | op_plugin 中无实现，backward 递归调用更高阶 polygamma |

---

## 接口 180：`torch.positive`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::positive` | view | N/A | PyTorch composite 接口，实现为 `return self`，纯 identity/alias，无数据复制 |

### 反向依赖

无（纯 identity/alias，`bool` 输入会报错）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 176 | `torch.ormqr` | ❌ | `aclnnInplaceCopy` | `aclnnTril`, `aclnnInplaceFillScalar`, `aclnnReduceSum`, `aclnnInplaceZero`, `aclnnCat`, `aclnnInplaceCopy` | 正向 **`aten::ormqr`（否）**，反向 **`aten::ormqr`（否）** |
| 177 | `torch.pdist` | ❌ | `aclnnPdist` | `aclnnReduceSum` | 反向 **`aten::_pdist_backward`（否）** |
| 178 | `torch.poisson` | ❌ | `aclnnInplaceZero` | `aclnnInplaceZero` | 正向 **`aten::poisson`（否）** |
| 179 | `torch.polygamma` | ❌ | - | `aclnnMul` | 正向 **`aten::polygamma`（否）**，反向 **`aten::polygamma`（否）** |
| 180 | `torch.positive` | ✅ | 全为 view，无需 aclnn | - | - |
