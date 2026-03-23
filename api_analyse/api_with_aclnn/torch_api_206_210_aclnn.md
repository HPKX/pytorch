# torch API 206-210 aclnn 接入分析

## 接口 206：`torch.triu_indices`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::triu_indices` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 207：`torch.true_divide`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::true_divide.Tensor` | 计算接口 | 是 | `aclnnDiv` | true_divide 为 div 的别名，opapi/DivKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::true_divide.Scalar` | 计算接口 | 是 | `aclnnDivs` | true_divide 为 div 的别名，opapi/DivKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

---

## 接口 208：`torch.vander`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::vander` | 计算接口 | **否** | - | CIA composite，源码分解为 `empty/fill_/copy_/cumprod/flip` |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::select.int` | view | N/A | - | view 类接口 |
| `aten::fill_.Scalar` | 计算接口 | 是 | `aclnnInplaceFillScalar` | opapi/FillKernelNpuOpApi.cpp |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |
| `aten::cumprod` | 计算接口 | 是 | `aclnnCumprod` | opapi/StructKernelNpuOpApi.cpp |
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::cumprod_backward` | 计算接口 | **否** | - | CIA composite，源码由 `cumprod/cumsum/masked_scatter/gather/sum/cat` 等子 op 组成 |
| `aten::slice_backward` | 计算接口 | **否** | - | CEA composite，源码为 `zeros + slice(view) + copy_` |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |
| `aten::flip` | 计算接口 | 是 | `aclnnFlip` | opapi/StructKernelNpuOpApi.cpp |
| `aten::zeros_like` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosLikeKernelNpuOpApi.cpp |

> `aten::vander` 在 `native_functions.yaml` 中无独立 backend dispatch，前向源码直接分解为 `empty/fill_/copy_/cumprod/flip`。反向中的 `cumprod_backward` 与 `slice_backward` 也都是 composite 实现，最终仍落到已接入 aclnn 的子 op 与 view/框架接口上。

---

## 接口 209：`torch.view_as_real`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::view_as_real` | view | N/A | metadata_change 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::contiguous` | view | N/A | composite_view |
| `aten::view_as_complex` | view | N/A | metadata_change 类接口 |

---

## 接口 210：`torch.vsplit`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::vsplit.int` | view | N/A | composite_view，tensor_split(dim=0) 的封装 |
| `aten::vsplit.array` | view | N/A | 同上 |
| `aten::tensor_split.sections` | view | N/A | composite_view 类接口 |
| `aten::tensor_split.indices` | view | N/A | composite_view 类接口 |
| `aten::slice.Tensor` | view | N/A | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::slice_backward` | 计算接口 | **否** | - | CEA composite，源码为 `zeros + slice(view) + copy_` |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

> `torch.vsplit` 只是 `tensor_split(dim=0)` 的包装；前向全为 view。反向里的 `aten::slice_backward` 是 CEA composite，源码分解为 `zeros + slice(view) + copy_`，因此不阻塞 A5 支持。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 206 | `torch.triu_indices` | ❌ | - | - | 正向 **`aten::triu_indices`（否）** |
| 207 | `torch.true_divide` | ✅ | `aclnnDiv`, `aclnnDivs` | `aclnnDiv`, `aclnnDivs`, `aclnnMul`, `aclnnNeg` | - |
| 208 | `torch.vander` | ✅ | `aclnnInplaceFillScalar`, `aclnnInplaceCopy`, `aclnnCumprod`, `aclnnFlip` | `aclnnFlip`, `aclnnInplaceZero` | - |
| 209 | `torch.view_as_real` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 210 | `torch.vsplit` | ✅ | 全为 view，无需 aclnn | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
