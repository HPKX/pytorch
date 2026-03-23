# torch API 201-205 aclnn 接入分析

## 接口 201：`torch.tanhshrink`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tanh` | 计算接口 | 是 | `aclnnTanh` | opapi/TanhKernelNpuOpApi.cpp |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tanh_backward` | 计算接口 | 是 | `aclnnTanhBackward` | opapi/StructKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp，sub 对第二参数的反向产生 neg |

> 前向就是 `input - input.tanh()`，没有独立 `aten::tanhshrink` schema，全部由已有 aclnn 子 op 承担。反向中 `sub.Tensor` 对第二参数 tanh(x) 的梯度为 `-grad`，即 `neg(grad)`。

---

## 接口 202：`torch.tensor_split`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::tensor_split.sections` | view | N/A | composite_view 类接口 |
| `aten::tensor_split.indices` | view | N/A | composite_view 类接口 |
| `aten::tensor_split.tensor_indices_or_sections` | view | N/A | composite_view 类接口 |
| `aten::slice.Tensor` | view | N/A | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::slice_backward` | 计算接口 | **否** | - | CEA composite，源码为 `zeros + slice(view) + copy_` |
| `aten::zeros` | 计算接口 | 是 | `aclnnInplaceZero` | ZerosKernelNpuOpApi.cpp |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::copy_` | 计算接口 | 是 | `aclnnInplaceCopy` | CopyKernelOpApi.cpp |

> `aten::tensor_split.*` 本身是 view/composite 接口。反向虽然出现 `aten::slice_backward`（表中标记为否），但其源码是 `zeros + slice(view) + copy_`，属于 CEA composite，可完全落到已支持子 op 上执行。

---

## 接口 203：`torch.tensordot`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tensordot` | 计算接口 | **否** | - | CIA composite，源码分解为 `sum/permute/reshape/mm` 或 `dot` |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::dot` | 计算接口 | 是 | `aclnnDot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::mm` | 计算接口 | 是 | `aclnnMm` | MmKernelNpuOpApi.cpp |
| `aten::dot` | 计算接口 | 是 | `aclnnDot` | opapi/StructKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::squeeze` | view | N/A | - | view 类接口 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

> `aten::tensordot` 在 `native_functions.yaml` 中无独立 backend dispatch，等价于 CIA composite。源码显示其前向分解为 `sum/permute/reshape/mm`，完全 contraction 时走 `dot`，反向也仅依赖这些已支持子 op 与 view 操作，因此应判定为可支持。

---

## 接口 204：`torch.trapezoid`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::trapezoid.x` | 计算接口 | **否** | - | CIA composite，源码分解为 `slice/select + add/sub + mul + sum + div` |
| `aten::trapezoid.dx` | 计算接口 | **否** | - | CIA composite，源码分解为 `sum + select + add + mul + sub` |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::slice.Tensor` | view | N/A | - | view 类接口 |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div` | 计算接口 | 是 | `aclnnDiv / aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::select.int` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::slice_backward` | 计算接口 | **否** | - | CEA composite，源码为 `zeros + slice(view) + copy_` |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.Scalar` | 计算接口 | 是 | `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |

> `aten::trapezoid.x` / `aten::trapezoid.dx` 在 `native_functions.yaml` 中均无独立 dispatch，源码直接由 `slice/select + add/sub + mul + sum + div` 拼装。反向中的 `aten::slice_backward` 也是 CEA composite（`zeros + slice(view) + copy_`），不构成真实阻塞。

---

## 接口 205：`torch.tril_indices`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::tril_indices` | 计算接口 | **否** | - | op_plugin 中未找到实现 |

### 反向依赖

无（工厂函数，不可微）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 201 | `torch.tanhshrink` | ✅ | `aclnnTanh`, `aclnnSub` | `aclnnTanhBackward`, `aclnnNeg` | - |
| 202 | `torch.tensor_split` | ✅ | 全为 view，无需 aclnn | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 203 | `torch.tensordot` | ✅ | `aclnnReduceSum`, `aclnnMm`, `aclnnDot`, `aclnnMul` | `aclnnMm`, `aclnnDot`, `aclnnReduceSum`, `aclnnMul` | - |
| 204 | `torch.trapezoid` | ✅ | `aclnnSub`, `aclnnAdd`, `aclnnMul`, `aclnnMuls`, `aclnnReduceSum`, `aclnnDiv` | `aclnnSub`, `aclnnAdd`, `aclnnMul`, `aclnnMuls`, `aclnnReduceSum`, `aclnnDivs` | - |
| 205 | `torch.tril_indices` | ❌ | - | - | 正向 **`aten::tril_indices`（否）** |
