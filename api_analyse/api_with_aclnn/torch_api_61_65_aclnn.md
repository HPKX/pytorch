# torch API 61-65 aclnn 接入分析

## 接口 61：`torch.isnan`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::isnan` | 计算接口 | 否 | - | NPU PTA 层实现为 self != self → ne.Tensor（aclnn=是），不阻塞 |
| `aten::ne.Tensor` | 计算接口 | 是 | `aclnnNeTensor` | NeKernelNpuOpApi.cpp |

### 反向依赖

无（`non_differentiable`）

> `aten::isnan` 虽标记为计算接口/否，但 NPU PTA 层将其实现为 `self != self`，等效于 `ne.Tensor`（aclnn=是），不阻塞 A5 支持。注意：`isnan` 在 `native_functions.yaml` 中有 CPU/CUDA/MPS dispatch，并非 CIA composite，NPU 上的支持来自 op_plugin 层的注册覆盖。该接口不可微，无反向依赖。

---

## 接口 62：`torch.isposinf`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::isposinf` | 计算接口 | 是 | `aclnnIsPosInf` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

无（返回 `bool`，不可微）

---

## 接口 63：`torch.isreal`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::isreal` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::ones_like` | 计算接口 | 是 | `aclnnInplaceOne` | OnesLikeKernelNpuOpApi.cpp |
| `aten::imag` | view | N/A | - | composite_view |
| `aten::eq.Scalar` | 计算接口 | 是 | `aclnnEqScalar` | opapi/EqKernelNpuOpApi.cpp |

### 反向依赖

无（返回 `bool`，不可微）

> `aten::isreal` 虽标记为计算接口/否，但它是 CIA 分解函数，前向分解为 `ones_like`（aclnn=是）+ `imag`（view）+ `eq.Scalar`（aclnn=是），实际计算由 aclnn 支持的子 op 完成。且不可微，无反向依赖。

---

## 接口 64：`torch.kaiser_window`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::kaiser_window` | 框架接口 | N/A | - | 窗函数创建接口 |
| `aten::kaiser_window.periodic` | 框架接口 | N/A | - | 窗函数创建接口 |
| `aten::kaiser_window.beta` | 框架接口 | N/A | - | 窗函数创建接口 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::ones` | 计算接口 | 是 | `aclnnInplaceOne` | OnesKernelNpuOpApi.cpp |
| `aten::arange` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::narrow` | view | N/A | - | composite_view |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 65：`torch.kron`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::kron` | 计算接口 | 否 | - | op-plugin 中未找到 kron 实现 |
| `aten::kron.out` | 计算接口 | 否 | - | op-plugin 中未找到 kron 实现 |
| `aten::_unsafe_view` | view | N/A | - | special 类 view 接口 |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

> `aten::kron` 和 `aten::kron.out` 虽标记为计算接口/否，但它们是 CIA 分解链，前向实际分解为 `_unsafe_view`（view）+ `mul.Tensor`（aclnn=是），实际计算由 aclnn 支持的子 op 完成。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 61 | `torch.isnan` | ✅ | `aclnnNeTensor` | - | - |
| 62 | `torch.isposinf` | ✅ | `aclnnIsPosInf` | - | - |
| 63 | `torch.isreal` | ✅ | `aclnnInplaceOne`, `aclnnEqScalar` | - | - |
| 64 | `torch.kaiser_window` | ✅ | `aclnnInplaceOne`, `aclnnArange` | - | - |
| 65 | `torch.kron` | ✅ | `aclnnMul` | `aclnnMul` | - |
