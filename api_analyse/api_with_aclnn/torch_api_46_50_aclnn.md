# torch API 46-50 aclnn 接入分析

## 接口 46：`torch.geqrf`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::geqrf` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::geqrf.a` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

### 反向依赖

无（`not_implemented("geqrf")`，backward 调用会抛异常）

---

## 接口 47：`torch.ger`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::ger` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::ger.out` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::outer` | 计算接口 | 否 | - | PyTorch composite 接口 |
| `aten::outer.out` | 计算接口 | 否 | - | PyTorch composite 接口 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::mul.out` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::conj` | view | N/A | - | CIA composite → `_conj`（仅翻 conj_bit，不调用 NPU kernel）；复数场景受限，见 API 5 adjoint 分析 |

> `aten::ger`、`aten::ger.out`、`aten::outer`、`aten::outer.out` 虽标记为计算接口/否，但它们是 CIA 分解链，前向实际分解为 `reshape`（view）+ `mul.Tensor`（aclnn=是），实际计算由 aclnn 支持的子 op 完成。

---

## 接口 48：`torch.hamming_window`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hamming_window` | 计算接口 | 是 | `aclnnArange + aclnnCos 等` | TensorFactories.cpp，PTA 注册 |
| `aten::hamming_window.periodic_alpha_beta` | 计算接口 | 是 | `aclnnArange + aclnnCos 等` | TensorFactories.cpp，PTA 注册 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::ones` | 计算接口 | 是 | `aclnnInplaceOne` | OnesKernelNpuOpApi.cpp |
| `aten::arange` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::mul_.Scalar` | 计算接口 | 是 | `aclnnInplaceMuls` | MulKernelNpuOpApi.cpp |
| `aten::cos_` | 计算接口 | 是 | `aclnnInplaceCos` | opapi/StructKernelNpuOpApi.cpp |
| `aten::add_.Scalar` | 计算接口 | 是 | `aclnnInplaceAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::narrow` | view | N/A | - | composite_view |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 49：`torch.hann_window`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::hann_window` | 计算接口 | 是 | `aclnnArange + aclnnCos 等` | TensorFactories.cpp，PTA 注册 |
| `aten::hann_window.periodic` | 计算接口 | 是 | `aclnnArange + aclnnCos 等` | TensorFactories.cpp，PTA 注册 |
| `aten::empty` | 框架接口 | N/A | - | 张量创建接口 |
| `aten::ones` | 计算接口 | 是 | `aclnnInplaceOne` | OnesKernelNpuOpApi.cpp |
| `aten::arange` | 计算接口 | 是 | `aclnnArange` | ArangeKernelNpuOpApi.cpp |
| `aten::mul_.Scalar` | 计算接口 | 是 | `aclnnInplaceMuls` | MulKernelNpuOpApi.cpp |
| `aten::cos_` | 计算接口 | 是 | `aclnnInplaceCos` | opapi/StructKernelNpuOpApi.cpp |
| `aten::add_.Scalar` | 计算接口 | 是 | `aclnnInplaceAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::narrow` | view | N/A | - | composite_view |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 50：`torch.heaviside`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::heaviside` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::heaviside.out` | 计算接口 | 否 | - | op-plugin 中未找到实现 |
| `aten::heaviside_` | 计算接口 | 否 | - | op-plugin 中未找到实现 |

### 反向依赖

无（structured op，`autogradNotImplementedFallback`）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 46 | `torch.geqrf` | ❌ | - | - | **`aten::geqrf`（正向）**, **`aten::geqrf.a`（正向）** |
| 47 | `torch.ger` | ✅ | `aclnnMul` | `aclnnMul` | - |
| 48 | `torch.hamming_window` | ✅ | `aclnnArange`, `aclnnInplaceOne`, `aclnnInplaceMuls`, `aclnnInplaceCos`, `aclnnInplaceAdds` | - | - |
| 49 | `torch.hann_window` | ✅ | `aclnnArange`, `aclnnInplaceOne`, `aclnnInplaceMuls`, `aclnnInplaceCos`, `aclnnInplaceAdds` | - | - |
| 50 | `torch.heaviside` | ❌ | - | - | **`aten::heaviside`（正向）**, **`aten::heaviside.out`（正向）**, **`aten::heaviside_`（正向）** |
