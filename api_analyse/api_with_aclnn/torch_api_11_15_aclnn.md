# torch API 11-15 aclnn 接入分析

## 接口 11：`torch.atleast_3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::atleast_3d` | view | N/A | - | CompositeImplicitAutograd，底层调用 reshape/unsqueeze，返回共享存储的 view |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::unsqueeze` | view | N/A | - | view 类接口；dim==1 走两次 unsqueeze，dim==2 走一次 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::reshape` | view | N/A | - | dim==0 时反向 reshape 回标量 |
| `aten::squeeze.dim` | view | N/A | - | unsqueeze 的反向；dim==1 对应两次 squeeze，dim==2 对应一次 |

---

## 接口 12：`torch.bartlett_window`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::bartlett_window` | 计算接口 | 是 | aclnnArange + aclnnMul 等 | CEA composite，PTA 注册 |
| `aten::bartlett_window.periodic` | 计算接口 | 是 | aclnnArange + aclnnMul 等 | CEA composite，PTA 注册 |
| `aten::empty` | 框架接口 | N/A | - | window_length==0 时创建空张量 |
| `aten::ones` | 计算接口 | 是 | aclnnInplaceOne | window_length==1 时创建全 1 张量 |
| `aten::arange` | 计算接口 | 是 | aclnnArange | 生成窗函数序列 |
| `aten::mul_` | 计算接口 | 是 | aclnnInplaceMul / aclnnInplaceMuls | 缩放操作 |
| `aten::narrow` | view | N/A | - | composite_view，截取窗口 |
| `aten::add_` | 计算接口 | 是 | aclnnInplaceAdd / aclnnInplaceAdds | 偏移操作 |

### 反向依赖

无（工厂函数，不可微）

---

## 接口 13：`torch.bitwise_left_shift`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::bitwise_left_shift.Tensor` | 计算接口 | 是 | aclnnLeftShift | structured op，Tensor+Tensor 重载 |
| `aten::bitwise_left_shift.Tensor_Scalar` | 计算接口 | 是 | aclnnLeftShifts | CEA，Tensor+Scalar 重载 |
| `aten::bitwise_left_shift.Scalar_Tensor` | 计算接口 | 是 | aclnnLeftShifts | CEA，Scalar+Tensor 重载 |

### 反向依赖

无（整数位运算，不可微）

---

## 接口 14：`torch.bitwise_right_shift`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::bitwise_right_shift.Tensor` | 计算接口 | 是 | aclnnRightShift | structured op，Tensor+Tensor 重载 |
| `aten::bitwise_right_shift.Tensor_Scalar` | 计算接口 | 是 | aclnnRightShift | CEA，Tensor+Scalar 重载 |
| `aten::bitwise_right_shift.Scalar_Tensor` | 计算接口 | 是 | aclnnRightShift | CEA，Scalar+Tensor 重载 |

### 反向依赖

无（整数位运算，不可微）

---

## 接口 15：`torch.blackman_window`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::blackman_window` | 计算接口 | 是 | aclnnArange + aclnnCos 等 | CEA composite，PTA 注册 |
| `aten::blackman_window.periodic` | 计算接口 | 是 | aclnnArange + aclnnCos 等 | CEA composite，PTA 注册 |
| `aten::empty` | 框架接口 | N/A | - | window_length==0 时创建空张量 |
| `aten::ones` | 计算接口 | 是 | aclnnInplaceOne | window_length==1 时创建全 1 张量 |
| `aten::arange` | 计算接口 | 是 | aclnnArange | 生成窗函数序列 |
| `aten::mul` | 计算接口 | 是 | aclnnMul / aclnnMuls | Blackman 系数乘法 |
| `aten::mul_` | 计算接口 | 是 | aclnnInplaceMul / aclnnInplaceMuls | inplace 缩放 |
| `aten::cos_` | 计算接口 | 是 | aclnnInplaceCos | 余弦变换 |
| `aten::narrow` | view | N/A | - | composite_view，截取窗口 |

### 反向依赖

无（工厂函数，不可微）

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 11 | `torch.atleast_3d` | ✅ | - | - | - |
| 12 | `torch.bartlett_window` | ✅ | `aclnnArange`、`aclnnInplaceMul`/`aclnnInplaceMuls`、`aclnnInplaceAdd`/`aclnnInplaceAdds`、`aclnnInplaceOne` | - | - |
| 13 | `torch.bitwise_left_shift` | ✅ | `aclnnLeftShift`、`aclnnLeftShifts` | - | - |
| 14 | `torch.bitwise_right_shift` | ✅ | `aclnnRightShift` | - | - |
| 15 | `torch.blackman_window` | ✅ | `aclnnArange`、`aclnnMul`/`aclnnMuls`、`aclnnInplaceMul`/`aclnnInplaceMuls`、`aclnnInplaceCos`、`aclnnInplaceOne` | - | - |
