# torch API 111-115 aclnn 接入分析

## 接口 111：`torch.nn.MaxPool2d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool2d` | 计算接口 | **否** | - | PyTorch composite 接口，委托给 max_pool2d_with_indices |
| `aten::max_pool2d_with_indices` | 计算接口 | 是 | `aclnnMaxPool2dWithIndices` / `aclnnMaxPool2dWithMask` | MaxPool2dWithIndicesKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool2d_backward` | 计算接口 | **否** | - | PyTorch composite 接口，委托给 max_pool2d_with_indices_backward |
| `aten::max_pool2d_with_indices_backward` | 计算接口 | 是 | `aclnnMaxPool2dWithIndicesBackward` / `aclnnMaxPool2dWithMaskBackward` | MaxPool2dWithIndicesBackwardKernelNpuOpApi.cpp |

> **备注**：`aten::max_pool2d` 和 `aten::max_pool2d_backward` 标记为计算接口=否，但它们是 composite 接口，实际计算由 `max_pool2d_with_indices` 和 `max_pool2d_with_indices_backward`（均已接入 aclnn）承担，因此不影响 A5 支持判定。

---

## 接口 112：`torch.nn.MaxPool3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool3d` | 计算接口 | **否** | - | PyTorch composite 接口，委托给 max_pool3d_with_indices |
| `aten::max_pool3d_with_indices` | 计算接口 | 是 | `aclnnMaxPool3dWithArgmax` | MaxPool3dWithIndicesKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool3d_with_indices_backward` | 计算接口 | 是 | `aclnnMaxPool3dWithArgmaxBackward` | MaxPool3dWithIndicesBackwardKernelNpuOpApi.cpp |

> **备注**：`aten::max_pool3d` 标记为计算接口=否，但它是 composite 接口，实际计算由 `max_pool3d_with_indices`（已接入 aclnn）承担，因此不影响 A5 支持判定。

---

## 接口 113：`torch.nn.MaxUnpool1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::max_unpool2d` | 计算接口 | 是 | `aclnnMaxUnpool2d` | StructKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::gather` | 计算接口 | 是 | `aclnnGather` | opapi/GatherKernelNpuOpApi.cpp |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |

---

## 接口 114：`torch.nn.MaxUnpool3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_unpool3d` | 计算接口 | 是 | `aclnnMaxUnpool3d` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::gather` | 计算接口 | 是 | `aclnnGather` | opapi/GatherKernelNpuOpApi.cpp |
| `aten::empty_like` | 框架接口 | N/A | - | 张量创建接口 |

---

## 接口 115：`torch.nn.MultiLabelMarginLoss`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multilabel_margin_loss` | 计算接口 | 是 | `aclnnMultilabelMarginLoss` | MultilabelMarginLossKernelNpuOpApi.cpp |
| `aten::multilabel_margin_loss_forward` | 计算接口 | 是 | `aclnnMultilabelMarginLoss` | MultilabelMarginLossKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multilabel_margin_loss_backward` | 计算接口 | **否** | - | op_plugin 中无 backward 实现 |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 111 | `torch.nn.MaxPool2d` | ✅ | `aclnnMaxPool2dWithIndices` | `aclnnMaxPool2dWithIndicesBackward` | - |
| 112 | `torch.nn.MaxPool3d` | ✅ | `aclnnMaxPool3dWithArgmax` | `aclnnMaxPool3dWithArgmaxBackward` | - |
| 113 | `torch.nn.MaxUnpool1d` | ✅ | `aclnnMaxUnpool2d` | `aclnnGather` | - |
| 114 | `torch.nn.MaxUnpool3d` | ✅ | `aclnnMaxUnpool3d` | `aclnnGather` | - |
| 115 | `torch.nn.MultiLabelMarginLoss` | ❌ | `aclnnMultilabelMarginLoss` | - | **`aten::multilabel_margin_loss_backward`** |
