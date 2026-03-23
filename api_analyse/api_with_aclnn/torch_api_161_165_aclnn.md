# torch API 161-165 aclnn 接入分析

## 接口 161：`torch.nn.functional.softsign`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::abs` | 计算接口 | 是 | `aclnnAbs` | StructKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::div.Tensor` | 计算接口 | 是 | `aclnnDiv` | opapi/DivKernelNpuOpApi.cpp |
| `aten::sgn` | 计算接口 | 是 | `aclnnSign` | opapi/SignKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |

---

## 接口 162：`torch.nn.functional.triplet_margin_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::triplet_margin_loss` | 计算接口 | 否 | - | CIA 接口，分解为下列子算子 |
| `aten::pairwise_distance` | 计算接口 | 否 | - | CIA 接口，分解为 sub → add.Scalar → norm |
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::norm.ScalarOpt_dim` | 计算接口 | 是 | `aclnnNorm` | NormKernelNpuOpApi.cpp |
| `aten::min.other` | 计算接口 | 是 | `aclnnMinimum` | MinKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::norm.ScalarOpt_dim` | 计算接口 | 是 | `aclnnNorm` | NormKernelNpuOpApi.cpp |
| `aten::min.other` | 计算接口 | 是 | `aclnnMinimum` | MinKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> **注意**：`aten::triplet_margin_loss` 和 `aten::pairwise_distance` 均是 CIA 接口（aclnn=否），但分解后的所有子算子均已接入 aclnn。

---

## 接口 163：`torch.nn.functional.upsample`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::upsample_nearest2d.vec` | 计算接口 | 是 | `aclnnUpsampleNearest2dV2` | opapi/UpsampleNearest2dKernelOpApi.cpp |
| `aten::upsample_nearest2d` | 计算接口 | 是 | `aclnnUpsampleNearest2d` | opapi/UpsampleNearest2dKernelOpApi.cpp |
| `aten::upsample_bilinear2d.vec` | 计算接口 | 是 | `aclnnUpsampleBilinear2d` | opapi/UpsampleBilinear2dKernelNpuOpApi.cpp |
| `aten::upsample_bilinear2d` | 计算接口 | 是 | `aclnnUpsampleBilinear2d` | opapi/UpsampleBilinear2dKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::upsample_nearest2d_backward` | 计算接口 | 是 | `aclnnUpsampleNearest2dBackward` | opapi/UpsampleNearest2dBackwardKernelNpuOpApi.cpp |
| `aten::upsample_bilinear2d_backward` | 计算接口 | 是 | `aclnnUpsampleBilinear2dBackward` | opapi/UpsampleBilinear2dBackwardKernelNpuOpApi.cpp |

---

## 接口 164：`torch.nn.modules.ChannelShuffle`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | StructKernelNpuOpApi.cpp |
| `aten::native_channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | StructKernelNpuOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::reshape` | view | N/A | - | composite_view |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::channel_shuffle` | 计算接口 | 是 | `aclnnChannelShuffle` | StructKernelNpuOpApi.cpp |
| `aten::view` | view | N/A | - | view 类接口 |
| `aten::permute` | view | N/A | - | view 操作 |
| `aten::contiguous` | view | N/A | - | composite_view |
| `aten::reshape` | view | N/A | - | composite_view |

---

## 接口 165：`torch.nn.modules.flatten.Flatten`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::flatten.using_ints` | view | N/A | - | view 类接口 |
| `aten::view` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `as_strided_backward` | view | N/A | - | 标准 view backward |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 161 | `torch.nn.functional.softsign` | ✅ | `aclnnAbs`、`aclnnAdds`、`aclnnDiv` | `aclnnDiv`、`aclnnSign`、`aclnnMul`、`aclnnAdds` | - |
| 162 | `torch.nn.functional.triplet_margin_loss` | ✅ | `aclnnSub`、`aclnnAdds`、`aclnnNorm`、`aclnnMinimum`、`aclnnClampMin`、`aclnnMean` | `aclnnSub`、`aclnnAdds`、`aclnnNorm`、`aclnnMinimum`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | `aten::triplet_margin_loss`、`aten::pairwise_distance`（均为 CIA 分解，子算子均可支持） |
| 163 | `torch.nn.functional.upsample` | ✅ | `aclnnUpsampleNearest2d`、`aclnnUpsampleNearest2dV2`、`aclnnUpsampleBilinear2d` | `aclnnUpsampleNearest2dBackward`、`aclnnUpsampleBilinear2dBackward` | - |
| 164 | `torch.nn.modules.ChannelShuffle` | ✅ | `aclnnChannelShuffle` | `aclnnChannelShuffle` | - |
| 165 | `torch.nn.modules.flatten.Flatten` | ✅ | 全为 view，无需 aclnn | 全为 view backward | - |
