# torch API 151-155 aclnn 接入分析

## 接口 151：`torch.nn.functional.max_unpool1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::max_unpool2d` | 计算接口 | 是 | `aclnnMaxUnpool2d` | StructKernelNpuOpApi.cpp |
| `aten::squeeze.dim` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gather` | 计算接口 | 是 | `aclnnGather` | opapi/GatherKernelNpuOpApi.cpp |
| `as_strided_backward` | view | N/A | - | view backward（unsqueeze/squeeze 引入） |

---

## 接口 152：`torch.nn.functional.max_unpool3d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_unpool3d` | 计算接口 | 是 | `aclnnMaxUnpool3d` | StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::gather` | 计算接口 | 是 | `aclnnGather` | opapi/GatherKernelNpuOpApi.cpp |

---

## 接口 153：`torch.nn.functional.multi_margin_loss`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multi_margin_loss` | 计算接口 | 否 | - | op_plugin 中无实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multi_margin_loss_backward` | 计算接口 | 否 | - | op_plugin 中无实现 |

---

## 接口 154：`torch.nn.functional.multilabel_margin_loss`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multilabel_margin_loss` | 计算接口 | 是 | `aclnnMultilabelMarginLoss` | MultilabelMarginLossKernelNpuOpApi.cpp |
| `aten::multilabel_margin_loss_forward` | 计算接口 | 是 | `aclnnMultilabelMarginLoss` | MultilabelMarginLossKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::multilabel_margin_loss_backward` | 计算接口 | 否 | - | op_plugin 中无 backward 实现 |

---

## 接口 155：`torch.nn.functional.multilabel_soft_margin_loss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::log_sigmoid` | 计算接口 | 是 | `aclnnLogSigmoidForward` | opapi/LogSigmoidNpuOpApi.cpp |
| `aten::log_sigmoid_forward` | 计算接口 | 是 | `aclnnLogSigmoidForward` | opapi/LogSigmoidNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::rsub.Scalar` | 计算接口 | 是 | `aclnnRsubs` | StructKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::add` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::log_sigmoid_backward` | 计算接口 | 是 | `aclnnLogSigmoidBackward` | StructKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::rsub.Scalar` | 计算接口 | 是 | `aclnnRsubs` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::add.Tensor` | 计算接口 | 是 | `aclnnAdd` | opapi/AddKernelNpuOpApi.cpp |
| `aten::sum.dim_IntList` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |
| `aten::div.Scalar` | 计算接口 | 是 | `aclnnDivs` | opapi/DivKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 151 | `torch.nn.functional.max_unpool1d` | ✅ | `aclnnMaxUnpool2d` | `aclnnGather` | - |
| 152 | `torch.nn.functional.max_unpool3d` | ✅ | `aclnnMaxUnpool3d` | `aclnnGather` | - |
| 153 | `torch.nn.functional.multi_margin_loss` | ❌ | - | - | **`aten::multi_margin_loss`（正向）、`aten::multi_margin_loss_backward`（反向）** |
| 154 | `torch.nn.functional.multilabel_margin_loss` | ❌ | `aclnnMultilabelMarginLoss` | - | **`aten::multilabel_margin_loss_backward`（反向）** |
| 155 | `torch.nn.functional.multilabel_soft_margin_loss` | ✅ | `aclnnLogSigmoidForward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | `aclnnLogSigmoidBackward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | - |
