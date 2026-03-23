# torch API 106-110 aclnn 接入分析

## 接口 106：`torch.nn.LSTM`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::lstm.input` | 计算接口 | **否** | - | aclops/LstmKernelNpu.cpp，无 aclnn，使用旧 ACL 算子 |
| `aten::_cudnn_rnn` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_cudnn_rnn_backward` | 计算接口 | **否** | - | CUDA 专用接口，NPU 未实现 |

> **备注**：`aten::lstm.input` 是 CIA composite，在 NPU 上走旧 ACL 算子实现（非 aclnn）；cuDNN 路径仅在 CUDA 设备上触发。NPU 有 LSTM 旧路径但未接入 aclnn。

---

## 接口 107：`torch.nn.LSTMCell`

**A5 支持结论：❌ 不可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::lstm_cell` | 计算接口 | **否** | - | aclops/LstmCellKernelNpu.cpp，无 aclnn，使用旧 ACL 算子 |
| `aten::t` | view | N/A | - | view 类接口 |
| `aten::matmul` | 计算接口 | **否** | - | PyTorch composite 接口，分解为 mm/bmm/dot 等 |
| `aten::_thnn_fused_lstm_cell` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现，fallback |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::_thnn_differentiable_lstm_cell_backward` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现，fallback |
| `aten::_thnn_fused_lstm_cell_backward` | 计算接口 | **否** | - | op_plugin 及 PTA 中均未找到实现，fallback |

> **备注**：`aten::matmul` 标记为否，但它是 composite 接口，会分解为 `mm`/`bmm` 等已接入 aclnn 的算子。然而 `lstm_cell` 和 `_thnn_fused_lstm_cell` 等核心接口均未接入 aclnn，因此整体不支持。

---

## 接口 108：`torch.nn.LeakyReLU`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::leaky_relu` | 计算接口 | 是 | `aclnnLeakyRelu` | opapi/StructKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::leaky_relu_backward` | 计算接口 | 是 | `aclnnLeakyReluBackward` | opapi/StructKernelNpuOpApi.cpp |

---

## 接口 109：`torch.nn.MarginRankingLoss`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::margin_ranking_loss` | 计算接口 | **否** | - | PyTorch composite 接口，无 NPU 注册 |
| `aten::sub` | 计算接口 | 是 | `aclnnSub` / `aclnnSubs` | opapi/SubKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul` | 计算接口 | 是 | `aclnnMul` / `aclnnMuls` | MulKernelNpuOpApi.cpp |
| `aten::add.Scalar` | 计算接口 | 是 | `aclnnAdds` | opapi/AddKernelNpuOpApi.cpp |
| `aten::clamp_min_` | 计算接口 | 是 | `aclnnClampMin` | 通过 clamp_min_out 路径，opapi/StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::sub.Tensor` | 计算接口 | 是 | `aclnnSub` | opapi/SubKernelNpuOpApi.cpp |
| `aten::neg` | 计算接口 | 是 | `aclnnNeg` | StructKernelNpuOpApi.cpp |
| `aten::mul.Tensor` | 计算接口 | 是 | `aclnnMul` | MulKernelNpuOpApi.cpp |
| `aten::clamp_min` | 计算接口 | 是 | `aclnnClampMin` | opapi/StructKernelNpuOpApi.cpp |
| `aten::mean` | 计算接口 | 是 | `aclnnMean` | MeanKernelNpuOpApi.cpp |
| `aten::sum` | 计算接口 | 是 | `aclnnReduceSum` | opapi/SumKernelNpuOpApi.cpp |

> **备注**：`aten::margin_ranking_loss` 标记为计算接口=否，但它是 CIA composite 函数，实际计算由 `sub`、`neg`、`mul`、`clamp_min`、`mean` 等已接入 aclnn 的算子承担，因此不影响 A5 支持判定。

---

## 接口 110：`torch.nn.MaxPool1d`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool1d` | 计算接口 | **否** | - | PyTorch composite 接口，无 NPU 注册 |
| `aten::max_pool1d_with_indices` | 计算接口 | **否** | - | PyTorch composite 接口，无 NPU 注册 |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::max_pool2d_with_indices` | 计算接口 | 是 | `aclnnMaxPool2dWithIndices` / `aclnnMaxPool2dWithMask` | MaxPool2dWithIndicesKernelNpuOpApi.cpp |
| `aten::squeeze` | view | N/A | - | view 类接口 |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::max_pool2d_with_indices_backward` | 计算接口 | 是 | `aclnnMaxPool2dWithIndicesBackward` / `aclnnMaxPool2dWithMaskBackward` | MaxPool2dWithIndicesBackwardKernelNpuOpApi.cpp |

> **备注**：`aten::max_pool1d` 和 `aten::max_pool1d_with_indices` 标记为计算接口=否，但它们是 CIA composite 函数，实际分解为 `unsqueeze -> max_pool2d_with_indices -> squeeze`，核心计算 `max_pool2d_with_indices` 已接入 aclnn，因此不影响 A5 支持判定。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 106 | `torch.nn.LSTM` | ❌ | - | - | **`aten::lstm.input`**、**`aten::_cudnn_rnn`**、**`aten::_cudnn_rnn_backward`** |
| 107 | `torch.nn.LSTMCell` | ❌ | - | - | **`aten::lstm_cell`**、**`aten::_thnn_fused_lstm_cell`**、**`aten::_thnn_differentiable_lstm_cell_backward`**、**`aten::_thnn_fused_lstm_cell_backward`** |
| 108 | `torch.nn.LeakyReLU` | ✅ | `aclnnLeakyRelu` | `aclnnLeakyReluBackward` | - |
| 109 | `torch.nn.MarginRankingLoss` | ✅ | `aclnnSub`、`aclnnNeg`、`aclnnMul`、`aclnnAdds`、`aclnnClampMin`、`aclnnMean` | `aclnnSub`、`aclnnNeg`、`aclnnMul`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | - |
| 110 | `torch.nn.MaxPool1d` | ✅ | `aclnnMaxPool2dWithIndices` | `aclnnMaxPool2dWithIndicesBackward` | - |
