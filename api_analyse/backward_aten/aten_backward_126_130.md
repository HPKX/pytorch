# torch API 126-130 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 126 | `torch.nn.Softmin` | Python 组合前向，无专属 backward | 无统一 node（`SoftmaxBackward0` + `NegBackward0` 组合） | 无单一条目；forward 为 `(-input).softmax(dim)` | `aten::_softmax_backward_data`、`aten::neg` | `softmin` 本身不是独立 ATen schema；梯度先过 inner softmax，再过外层取负 |
| 127 | `torch.nn.Softsign` | Python 组合前向，无专属 backward | 无统一 node（`DivBackward0`、`AbsBackward0` 等组合） | 无条目；forward 为 `input / (input.abs() + 1)` | `aten::div.Tensor`、`aten::abs`、`aten::sgn`、`aten::add.Scalar`、`aten::mul.Tensor`、`aten::neg` | `abs` 的 backward 来自 `grad * self.sgn()` |
| 128 | `torch.nn.Tanhshrink` | Python 组合前向，无专属 backward | 无统一 node（`SubBackward0` + `TanhBackward0`） | 无条目；forward 为 `input - input.tanh()` | `aten::tanh_backward`、`aten::neg` | 直接梯度是 identity 减去 `tanh` 分支梯度 |
| 129 | `torch.nn.Transformer` | module 组合前向，无统一 whole-module backward | 由 MHA / LayerNorm / FFN / Dropout 等子模块的 backward node 组合 | 无单一条目 | `aten::native_layer_norm_backward`、选中的 SDPA backward、math fallback 的 `aten::_softmax_backward_data` / `aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward` / `aten::gelu_backward`、`aten::add.Tensor` | `_transformer_encoder_layer_fwd`、`_native_multi_head_attention` 只在推理 fastpath 使用，autograd 不会走它们 |
| 130 | `torch.nn.TransformerDecoder` | module 组合前向，无统一 whole-module backward | 由 self-attn、cross-attn、FFN、LayerNorm 的 backward node 组合 | 无单一条目 | 与 `Transformer` 类似：`aten::native_layer_norm_backward`、选中的 SDPA backward、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward` / `aten::gelu_backward`、`aten::add.Tensor` | 当前前向文档同时包含常规训练路径和 self-attn fastpath；有梯度时实际落回常规路径 |

## 详细分析

### 126. torch.nn.Softmin

**反向来源类型**：Python 组合前向，无专属 backward

**锁定前向表达式**（`functional.py`）：

```python
ret = (-input).softmax(dim)
```

因此 backward 分两层：
1. inner `softmax(dim)` 的梯度
2. outer `-input` 的梯度

**反向 ATen 依赖**：
- `aten::_softmax_backward_data`
- `aten::neg`

**备注**：
- `softmin` 本身不是独立 native schema；反向完全继承 `softmax` 和 `neg`

---

### 127. torch.nn.Softsign

**反向来源类型**：Python 组合前向，无专属 backward

**锁定前向表达式**（`functional.py`）：

```python
return input / (input.abs() + 1)
```

**子 op 拆解**：
- `aten::abs`
- `aten::add.Scalar`
- `aten::div.Tensor`

**关键 backward 继承**：
- `abs` 的 backward：`grad * self.sgn()`
- `div.Tensor` 的 backward：包含除法、乘法、取负等标准依赖

**反向 ATen 依赖**：
- `aten::div.Tensor`
- `aten::abs`
- `aten::sgn`
- `aten::add.Scalar`
- `aten::mul.Tensor`
- `aten::neg`

**备注**：
- 这里没有统一的 `SoftsignBackward0`

---

### 128. torch.nn.Tanhshrink

**反向来源类型**：Python 组合前向，无专属 backward

**锁定前向表达式**（`functional.py`）：

```python
return input - input.tanh()
```

**子 op 拆解**：
- `aten::tanh`
- `aten::sub.Tensor`

**关键 backward**：
- `tanh` 的梯度来自 `aten::tanh_backward`
- `sub` 对第二个分支会引入一个符号翻转

**反向 ATen 依赖**：
- `aten::tanh_backward`
- `aten::neg`

**备注**：
- 第一项 `input` 的梯度是 identity，不额外引入 ATen op

---

### 129. torch.nn.Transformer

**反向来源类型**：module 组合前向，无统一 whole-module backward

**锁定前向范围**（对应 forward 文档）：
- 内置 encoder + decoder 的常规训练路径
- encoder side 的 `_transformer_encoder_layer_fwd` 推理 fastpath

**关键结论**：
- `_transformer_encoder_layer_fwd` 和 `_native_multi_head_attention` 在 autograd key 上都注册为 `autogradNotImplementedFallback()`
- 所以训练 / `requires_grad=True` 的 backward 不会落在 fused fastpath，而是落在模块内部标准 ATen 子图上

**训练路径的主要 backward 依赖**：
- Attention：
  - 选中的 SDPA backend backward：
    - `aten::_scaled_dot_product_flash_attention_backward`
    - `aten::_scaled_dot_product_efficient_attention_backward`
    - `aten::_scaled_dot_product_cudnn_attention_backward`
  - math fallback：
    - `aten::matmul`
    - `aten::_softmax_backward_data`
    - `aten::native_dropout_backward`
    - `aten::matmul`
- 投影 / FFN：
  - `aten::addmm` / `aten::mm` / `aten::t`
  - `aten::threshold_backward` 或 `aten::gelu_backward`
  - `aten::native_dropout_backward`
- 归一化 / 残差：
  - `aten::native_layer_norm_backward`
  - `aten::add.Tensor`

**备注**：
- `Transformer` 只是把这些子模块串起来，没有统一的单 ATen schema 和单 backward node

---

### 130. torch.nn.TransformerDecoder

**反向来源类型**：module 组合前向，无统一 whole-module backward

**锁定前向范围**：
- 常规训练路径：self-attn + cross-attn + FFN
- self-attn 满足条件时可进入 `_native_multi_head_attention` fastpath，但该 fastpath 不支持 autograd

**实际 backward 依赖**：
- self-attn / cross-attn：
  - 选中的 SDPA backward 或 math fallback
  - 输出投影 `linear` 的 `addmm/mm/t`
- FFN：
  - `aten::addmm`
  - `aten::threshold_backward` / `aten::gelu_backward`
  - `aten::native_dropout_backward`
- norm / residual：
  - `aten::native_layer_norm_backward`
  - `aten::add.Tensor`

**备注**：
- 有梯度时不会停留在 `_native_multi_head_attention`

