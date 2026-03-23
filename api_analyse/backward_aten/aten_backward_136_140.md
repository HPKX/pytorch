# torch API 136-140 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 136 | `torch.nn.functional.celu` | helper / backward ATen op | `CeluBackward0`（`inplace=True` 时为 `CeluBackward1`） | `self: elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), false, self)` | `aten::elu_backward` | `aten::celu` 是 `CompositeExplicitAutograd` 包装，反向直接挂在 `elu_backward` helper 上 |
| 137 | `torch.nn.functional.cosine_similarity` | CIA 前向分解，无专属 backward | 无统一 node；由 `LinalgVectorNormBackward0`、`DivBackward0`、`MulBackward0`、`SumBackward1`、view/broadcast backward 组合 | 无 derivatives.yaml 条目 | `aten::linalg_vector_norm_backward`、`aten::div.Tensor`、`aten::mul.Tensor`、`aten::sum.dim_IntList`、`aten::expand` | `aten::cosine_similarity` 是 CIA；梯度来自内部 `norm/div/mul/sum` 子算子，广播还会引入 `sum_to_size` / view 类回传 |
| 138 | `torch.nn.functional.ctc_loss` | CIA 前向分解到 backend loss op | 常见为 `CtcLossBackward0` / `CudnnCtcLossBackward0` / `MiopenCtcLossBackward0` | `_ctc_loss: log_probs: _ctc_loss_backward(...)`；`_cudnn_ctc_loss: log_probs: _cudnn_ctc_loss_backward(...)`；`miopen_ctc_loss: log_probs: _miopen_ctc_loss_backward(...)` | `aten::_ctc_loss_backward`、`aten::_cudnn_ctc_loss_backward`、`aten::_miopen_ctc_loss_backward` | 用户入口 `ctc_loss.Tensor` 本身无专属 node；普通 fallback 路径通常先把 lengths 转成 CPU `Long` 后落到 `_ctc_loss` IntList 版本 |
| 139 | `torch.nn.functional.dropout1d` | CIA 前向分解，无专属 backward | 无统一 node；训练态主要为 `MulBackward0` + view backward | 无 derivatives.yaml 条目 | `aten::mul.Tensor`、`as_strided_backward` | `feature_dropout` 训练态本质是 `input * mask`；2D 无 batch 输入时外层 `unsqueeze/squeeze` 额外引入 view backward；`training=False` 或 `p=0` 时为 identity |
| 140 | `torch.nn.functional.dropout2d` | CIA 前向分解，无专属 backward | 无统一 node；训练态主要为 `MulBackward0` | 无 derivatives.yaml 条目 | `aten::mul.Tensor` | 与 `dropout1d` 相同，但标准 4D 路径没有额外 `unsqueeze/squeeze` view backward |

## 详细分析

### 136. torch.nn.functional.celu

**反向来源类型**：helper / backward ATen op

**Forward 路径**：
- `aten::celu` → `CompositeExplicitAutograd(native::celu)` → `aten::elu`
- `inplace=True` 时：`aten::celu_` → `CompositeExplicitAutograd(native::celu_)` → `aten::elu_`

**derivatives.yaml 条目**：
```yaml
- name: celu(Tensor self, Scalar alpha=1.0) -> Tensor
  self: elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ false, self)

- name: celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
  self: elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ true, result)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::celu` 生成 `CeluBackward0`
- inplace 变体 `aten::celu_` 生成 `CeluBackward1`
- `Functions.cpp` 中 `CeluBackward0/1::apply()` 都直接调用 `elu_backward(...)`

**反向 ATen 依赖**：
- `aten::elu_backward` — CELU 的 backward 完全复用 ELU backward schema

**备注**：
- `elu_backward` 是 canonical schema；这里没有额外的二级 ATen 分解，实际数值工作在 backend `elu_backward_stub` 内完成。

---

### 137. torch.nn.functional.cosine_similarity

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 路径**（见对应正向文档）：
`aten::cosine_similarity` → `native::cosine_similarity` → `to(可选)` → `expand` → `linalg_vector_norm` ×2 → `clone` → `clamp_min_` → `div` → `mul` → `sum`

`aten::cosine_similarity` 在 `derivatives.yaml` 中**无条目**，因此 autograd 逐个记录内部子 op：

| 子 op | backward 来源 | 反向依赖 |
| --- | --- | --- |
| `aten::linalg_vector_norm` | `LinalgVectorNormBackward0` | `aten::linalg_vector_norm_backward` |
| `aten::div.Tensor` | 标准除法 backward | `aten::div.Tensor`、`aten::mul.Tensor` |
| `aten::mul.Tensor` | 标准乘法 backward | `aten::mul.Tensor` |
| `aten::sum.dim_IntList` | 归约 backward | `aten::expand` / `sum_to_size` |
| `aten::expand` | view/broadcast backward | broadcast 回传到原 shape |
| `aten::clone` | identity | 无新增算子依赖 |

**反向 ATen 依赖**：
- `aten::linalg_vector_norm_backward`
- `aten::div.Tensor`
- `aten::mul.Tensor`
- `aten::sum.dim_IntList`
- `aten::expand`

**备注**：
- `clamp_min_` 只负责把分母下界钳到 `eps`；其梯度是阈值 mask 语义，但 `cosine_similarity` 本身没有统一 backward node。
- 若输入先发生整型提升，`aten::to` 只影响前向 dtype，不为输入提供梯度。

---

### 138. torch.nn.functional.ctc_loss

**反向来源类型**：CIA 前向分解到 backend loss op

**用户入口 Forward 路径**：
- fallback：`aten::ctc_loss.Tensor` → `native::ctc_loss` → `aten::_ctc_loss`
- CUDA/cuDNN：`aten::ctc_loss.Tensor` → `aten::_cudnn_ctc_loss`
- CUDA/MIOpen：`aten::ctc_loss.Tensor` → `aten::miopen_ctc_loss`

**derivatives.yaml 条目**：
```yaml
- name: _ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  log_probs: _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)

- name: _cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
  log_probs: _cudnn_ctc_loss_backward(grad, result0, result1, zero_infinity)

- name: miopen_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
  log_probs: _miopen_ctc_loss_backward(grad, result0, result1, zero_infinity)
```

**生成代码**：
- `_ctc_loss` 常见路径在 `VariableTypeEverything.cpp` 生成 `CtcLossBackward0`
- `_cudnn_ctc_loss` 生成 `CudnnCtcLossBackward0`
- `miopen_ctc_loss` 生成 `MiopenCtcLossBackward0`
- `Functions.cpp` 对应 `apply()` 分别调用 `_ctc_loss_backward(...)`、`_cudnn_ctc_loss_backward(...)`、`_miopen_ctc_loss_backward(...)`

**反向 ATen 依赖**：
- `aten::_ctc_loss_backward`
- `aten::_cudnn_ctc_loss_backward`
- `aten::_miopen_ctc_loss_backward`

**helper 展开**：
- `_cudnn_ctc_loss_backward` / `_miopen_ctc_loss_backward`（`FunctionsManual.cpp`）会进一步调用：
  - `aten::unsqueeze`
  - `aten::where`（`zero_infinity=True` 时）
  - `aten::zeros`
  - `aten::mul.Tensor`

**备注**：
- `ctc_loss.Tensor` 本身是 CIA convenience wrapper，不单独生成 backward node。
- 常见 `F.ctc_loss` fallback 路径会先把 length tensor 转成 CPU `Long` 再调 `_ctc_loss` IntList 版，因此通常观察到的是 `CtcLossBackward0`。

---

### 139. torch.nn.functional.dropout1d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 路径**：
- batched 3D：`feature_dropout` → `_dropout_impl(feature=true)` → `mul`
- 2D 非 batched：`unsqueeze` → `feature_dropout` → `mul` → `squeeze`
- `training=False` 或 `p=0`：直接返回输入

`feature_dropout` 在 `derivatives.yaml` 中**无条目**，反向落在分解出的子 op 上。

**训练态反向 ATen 依赖**：
- `aten::mul.Tensor` — `input * noise`
- `as_strided_backward` — 仅 2D 无 batch 输入时，由 `unsqueeze/squeeze` 的 view backward 引入

**备注**：
- `new_empty` / `bernoulli_` / `div_` 只参与生成 mask，不对输入张量产生梯度。
- `training=False` 或 `p=0` 时没有乘法链，反向就是 identity。

---

### 140. torch.nn.functional.dropout2d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 路径**：
`feature_dropout` → `_dropout_impl(feature=true)` → `new_empty` → `bernoulli_` → `div_` → `mul`

**反向 ATen 依赖**：
- `aten::mul.Tensor`

**备注**：
- 与 `dropout1d` 相同，mask 采样链本身不提供输入梯度。
- 历史兼容的 3D 输入仍复用同一条 `feature_dropout` 反向机制。
