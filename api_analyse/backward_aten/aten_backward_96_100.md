# torch API 96-100 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 96 | `torch.nn.GRUCell` | CIA 前向分解 → 内部 op 各自反向 | CUDA fused：`ThnnFusedGruCellBackward0`；CPU fallback：各子 op 独立 backward | CUDA fused：同 GRU 的 `_thnn_fused_gru_cell` 公式；CPU：各子 op 各自公式 | CUDA fused：`aten::_thnn_fused_gru_cell_backward` / `aten::_thnn_differentiable_gru_cell_backward`；CPU：`aten::mul.Tensor`、`aten::sigmoid_backward`、`aten::tanh_backward` 等 | `gru_cell` CIA 分解为 `linear` + GRU gates 计算 |
| 97 | `torch.nn.GaussianNLLLoss` | CIA 前向分解（Python composite），无专属 backward | 无统一 backward node（各基础 ATen op 独立 backward） | 无条目（纯 Python composite） | `aten::neg`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::div.Tensor`、`aten::conj`、`aten::pow.Tensor_Scalar`、`aten::expand_symint`、`aten::div.Scalar` | 没有 `aten::gaussian_nll_loss`；模块完全由基础 ATen op 组合，这里按递归展开口径列到基础 backward ATen op |
| 98 | `torch.nn.HardTanh` | backward ATen op | `HardtanhBackward0` | `self: hardtanh_backward(grad, self, min_val, max_val)` | `aten::hardtanh_backward`（CPU/CUDA/MPS backend kernel） | `hardtanh` 前向 CEA；backward 有独立 backend kernel |
| 99 | `torch.nn.HingeEmbeddingLoss` | CIA 前向分解，无专属 backward | 无统一 backward node（各子 op 独立 backward） | 无条目（CIA） | `aten::neg`、`aten::ge.Scalar`、`aten::where`、`aten::expand_symint`、`aten::div.Scalar` | `hinge_embedding_loss` CIA 分解为 `zeros_like` → `rsub` → `clamp_min` → `where` → `add` → `mean`，这里按递归展开口径列出最终 backward 依赖 |
| 100 | `torch.nn.HuberLoss` | backward ATen op | `HuberLossBackward0` | `self: huber_loss_backward(grad, self, target, reduction, delta)`；`target: huber_loss_backward(grad, target, self, reduction, delta)` | `aten::huber_loss_backward`（CEA，内部使用 backend kernel） | self 和 target 的梯度通过交换参数位置使用同一 backward op |

## 详细分析

### 96. torch.nn.GRUCell

**反向来源类型**：CIA 前向分解 → 内部 op 各自反向

**Forward 分解路径**：`aten::gru_cell` (CIA) → `native::GRUCell<CellParams>` → 多个路径：

1. **CPU fallback 路径**：→ `aten::linear(input, w_ih, b_ih)` + `aten::linear(hx, w_hh, b_hh)` → `aten::unsafe_chunk` → `aten::add_`/`aten::sigmoid_`/`aten::mul_`/`aten::tanh_`/`aten::sub`
2. **CUDA fused 路径**：→ `aten::t` + `aten::matmul`（input/hx 各一次）→ `aten::_thnn_fused_gru_cell` → CUDA kernel

#### CPU fallback 路径

各基础 ATen op 独立录制 autograd：
- `linear` → `LinearBackward0`（依赖 `aten::matmul`、`aten::t`、`aten::sum` 等）
- `sigmoid_` → `SigmoidBackward1`（依赖 `aten::sigmoid_backward`）
- `tanh_` → `TanhBackward1`（依赖 `aten::tanh_backward`）
- `mul_` → `MulBackward0`（依赖 `mul_tensor_backward`）
- `add_` / `sub` → `AddBackward0` / `SubBackward0`

#### CUDA fused 路径

与 GRU（接口 95）的 fused 路径完全相同：

**`_thnn_fused_gru_cell` 的 derivatives.yaml 条目**：
```yaml
- name: _thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
  input_gates, hidden_gates, hx, input_bias, hidden_bias: "grad.defined() ? (GradMode::is_enabled() ? _thnn_differentiable_gru_cell_backward(...) : _thnn_fused_gru_cell_backward(...)) : ..."
```

**Backward Node**：`ThnnFusedGruCellBackward0`

**反向 ATen 依赖（CUDA fused，grad mode off）**：
- `aten::_thnn_fused_gru_cell_backward` — CUDA backend kernel

**反向 ATen 依赖（CUDA fused，grad mode on / differentiable）**：
- `aten::_thnn_differentiable_gru_cell_backward` — 展开为：
  - `aten::add.Tensor`、`aten::unsafe_chunk`、`aten::sigmoid`、`aten::tanh`
  - `aten::sigmoid_backward`、`aten::tanh_backward`
  - `aten::mul.Tensor`、`aten::sub.Tensor`、`aten::cat`、`aten::sum`

---

### 97. torch.nn.GaussianNLLLoss

**反向来源类型**：CIA 前向分解（Python composite），无专属 backward

**Forward 分解路径**：`F.gaussian_nll_loss` 是纯 Python 函数，没有对应的 `aten::gaussian_nll_loss` schema。内部由基础 ATen op 组合：

```python
# 简化逻辑：
loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)
# 实际 ATen 调用链：
# aten::sub → aten::pow.Tensor_Scalar → aten::div.Tensor → aten::log → aten::add.Tensor → aten::mul.Scalar → aten::mean
```

每个子 op 独立录制 autograd，没有统一的 backward node。

**各子 op 的反向 ATen 依赖**：

| 子 op | Backward Node | 反向依赖 |
| --- | --- | --- |
| `aten::sub` | `SubBackward0` | `aten::neg`（other 梯度） |
| `aten::pow.Tensor_Scalar` | `PowBackward1` | `aten::mul.Scalar`、`aten::pow.Tensor_Scalar` |
| `aten::div.Tensor` | `DivBackward1` | `aten::div.Tensor`、`aten::mul.Tensor`、`aten::neg`、`aten::conj` |
| `aten::log` | `LogBackward0` | `aten::div.Tensor`（`grad / self`） |
| `aten::add.Tensor` | `AddBackward0` | 直接传递 grad（identity） |
| `aten::mul.Scalar` | `MulBackward1` | `aten::mul.Scalar`/`aten::mul.Tensor` |
| `aten::mean` | `MeanBackward0` | `aten::expand_symint`、`aten::div.Scalar` |

---

### 98. torch.nn.HardTanh

**反向来源类型**：backward ATen op

**Forward 路径**：`aten::hardtanh` → backend wrapper → `aten::hardtanh.out` → `aten::clamp.out`

**derivatives.yaml 条目**：
```yaml
- name: hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
  self: hardtanh_backward(grad, self, min_val, max_val)
  result: auto_element_wise
```

**Backward Node**：`HardtanhBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = hardtanh_backward(grad, self, min_val, max_val);
```

**`hardtanh_backward`** 是 ATen op，有独立的 CPU/CUDA/MPS backend kernel：
- CPU, CUDA: `hardtanh_backward` / `hardtanh_backward_out`
- MPS: `hardtanh_backward_mps` / `hardtanh_backward_out_mps`

内部逻辑：当 `min_val < self < max_val` 时传递 grad，否则返回 0。

**反向 ATen 依赖**：
- `aten::hardtanh_backward` — 整体作为 backend kernel 执行

---

### 99. torch.nn.HingeEmbeddingLoss

**反向来源类型**：CIA 前向分解，无专属 backward

**Forward 分解路径**：`aten::hinge_embedding_loss` (CIA) → `at::native::hinge_embedding_loss` → `aten::zeros_like` → `aten::rsub.Scalar`（`margin - self`）→ `aten::clamp_min_` / `aten::clamp_min` → `aten::where`（×2，分别处理 y==1 和 y==-1）→ `aten::add` → reduction（`aten::mean` / `aten::sum`）

无 derivatives.yaml 条目。autograd 对每个子 op 独立录制。

**各子 op 的反向 ATen 依赖**：

| 子 op | Backward Node | 反向依赖 |
| --- | --- | --- |
| `aten::rsub.Scalar` | `RsubBackward1` | `aten::neg` |
| `aten::clamp_min` | `ClampMinBackward0` | `aten::ge.Scalar`、`aten::where` |
| `aten::where` | `WhereBackward0` | `aten::where`（条件选择 grad 或 zeros） |
| `aten::add.Tensor` | `AddBackward0` | 直接传递 grad |
| `aten::mean` | `MeanBackward0` | `aten::expand_symint`、`aten::div.Scalar` |

注意：`zeros_like` 创建的零张量不需要梯度。`target`（y）通常为 label（+1/-1），不参与梯度计算。

---

### 100. torch.nn.HuberLoss

**反向来源类型**：backward ATen op

**Forward 路径**：`aten::huber_loss` → backend wrapper → `aten::empty_like` → `huber_stub` → backend kernel → reduction（`aten::mean`）

**derivatives.yaml 条目**：
```yaml
- name: huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
  self: huber_loss_backward(grad, self, target, reduction, delta)
  target: huber_loss_backward(grad, target, self, reduction, delta)
```

**Backward Node**：`HuberLossBackward0`

**生成代码**（`Functions.cpp`）：
```cpp
// self 梯度
auto grad_result = huber_loss_backward(grad, self, target, reduction, delta);
// target 梯度（注意参数交换：target, self）
auto grad_result = huber_loss_backward(grad, target, self, reduction, delta);
```

self 和 target 的梯度通过交换参数位置使用同一 backward op。

**`huber_loss_backward`** 是 CEA ATen op，内部使用 backend kernel：
- CPU, CUDA: `huber_loss_backward_out`
- MPS: `huber_loss_backward_out_mps`

Huber loss backward 的逻辑：
- 当 `|self - target| < delta` 时（二次区域）：`grad * (self - target) / delta * reduction_factor`
- 当 `|self - target| >= delta` 时（线性区域）：`grad * sign(self - target) * reduction_factor`

**反向 ATen 依赖**：
- `aten::huber_loss_backward` — 整体作为 CEA op + backend kernel 执行
