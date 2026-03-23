# torch API 131-135 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 131 | `torch.nn.TransformerDecoderLayer` | module 组合前向，无统一 whole-layer backward | 由 self-attn、cross-attn、FFN、LayerNorm 的 backward node 组合 | 无单一条目 | `aten::native_layer_norm_backward`、选中的 SDPA backward、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward` / `aten::gelu_backward`、`aten::add.Tensor` | 本层没有 fused ATen forward op；完全继承内部子模块 |
| 132 | `torch.nn.TransformerEncoder` | module 组合前向，无统一 whole-module backward | 训练态由每层 `TransformerEncoderLayer` 的 backward 叠加 | 无单一条目 | `aten::native_layer_norm_backward`、选中的 SDPA backward、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward` / `aten::gelu_backward`、`aten::add.Tensor` | NestedTensor / `_transformer_encoder_layer_fwd` fastpath 只在推理态工作 |
| 133 | `torch.nn.TransformerEncoderLayer` | 训练态：module 组合前向；推理态 fused fastpath 不参与 backward | 训练态无统一 node；推理态 `_transformer_encoder_layer_fwd` 无 autograd | 无单一条目 | `aten::native_layer_norm_backward`、选中的 SDPA backward、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward` / `aten::gelu_backward`、`aten::add.Tensor` | `_transformer_encoder_layer_fwd` 在 `VariableTypeEverything.cpp` 上是 `autogradNotImplementedFallback()` |
| 134 | `torch.nn.TripletMarginLoss` | CIA 前向分解，无专属 backward | 无统一 node（各基础 ATen op 独立 backward） | 无条目；forward composite 为 `pairwise_distance -> [minimum] -> clamp_min -> reduction` | `aten::pairwise_distance`、`aten::norm`、`aten::minimum`、`aten::clamp_min`、`aten::mean` / `aten::sum` | `swap=True` 时多一条 `pairwise_distance(positive, negative)` 和 `minimum` 支路 |
| 135 | `torch.nn.functional.affine_grid` | backward ATen op | `AffineGridGeneratorBackward0` | `theta: affine_grid_generator_backward_symint(grad, size, align_corners)` | `aten::affine_grid_generator_backward` | 规范化后 `_symint` 对应 schema `aten::affine_grid_generator_backward`；前向不是 `cudnn_affine_grid_generator` |

## 详细分析

### 131. torch.nn.TransformerDecoderLayer

**反向来源类型**：module 组合前向，无统一 whole-layer backward

**锁定前向子块**：
- self-attn
- cross-attn
- FFN
- LayerNorm + residual add

**实际 backward 依赖**：
- attention 部分：
  - 选中的 SDPA backend backward：
    - `aten::_scaled_dot_product_flash_attention_backward`
    - `aten::_scaled_dot_product_efficient_attention_backward`
    - `aten::_scaled_dot_product_cudnn_attention_backward`
  - math fallback：
    - `aten::_softmax_backward_data`
    - `aten::native_dropout_backward`
    - `aten::matmul`
- 线性层 / FFN：
  - `aten::addmm`
  - `aten::threshold_backward` 或 `aten::gelu_backward`
- 归一化与残差：
  - `aten::native_layer_norm_backward`
  - `aten::add.Tensor`

**备注**：
- 这是典型的“模块级无统一 backward，完全由内部子 op 反向叠加”的场景

---

### 132. torch.nn.TransformerEncoder

**反向来源类型**：module 组合前向，无统一 whole-module backward

**锁定前向范围**：
- 普通 dense 输入：`N × TransformerEncoderLayer.forward`
- 高 padding 推理 fastpath：`_nested_tensor_from_mask -> N × _transformer_encoder_layer_fwd -> to_padded_tensor`

**关键结论**：
- 有梯度时不会走 `_transformer_encoder_layer_fwd` 这条 fused fastpath
- 因此 backward 只是每一层 `TransformerEncoderLayer` backward 的重复叠加，外加可选末尾 `LayerNorm`

**反向 ATen 依赖**：
- `aten::native_layer_norm_backward`
- 选中的 SDPA backward
- `aten::_softmax_backward_data`
- `aten::native_dropout_backward`
- `aten::addmm`
- `aten::threshold_backward` / `aten::gelu_backward`
- `aten::add.Tensor`

---

### 133. torch.nn.TransformerEncoderLayer

**反向来源类型**：训练态是 module 组合前向；推理态 fused fastpath 不参与 backward

**锁定 schema / fastpath**：
- 推理 fastpath：`aten::_transformer_encoder_layer_fwd(...) -> Tensor`
- `VariableTypeEverything.cpp` 明确把它注册为 `autogradNotImplementedFallback()`

**因此实际 backward 只存在于训练态常规路径**：
- self-attn：
  - SDPA backend backward 或 math fallback
- FFN：
  - `aten::addmm`
  - `aten::threshold_backward` / `aten::gelu_backward`
  - `aten::native_dropout_backward`
- norm / residual：
  - `aten::native_layer_norm_backward`
  - `aten::add.Tensor`

**备注**：
- 这也是 forward 文档里“fastpath 只在 eval/no_grad 条件满足时命中”的反向对应结论

---

### 134. torch.nn.TripletMarginLoss

**反向来源类型**：CIA 前向分解，无专属 backward

**锁定 schema**：
- `aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor`

**derivatives.yaml**：
- `aten::triplet_margin_loss` **无条目**

**CIA 注册验证**：
- `RegisterCompositeImplicitAutogradEverything.cpp` 把 `triplet_margin_loss` 注册为 `CompositeImplicitAutograd`

**native/composite 实现**（`Loss.cpp`）：

```cpp
auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
if (swap) {
  auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
  dist_neg = at::min(dist_neg, dist_swap);
}
auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
return apply_loss_reduction(output, reduction);
```

**立即反向依赖**：
- `aten::pairwise_distance`
- `aten::minimum`
- `aten::clamp_min`
- `aten::mean` / `aten::sum`

**展开后的关键依赖**：
- `aten::norm`

**备注**：
- `swap=False` 时没有 `minimum`
- `swap=True` 时多一条 `pairwise_distance(positive, negative)` 分支

---

### 135. torch.nn.functional.affine_grid

**反向来源类型**：backward ATen op

**锁定 schema**：
- forward：`aten::affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor`
- backward：`aten::affine_grid_generator_backward(Tensor grad, SymInt[] size, bool align_corners) -> Tensor`

**derivatives.yaml 条目**：

```yaml
- name: affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor
  theta: affine_grid_generator_backward_symint(grad, size, align_corners)
```

**name canonicalization**：
- `affine_grid_generator_backward_symint` 规范化后记录为 schema `aten::affine_grid_generator_backward`

**生成代码验证**：
- `VariableTypeEverything.cpp` 为 `affine_grid_generator` 建立 `AffineGridGeneratorBackward0`
- `Functions.cpp` 中：

```cpp
auto grad_result = any_grad_defined ? (affine_grid_generator_backward_symint(grad, size, align_corners)) : Tensor();
```

**native 实现验证**（`AffineGridGenerator.cpp`）：
- 4D 输入走 `affine_grid_generator_4D_backward`
- 5D 输入走 `affine_grid_generator_5D_backward`

**反向 ATen 依赖**：
- 直接依赖：`aten::affine_grid_generator_backward`
- native body 的展开依赖：
  - `aten::empty`
  - `aten::linspace`
  - `aten::copy_`
  - `aten::fill_`
  - `aten::transpose`
  - `aten::bmm`

**备注**：
- 当前接口对应的是 `affine_grid_generator`，不是 `cudnn_affine_grid_generator`

