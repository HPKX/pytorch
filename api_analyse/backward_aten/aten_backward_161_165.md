# torch API 161-165 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 161 | `torch.nn.functional.softsign` | Python 组合前向，无专属 backward | 无统一 node；由 `DivBackward0`、`AbsBackward0`、`AddBackward0` 组合 | 无单独条目 | `aten::div.Tensor`、`aten::sgn`、`aten::mul.Tensor`、`aten::add.Scalar` | `softsign(x)=x/(1+abs(x))`；`abs` 的 backward 进一步依赖 `sgn` 和乘法 |
| 162 | `torch.nn.functional.triplet_margin_loss` | CIA 前向分解，无专属 backward | 无统一 node；由多个 `pairwise_distance` 子图与 `clamp_min/min/reduction` 组合 | 无 derivatives.yaml 条目 | `aten::sub.Tensor`、`aten::add.Scalar`、`aten::norm.ScalarOpt_dim`、`aten::min.other`、`aten::clamp_min`、`aten::mean`、`aten::sum` | `pairwise_distance` 本身也是 CIA：`norm(x1 - x2 + eps, p, dim)`；`swap=True` 时多一条 `positive-negative` 分支 |
| 163 | `torch.nn.functional.upsample` | CIA vec helper 转到真实 upsample op | `UpsampleNearest2DBackward0` / `UpsampleBilinear2DBackward0` | `upsample_nearest2d: self: upsample_nearest2d_backward_symint(...)`；`upsample_bilinear2d: self: upsample_bilinear2d_backward_symint(...)` | `aten::upsample_nearest2d_backward`、`aten::upsample_bilinear2d_backward` | `upsample_nearest2d.vec` / `upsample_bilinear2d.vec` 只是 convenience 包装；真正 backward 挂在实际 2D op 上 |
| 164 | `torch.nn.modules.ChannelShuffle` | helper / backward ATen op | `ChannelShuffleBackward0` | `self: channel_shuffle_symint(grad, grad.sym_size(1) / groups)` | `aten::channel_shuffle`、`aten::view`、`aten::permute`、`aten::contiguous`、`aten::reshape` | backward 公式再次调用 `channel_shuffle`；CUDA / math 路径向下会分解成 view-permute-contiguous-reshape |
| 165 | `torch.nn.modules.flatten.Flatten` | CIA 前向分解为 reshape / view | 常见连续内存路径为 `ViewBackward0` | 无 derivatives.yaml 条目 | `as_strided_backward` | 默认连续输入上 `flatten.using_ints` → `reshape_symint` → `view_symint`，因此实际 backward 是标准 view backward |

## 详细分析

### 161. torch.nn.functional.softsign

**反向来源类型**：Python 组合前向，无专属 backward

**Forward 公式**（`functional.py`）：
```python
return input / (input.abs() + 1)
```

**反向 ATen 依赖**：
- `aten::div.Tensor`
- `aten::abs`
  - `derivatives.yaml` 为 `abs` 给出 `grad * self.sgn()`
  - 因此展开依赖：
    - `aten::sgn`
    - `aten::mul.Tensor`
- `aten::add.Scalar`

**备注**：
- `softsign` 没有独立 schema，文档应直接记录组成它的基础算子 backward。

---

### 162. torch.nn.functional.triplet_margin_loss

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 公式**（`Loss.cpp`）：
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

其中 `pairwise_distance` 自身也是 CIA：
```cpp
return at::norm(x1 - x2 + eps, p, innermost_dim, keepdim);
```

**反向 ATen 依赖**：
- `aten::sub.Tensor`
- `aten::add.Scalar`
- `aten::norm.ScalarOpt_dim`
- `aten::min.other` — 仅 `swap=True`
- `aten::clamp_min`
- `aten::mean` / `aten::sum`

**备注**：
- `pairwise_distance` 的核心 backward 由 `norm_backward` 驱动，但公开图层面记录为 `aten::norm.ScalarOpt_dim` 即可。
- `swap=False` 时有两条 distance 支路；`swap=True` 时增加第三条 `positive-negative` distance 支路和一次 `min.other`。

---

### 163. torch.nn.functional.upsample

**反向来源类型**：CIA vec helper 转到真实 upsample op

**Forward 路径**：
- `mode='nearest'`：
  - `aten::upsample_nearest2d.vec`（CIA）→ `aten::upsample_nearest2d`
- `mode='bilinear'`：
  - `aten::upsample_bilinear2d.vec`（CIA）→ `aten::upsample_bilinear2d`

**derivatives.yaml 条目**：
```yaml
- name: upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
  self: upsample_nearest2d_backward_symint(grad, output_size, self.sym_sizes(), scales_h, scales_w)

- name: upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
  self: upsample_bilinear2d_backward_symint(grad, output_size, self.sym_sizes(), align_corners, scales_h, scales_w)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为真实 2D op 生成：
  - `UpsampleNearest2DBackward0`
  - `UpsampleBilinear2DBackward0`
- `Functions.cpp` 的 `apply()` 分别调用 `upsample_nearest2d_backward_symint(...)`、`upsample_bilinear2d_backward_symint(...)`

**反向 ATen 依赖**：
- `aten::upsample_nearest2d_backward`
- `aten::upsample_bilinear2d_backward`

**备注**：
- 文档里对 `_symint` 做 canonicalization，统一记为对应 schema 的 backward。

---

### 164. torch.nn.modules.ChannelShuffle

**反向来源类型**：helper / backward ATen op

**derivatives.yaml 条目**：
```yaml
- name: channel_shuffle(Tensor self, SymInt groups) -> Tensor
  self: channel_shuffle_symint(grad, grad.sym_size(1) / groups)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::channel_shuffle` 生成 `ChannelShuffleBackward0`
- `Functions.cpp` 中 `ChannelShuffleBackward0::apply()` 调用 `channel_shuffle_symint(...)`

**反向 ATen 依赖**：
- 立即依赖：
  - `aten::channel_shuffle`
- math / CUDA 展开依赖：
  - `aten::view`
  - `aten::permute`
  - `aten::contiguous`
  - `aten::reshape`

**备注**：
- CPU 前向常走 `native_channel_shuffle` 专用 kernel；但 backward 公式层面仍是“再次 channel shuffle”。

---

### 165. torch.nn.modules.flatten.Flatten

**反向来源类型**：CIA 前向分解为 reshape / view

**Forward 路径**（默认连续输入）：
`aten::flatten.using_ints` → `native::flatten` → `reshape_symint` → `view_symint`

`native::flatten`（`TensorShape.cpp`）：
```cpp
return native::reshape_symint(self, shape);
```

而 `reshape_symint` 对连续 tensor 直接：
```cpp
if (self.is_contiguous_or_false() && !self.is_mkldnn()) {
  return self.view_symint(proposed_shape);
}
```

**生成代码**：
- 连续输入路径最终由 `aten::view` 生成 `ViewBackward0`

**反向 ATen 依赖**：
- `as_strided_backward`

**备注**：
- 若输入不是连续内存，`reshape_symint` 可能改走 `_reshape_alias` 或 `_unsafe_view`，但对应正向文档锁定的是“连续内存输入”常见路径，因此这里以 `view` backward 为准。
