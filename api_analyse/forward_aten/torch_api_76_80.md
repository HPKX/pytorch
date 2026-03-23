# torch API 76-80 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.lu_unpack` | 对 LU 分解结果进行拆包，提取 P、L、U 矩阵 | `torch.lu_unpack` → `aten::lu_unpack` → CPU/CUDA/MPS structured functional wrapper（`meta + impl`）→ `lu_unpack_out`：若 `unpack_data=True`，调用 `aten::triu.out`、`aten::tril.out`、`aten::diagonal`、`aten::fill_.Scalar`；若 `unpack_pivots=True`，调用 `aten::arange` → `aten::expand` → `aten::contiguous`，随后进入 `unpack_pivots_stub`，最后 `aten::zero_` + `aten::unsqueeze` + `aten::scatter_.value` 构造置换矩阵 | `aten::lu_unpack`, `aten::triu.out`, `aten::tril.out`, `aten::diagonal`, `aten::fill_.Scalar`, `aten::arange`, `aten::expand`, `aten::contiguous`, `aten::zero_`, `aten::unsqueeze`, `aten::scatter_.value` |
| `torch.moveaxis` | 将张量指定维度移动到目标位置（`movedim` 的别名） | `torch.moveaxis` → `aten::moveaxis` (CompositeImplicitAutograd) → 直接转到 `aten::movedim` → 计算排列顺序 → `aten::permute` | `aten::moveaxis`, `aten::movedim`, `aten::permute` |
| `torch.movedim` | 将张量指定维度移动到目标位置 | `torch.movedim` → `aten::movedim` (CompositeImplicitAutograd) → 计算维度排列顺序 → `aten::permute` | `aten::movedim`, `aten::permute` |
| `torch.msort` | 沿第 0 维对张量排序 | `torch.msort` → `aten::msort` (CompositeImplicitAutograd) → `std::get<0>(aten::sort(self, 0, false))`；该 3 参 `sort` 实现再转到 `aten::sort.stable(self, stable=false, dim=0, descending=false)`，随后进入 CPU/CUDA/MPS structured wrapper（`meta + impl`）并落到 `sort_stable_out` / `sort_stable_out_mps` | `aten::msort`, `aten::sort`, `aten::sort.stable` |
| `torch.multiply` | 逐元素乘法（`mul` 的别名） | `torch.multiply` 的 Tensor 重载：`aten::multiply.Tensor` (CompositeImplicitAutograd) → `native::multiply` → `self.mul(other)` → `aten::mul.Tensor` → structured wrapper（`meta + impl`）→ backend kernel；Scalar 重载：`aten::multiply.Scalar` → `native::multiply` → `self.mul(other)` → `aten::mul.Scalar` | `aten::multiply.Tensor`, `aten::mul.Tensor`, `aten::multiply.Scalar`, `aten::mul.Scalar` |

## 备注

1. **`torch.lu_unpack`**: 使用 structured 机制，但 `structured_delegate: lu_unpack.out` 只是 codegen 关系；运行时 functional 变体有自己的 wrapper，直接执行 `meta + impl`。前向 ATen hop 主要是 `aten::triu.out` / `aten::tril.out` / `aten::diagonal` / `aten::fill_` 以及 pivot 路径上的 `aten::arange` / `aten::expand` / `aten::contiguous` / `aten::zero_` / `aten::unsqueeze` / `aten::scatter_`。`unpack_pivots_stub` 是 backend 分发阶段。derivatives.yaml 中定义了反向传播：P 不可微，L 和 U 可微。

2. **`torch.moveaxis`**: 是 `movedim` 的完全别名，YAML 中无 dispatch key，属于 CompositeImplicitAutograd。两个重载（intlist 和 int）均直接转到对应的 `aten::movedim`，autograd 通过 `aten::permute` 的导数自动传播。

3. **`torch.movedim`**: CompositeImplicitAutograd，核心逻辑是计算一个维度排列数组然后调用 `aten::permute`。int 单参数重载会包装为 IntArrayRef 后调用 intlist 重载。

4. **`torch.msort`**: CompositeImplicitAutograd，是 `aten::sort(self, dim=0, descending=false)` 的简单封装。`msort.out` 变体内部调用 `aten::sort_out`。`sort` 的 3 参实现会再转到 `aten::sort.stable(..., stable=false, ...)`；`structured_delegate: sort.values_stable` 是 codegen 关系，不应单独算作额外 runtime hop。

5. **`torch.multiply`**: 是 `mul` 的完全别名。Tensor 重载走 `aten::multiply.Tensor` → `aten::mul.Tensor`；Scalar 重载走 `aten::multiply.Scalar` → `aten::mul.Scalar`。`mul.out` 只是在 Tensor 重载的 backend wrapper 内部对应的 structured 实现，不应单独写成 `torch.multiply` 的前向依赖。
