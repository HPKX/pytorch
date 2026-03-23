# torch API 61-65 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.isnan` | 密集 CPU/CUDA/MPS 张量 | `torch.isnan(x)` → `aten::isnan` → dispatch 到 CPU/CUDA/MPS/MTIA 内核；native 函数体实际为 `self != self`，即前向继续进入 `aten::ne.Tensor` | `aten::isnan`, `aten::ne.Tensor` |
| `torch.isnan` | Sparse 张量 | `torch.isnan(x)` → `aten::isnan` → 同一 schema 分发到 Sparse / SparseCsr 专用实现；这里没有额外公开的前向 ATen hop | `aten::isnan` |
| `torch.isnan` | autograd | `aten::isnan` → derivatives.yaml 标记为 `non_differentiable`，不生成反向传播逻辑 | `aten::isnan` |
| `torch.isposinf` | 密集 CPU/CUDA/MPS 张量 | `torch.isposinf(x)` → `aten::isposinf` → CPU/CUDA/MPS structured functional wrapper（`meta + impl`）→ `structured_isposinf_out::impl`：整型直接写 false，浮点走 `isposinf_stub` 到各后端 kernel | `aten::isposinf` |
| `torch.isposinf` | Sparse 张量 | `torch.isposinf(x)` → `aten::isposinf` → 同一 schema 分发到 Sparse / SparseCsr 特化实现；无额外公开的前向 ATen hop | `aten::isposinf` |
| `torch.isposinf` | autograd | `aten::isposinf` → derivatives.yaml 中无条目；结果张量不参与梯度计算 | `aten::isposinf` |
| `torch.isreal` | 所有场景（CompositeImplicitAutograd） | `torch.isreal(x)` → `aten::isreal` → 无 dispatch key，CompositeImplicitAutograd：整型/浮点返回 `aten::ones_like(..., bool)`；复数路径为 `aten::imag(self)` → `aten::eq.Scalar(..., 0)` | `aten::isreal`, `aten::ones_like`, `aten::imag`, `aten::eq.Scalar` |
| `torch.kaiser_window` | 基础重载 (window_length) | `torch.kaiser_window(L)` → `aten::kaiser_window` (CompositeExplicitAutograd) → native 重载补默认参数 `periodic=true, beta=12.0` → 最终实现：`device==Meta` 或 `L==0` 走 `aten::empty`，`L==1` 走 `aten::ones`，其余路径为 `aten::arange` → `kaiser_window_stub`；若 `periodic` 则末尾 `aten::narrow` | `aten::kaiser_window`, `aten::empty`, `aten::ones`, `aten::arange`, `aten::narrow` |
| `torch.kaiser_window` | periodic 重载 | `torch.kaiser_window(L, periodic)` → `aten::kaiser_window.periodic` (CompositeExplicitAutograd) → native 重载补默认参数 `beta=12.0`，随后走同一最终实现：`aten::empty` / `aten::ones` / `aten::arange` / `kaiser_window_stub` / `aten::narrow` | `aten::kaiser_window.periodic`, `aten::empty`, `aten::ones`, `aten::arange`, `aten::narrow` |
| `torch.kaiser_window` | beta 重载（最终实现） | `torch.kaiser_window(L, periodic, beta)` → `aten::kaiser_window.beta` (CompositeExplicitAutograd) → `aten::empty` / `aten::ones` 处理短路场景；其余路径 `aten::arange` 生成输入、`aten::empty` 分配输出，`TensorIterator` 交给 `kaiser_window_stub`，最后按需 `aten::narrow` | `aten::kaiser_window.beta`, `aten::empty`, `aten::ones`, `aten::arange`, `aten::narrow` |
| `torch.kron` | 密集张量（CompositeImplicitAutograd） | `torch.kron(a, b)` → `aten::kron` → 无 dispatch key，CompositeImplicitAutograd → `KronImpl(self, other).kron()`：`aten::_unsafe_view(self, a_reshape)` + `aten::_unsafe_view(other, b_reshape)` → `aten::mul.Tensor` → `aten::_unsafe_view(result, result_reshape)` | `aten::kron`, `aten::_unsafe_view`, `aten::mul.Tensor` |
| `torch.kron` | out 变体（CompositeImplicitAutograd） | `torch.kron.out(a, b, out=result)` → `aten::kron.out` → `KronImpl(self, other).kron_out(result)`：对结果张量做内部 resize 后 `aten::_unsafe_view(result, mul_shape)` → `aten::mul.out` | `aten::kron.out`, `aten::_unsafe_view`, `aten::mul.out` |

## 备注

1. **`torch.isnan`**：YAML 中对 CPU/CUDA/MPS/MTIA 有显式 dispatch key，dense native 实现 `isnan()` 的函数体是 `return self != self`，因此会继续进入 `aten::ne.Tensor`。Sparse / SparseCsr 走同一 schema 的特化分支。derivatives.yaml 将其标记为 `non_differentiable`；`autogen: isnan.out` 仅表示 out 变体由代码生成。

2. **`torch.isposinf`**：使用 structured kernel 机制。`structured_delegate: isposinf.out` 只是 codegen 关系，运行时 functional 变体有自己的 wrapper，直接执行 `meta + impl`；dense 实现对整型直接填 false，浮点类型走 `isposinf_stub`。Sparse / SparseCsr 仍然是同一 `aten::isposinf` schema 的分发分支。derivatives.yaml 中无条目。

3. **`torch.isreal`**：YAML 中无 dispatch key，为 CompositeImplicitAutograd。对整型/浮点直接返回全 true 的 bool 张量（`aten::ones_like`），对复数类型则走 `aten::imag` + `aten::eq.Scalar`。

4. **`torch.kaiser_window`**：三个重载均为 CompositeExplicitAutograd。前两个 native 重载只是在 C++ 层补默认参数；真正的前向 ATen hop 仍是各自入口 schema 加上最终实现中的 `aten::empty` / `aten::ones` / `aten::arange` / `aten::narrow`。`kaiser_window_stub` 是 backend 分发阶段，不属于额外 ATen schema。

5. **`torch.kron`**：YAML 中无 dispatch key，为 CompositeImplicitAutograd。其前向核心是 `_unsafe_view` + `mul` 的组合；out 变体会在进入 `aten::mul.out` 之前对输出张量做内部 resize，但那不是公开的 ATen schema。derivatives.yaml 中无 kron 条目，梯度由组合操作自动推导。
