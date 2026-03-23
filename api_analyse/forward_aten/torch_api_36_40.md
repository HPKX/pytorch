# torch API 36-40 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.distribution.uniform.Uniform` | rsample (核心采样) | Python 层: `torch.rand(shape)` → `aten::rand` → CompositeExplicitAutograd → `at::native::rand` → `aten::uniform_(result, 0, 1)` → CPU/CUDA key → `uniform_` 内核; 然后 `self.low + rand * (self.high - self.low)` 触发 `aten::mul`, `aten::sub`, `aten::add` | `aten::rand`, `aten::uniform_`, `aten::mul`, `aten::sub`, `aten::add` |
| `torch.distribution.uniform.Uniform` | log_prob | Python 层: `self.low.le(value)` → `aten::le.Tensor`; `self.high.gt(value)` → `aten::gt.Tensor`; `lb.mul(ub)` → `aten::mul.Tensor`; `torch.log(...)` → `aten::log`; `self.high - self.low` → `aten::sub.Tensor` | `aten::le.Tensor`, `aten::gt.Tensor`, `aten::mul.Tensor`, `aten::log`, `aten::sub.Tensor` |
| `torch.distribution.uniform.Uniform` | cdf | Python 层: `(value - self.low) / (self.high - self.low)` → `aten::sub.Tensor`, `aten::div.Tensor`; `.clamp(min=0, max=1)` → `aten::clamp` | `aten::sub.Tensor`, `aten::div.Tensor`, `aten::clamp` |
| `torch.distribution.uniform.Uniform` | entropy | Python 层: `torch.log(self.high - self.low)` → `aten::sub.Tensor` → `aten::log` | `aten::sub.Tensor`, `aten::log` |
| `torch.dsplit` | int sections | `aten::dsplit.int` → CompositeImplicitAutograd → `at::native::dsplit` → `at::tensor_split(self, split_size, 2)` → `aten::tensor_split.sections` → CompositeImplicitAutograd → `tensor_split_sections_symint` → 循环调用 `at::slice_symint` → `aten::slice.Tensor` | `aten::dsplit.int`, `aten::tensor_split.sections`, `aten::slice.Tensor` |
| `torch.dsplit` | indices 列表 | `aten::dsplit.array` → CompositeImplicitAutograd → `at::native::dsplit` → `at::tensor_split(self, split_sizes, 2)` → `aten::tensor_split.indices` → CompositeImplicitAutograd → `tensor_split_indices_symint` → 循环调用 `at::slice_symint` → `aten::slice.Tensor` | `aten::dsplit.array`, `aten::tensor_split.indices`, `aten::slice.Tensor` |
| `torch.dstack` | 默认 (functional) | `aten::dstack` → `CompositeImplicitAutograd` → `at::native::dstack` → `aten::atleast_3d` → `CompositeImplicitAutograd` → `at::native::atleast_3d` → `aten::cat` → `CPU/CUDA/MPS` structured wrapper（`precompute + op.meta() + op.impl()`）→ `cat_out_cpu` / `cat_out_cuda` / `cat_out_mps` | `aten::dstack`, `aten::atleast_3d`, `aten::cat` |
| `torch.dstack` | out 变体 | `aten::dstack.out` → `CompositeImplicitAutograd` → `at::native::dstack_out` → `aten::atleast_3d` → `CompositeImplicitAutograd` → `at::native::atleast_3d` → `aten::cat.out` → `CPU/CUDA/MPS` structured wrapper（`precompute + op.meta() + op.impl()`）→ `cat_out_cpu` / `cat_out_cuda` / `cat_out_mps` | `aten::dstack.out`, `aten::atleast_3d`, `aten::cat.out` |
| `torch.fliplr` | 默认 (dim >= 2) | `aten::fliplr` → CompositeImplicitAutograd → `at::native::fliplr` → `self.flip({1})` → `aten::flip` → CPU/CUDA key → `flip` 内核 | `aten::fliplr`, `aten::flip` |
| `torch.flipud` | 默认 (dim >= 1) | `aten::flipud` → CompositeImplicitAutograd → `at::native::flipud` → `self.flip({0})` → `aten::flip` → CPU/CUDA key → `flip` 内核 | `aten::flipud`, `aten::flip` |

**备注:**
- `torch.distribution.uniform.Uniform`: 这里按 [torch_api.md](/Users/hz/Desktop/Code/pytorch-codes/pytorch/api_analyse/torch_api.md) 的标准接口名保持为单数 `distribution`；其实现本质上仍是纯 Python 分布类组合多个基础 ATen 算子。
- `torch.dsplit`: CompositeImplicitAutograd，无 dispatch key 也无 derivatives.yaml 条目，autograd 穿透至 `tensor_split`（同为 CompositeImplicitAutograd），最终穿透至 `narrow`/`slice` 等基础视图操作
- `torch.dstack`: CompositeImplicitAutograd，无 derivatives.yaml 条目，autograd 穿透至 `atleast_3d` 和 `cat`。`cat` 虽然在 YAML 里有 `structured_delegate: cat.out`，但这只是 codegen 关系；运行时 `aten::cat` 直接进入 backend structured wrapper，再执行 `precompute + meta + impl`。
- `torch.fliplr` / `torch.flipud`: CompositeImplicitAutograd，无 derivatives.yaml 条目，autograd 穿透至 `flip`。`flip` 在 derivatives.yaml 中有梯度定义 (`grad.flip(dims)`)，有 CPU/CUDA/MPS 独立 dispatch 内核
