# torch API 36-40 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 36 | `torch.distribution.uniform.Uniform` | 纯 Python 分布类，各子 op 独立 autograd | 无统一 backward node | 无统一 derivatives.yaml 条目 | `aten::mul.Tensor`、`aten::neg`、`aten::div.Tensor` | 纯 Python 分布类；`rand`/`uniform_` 采样不可微，梯度通过重参数化路径 `low + rand * (high - low)` 传播至 `low`/`high` |
| 37 | `torch.dsplit` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::zeros`、`aten::slice.Tensor`、`aten::copy_` | CIA 分解到 `tensor_split` / `slice` 后，递归展开最终落到 `slice_backward` 的基础 ATen op |
| 38 | `torch.dstack` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::narrow`、`aten::slice.Tensor`、`aten::zeros`、`aten::real`、`aten::squeeze.dim`、`aten::reshape` | `cat` 负责主梯度切分，`atleast_3d` 引入的 `unsqueeze`/`reshape` view 反向分别对应 `squeeze.dim`/`reshape`；复数到实数时条件性经过 `real` |
| 39 | `torch.fliplr` | CIA 前向分解 → `aten::flip` 的 inline 公式 | `FlipBackward0`（挂在 `flip` 上） | `self: grad.flip(dims)` | `aten::flip` | CIA；`fliplr` 分解为 `self.flip({1})`；backward 挂在 `flip` 上 |
| 40 | `torch.flipud` | CIA 前向分解 → `aten::flip` 的 inline 公式 | `FlipBackward0`（挂在 `flip` 上） | `self: grad.flip(dims)` | `aten::flip` | CIA；`flipud` 分解为 `self.flip({0})`；backward 挂在 `flip` 上 |

## 详细分析

### 36. torch.distribution.uniform.Uniform

**反向来源类型**：纯 Python 分布类，各子 op 独立 autograd，无统一 backward node

**Forward 路径**（`rsample()`）：
`Uniform.rsample` → `torch.rand(shape)` → `aten::rand` → `aten::uniform_`（RNG kernel） → 然后 Python 层: `self.low + rand * (self.high - self.low)` → `aten::sub` → `aten::mul` → `aten::add`

Uniform 重参数化技巧：`sample = low + rand * (high - low)`，其中 `rand ~ Uniform(0, 1)`。

**derivatives.yaml 条目**（各子 op 独立）：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `rand` / `uniform_` | 不可微（工厂/采样函数） | 无有效梯度 |
| `sub.Tensor`（`high - low`） | `self: grad`；`other: -grad` | `aten::neg` |
| `mul.Tensor`（`rand * diff`） | `self: grad * other`；`other: grad * self` | `aten::mul.Tensor` |
| `add.Tensor`（`low + ...`） | `self: grad`；`other: grad * alpha` | 无额外 op（alpha=1 时直接传递 grad） |

梯度通过重参数化路径传播：`rand` 本身无梯度，但 `low` 通过 `add` 获得梯度，`high` 和 `low` 通过 `sub` 和 `mul` 获得梯度。

**`log_prob()` 路径**：
`aten::le.Tensor` → `aten::gt.Tensor` → `aten::mul.Tensor` → `aten::log` → `aten::sub.Tensor`
其中 `le`/`gt` 输出 bool 不可微（`output_differentiability: [False]`），`log` 和 `sub` 有独立 backward。

---

### 37. torch.dsplit

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- `aten::dsplit.int` → CIA → `at::native::dsplit` → `at::tensor_split(self, sections, 2)` → CIA → `tensor_split_sections_symint` → 循环调用 `at::slice_symint(self, dim, start, end)` → `aten::slice.Tensor`
- `aten::dsplit.array` → CIA → `at::native::dsplit` → `at::tensor_split(self, indices, 2)` → CIA → `tensor_split_indices_symint` → 循环调用 `at::slice_symint` → `aten::slice.Tensor`

**无 derivatives.yaml 条目**（`dsplit`、`tensor_split` 均无条目）。autograd 引擎穿透到底层 `slice.Tensor` 视图操作。

`slice.Tensor` 的 derivatives.yaml 条目：
```yaml
- name: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)
  result: auto_linear
```

`slice_backward_wrapper` 内部创建 `aten::zeros` 并将 grad 通过 `aten::slice.Tensor` + `aten::copy_` 填入对应位置。

**反向 ATen 依赖**（来自 `slice` 的 backward）：
- `aten::zeros` — 创建全零张量
- `aten::slice.Tensor` — 创建 slice 视图
- `aten::copy_` — 将 grad 填入对应 slice

---

### 38. torch.dstack

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
`aten::dstack` → CIA → `at::native::dstack` → `aten::atleast_3d`（CIA → 对 0-D/1-D/2-D 输入做 `unsqueeze`/`reshape`）→ `aten::cat`（dim=2）→ structured backend kernel

**无 `dstack` 的 derivatives.yaml 条目**。autograd 穿透到 `cat` 和各 `unsqueeze`/`reshape` 子 op。

`cat` 的 derivatives.yaml 条目：
```yaml
- name: cat(Tensor[] tensors, int dim=0) -> Tensor
  tensors: cat_tensors_backward(grad, to_args_sizes_symint(tensors), to_args_scalartypes(tensors), dim)
  result: cat_jvp(tensors, dim)
```

**`cat_tensors_backward` 实现**（`FunctionsManual.cpp:1058`）：
```cpp
std::vector<Tensor> cat_tensors_backward(const Tensor& grad,
    const std::vector<std::vector<c10::SymInt>>& sizes,
    const std::vector<ScalarType>& dtypes, int64_t dim) {
  // ...
  for (const auto i : c10::irange(sizes.size())) {
    // 空输入 → zeros({0})
    grad_inputs[i] = grad_val.narrow_symint(dim, accumulate - size, size);  // aten::narrow
  }
  return grad_inputs;
}
```

**反向 ATen 依赖**（来自 `cat` 的 backward）：
- `aten::narrow`（→ `aten::slice.Tensor`）— 从 grad 中按 dim 切分出各输入的梯度
- `aten::zeros` — 空输入返回空梯度

**条件依赖**：
- `aten::real` — 复数 grad + 实数输入场景

**`atleast_3d` 内部 view op 的 backward**（与 `hstack`/`vstack` 对齐）：
- `aten::squeeze.dim` — 来自 `unsqueeze` 的 backward（1-D/2-D 输入路径）
- `aten::reshape` — 来自 `reshape({1,1,1})` 的 backward（0-D 输入路径）

---

### 39. torch.fliplr

**反向来源类型**：`CompositeImplicitAutograd` 前向分解 → `aten::flip` 的 inline 公式

**Forward 分解路径**：
`aten::fliplr` → CIA → `at::native::fliplr` → `self.flip({1})` → `aten::flip` → CPU/CUDA/MPS backend kernel

**无 `fliplr` 的 derivatives.yaml 条目**。autograd 穿透到 `flip`。

`flip` 的 derivatives.yaml 条目：
```yaml
- name: flip(Tensor self, int[] dims) -> Tensor
  self: grad.flip(dims)
  result: auto_linear
```

**Backward Node**：`FlipBackward0`（挂在 `flip` 上）

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = any_grad_defined ? (grad.flip(dims)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::flip` — `grad.flip({1})`；flip 是自身的逆操作

---

### 40. torch.flipud

**反向来源类型**：`CompositeImplicitAutograd` 前向分解 → `aten::flip` 的 inline 公式

**Forward 分解路径**：
`aten::flipud` → CIA → `at::native::flipud` → `self.flip({0})` → `aten::flip` → CPU/CUDA/MPS backend kernel

**无 `flipud` 的 derivatives.yaml 条目**。autograd 穿透到 `flip`。

`flip` 的 derivatives.yaml 条目：
```yaml
- name: flip(Tensor self, int[] dims) -> Tensor
  self: grad.flip(dims)
  result: auto_linear
```

**Backward Node**：`FlipBackward0`（挂在 `flip` 上）

**生成代码**（`Functions.cpp`）：
```cpp
auto grad_result = any_grad_defined ? (grad.flip(dims)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::flip` — `grad.flip({0})`；flip 是自身的逆操作
