# torch API 21-25 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | `torch.column_stack` | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无条目 | 前向分解产生的各 op 各自反向：`aten::reshape` 反向为 `aten::reshape`；`aten::cat` 反向为各 slice 的 `aten::narrow`/`aten::slice.Tensor` | CIA；分解为 `aten::reshape`（1D 输入升维）→ `aten::hstack` → `aten::atleast_1d.Sequence` → `aten::cat` |
| 22 | `torch.combinations` | CIA 前向分解，各子 op 独立 backward | 各子 op 独立：`MaskedSelectBackward0`、`StackBackward0` 等 | `masked_select`: `self: masked_select_backward(grad, self, mask)`；`stack`: `tensors: stack_tensors_backward(grad, dim, ...)` | `aten::masked_select_backward`（→ `aten::zeros_like`、`aten::expand`、`aten::masked_scatter_`）；`aten::select.int` | CIA；分解到 `aten::meshgrid`、`aten::masked_select`、`aten::stack`；梯度主要来自 `masked_select` 和 `stack` |
| 23 | `torch.cond` | HigherOrderOperator，Python 层 autograd | `CondAutogradOp`（`torch.autograd.Function`） | 无 derivatives.yaml 条目（非 ATen op） | 取决于 true_fn/false_fn 内部的算子 | 不是传统 ATen 算子；autograd 通过 `CondAutogradOp.apply` 在 Python 层实现，反向时递归调用 `cond_op` 对 backward 函数求值 |
| 24 | `torch.conj` | CIA 前向分解 → `_conj` 的 inline 公式 | `ConjBackward0`（挂在 `_conj` 上） | `_conj`: `self: grad.conj()` | `aten::conj`（实数时为 no-op） | `conj` 是 CIA→ `_conj`（复数）或返回 self（实数）；`_conj` 的反向为 `grad.conj()`，形成对称的共轭关系 |
| 25 | `torch.copysign` | helper 函数 | `CopysignBackward0`（Tensor 重载）/ `CopysignBackward1`（Scalar 重载） | Tensor: `self: copysign_tensor_self_backward(grad, self, result)`、`other: zeros_like(other)`；Scalar: `self: copysign_tensor_self_backward(grad, self, result)` | `aten::div.Tensor`、`aten::masked_fill_.Scalar`、`aten::eq.Scalar`、`aten::mul.Tensor`；Tensor 重载额外 `aten::zeros_like` | `copysign_tensor_self_backward` 是手写 helper（`FunctionsManual.cpp:103`），计算 `grad * result / self` 并对 `self==0` 处梯度置零 |

## 详细分析

### 21. torch.column_stack

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**（`TensorShape.cpp:3627`）：
```
column_stack(tensors)
  → 对每个 dim <= 1 的输入: self.reshape({numel, 1})  // aten::reshape
  → hstack(reshaped_tensors)                           // aten::hstack
    → atleast_1d.Sequence(tensors)                      // aten::atleast_1d.Sequence
    → cat(result, dim=1)                                // aten::cat
```

**无 derivatives.yaml 条目**。

**各子 op 反向**：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `reshape`/`view` | `self: grad.reshape_symint(self.sym_sizes())` | `aten::reshape` |
| `cat` | `tensors: cat_tensors_backward(grad, ...)` | `aten::narrow`/`aten::slice.Tensor`（对各输入按 dim 切分 grad） |

反向本质是 `cat` 的反向（沿 dim=1 切分 grad），加上各 `reshape` 的反向（恢复原 shape）。

---

### 22. torch.combinations

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，各子 op 独立 backward

**Forward 分解路径**（`Itertools.cpp:60`）：
```
combinations(self, r, with_replacement)
  → r==0 时: at::empty({0})                             // 直接返回
  → 否则:
    → meshgrid(self 的 r 份拷贝)                         // aten::meshgrid
    → _triu_mask 构造上三角掩码                           // aten::arange, aten::meshgrid, aten::full
    → 对每个 grid: masked_select(grid, mask)             // aten::masked_select
    → stack(selected_tensors)                            // aten::stack
```

**无 `combinations` 自身的 derivatives.yaml 条目**。各子 op 独立注册 backward：

**`masked_select` 的 backward**：
```yaml
- name: masked_select(Tensor self, Tensor mask) -> Tensor
  self: masked_select_backward(grad, self, mask)
  mask: non_differentiable
```

**`masked_select_backward` 实现**（`TensorAdvancedIndexing.cpp:2624`）：
```cpp
Tensor masked_select_backward(const Tensor& grad, const Tensor& input, const Tensor& mask) {
  auto result = at::zeros_like(
      input.expand(at::infer_size(input.sizes(), mask.sizes())),
      at::MemoryFormat::Preserve);                     // aten::zeros_like, aten::expand
  result.masked_scatter_(mask, grad);                   // aten::masked_scatter_
  return result;
}
```

**`stack` 的 backward**：
```yaml
- name: stack(Tensor[] tensors, int dim=0) -> Tensor
  tensors: stack_tensors_backward(grad, dim, to_args_scalartypes(tensors))
```

**`stack_tensors_backward` 实现**（`FunctionsManual.cpp:1098`）：
```cpp
std::vector<Tensor> stack_tensors_backward(const Tensor& grad, int64_t dim, ...) {
  for (const auto i : c10::irange(dtypes.size())) {
    auto gr = grad.select(dim, static_cast<int64_t>(i));  // aten::select.int
    if (grad_is_complex && !at::isComplexType(dtypes[i])) {
      gr = at::real(gr);                                   // aten::real（条件）
    }
    grad_inputs[i] = gr;
  }
}
```

**反向 ATen 依赖（汇总）**：
- `masked_select` 反向：`aten::zeros_like`、`aten::expand`、`aten::masked_scatter_`
- `stack` 反向：`aten::select.int`
- 条件依赖：`aten::real`（复数→实数）

---

### 23. torch.cond

**反向来源类型**：HigherOrderOperator，Python 层 autograd

**`torch.cond` 不是传统 ATen 算子**，在 `native_functions.yaml` 中无定义。它是 Python 层的 `HigherOrderOperator`（`torch/_higher_order_ops/cond.py`）。

**Autograd 实现**（`cond.py:288`）：
```python
class CondAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, true_fn, false_fn, *operands):
        ctx._pred = pred
        ctx._true_bw_fn = create_bw_fn(true_fn, operands)
        ctx._false_bw_fn = create_bw_fn(false_fn, operands)
        save_tensors_and_symints_for_backward(ctx, operands)
        with torch._C._AutoDispatchBelowAutograd():
            return cond_op(pred, true_fn, false_fn, operands)

    @staticmethod
    def backward(ctx, *flat_grads):
        # 反向时再次调用 cond_op，根据 pred 选择执行 true_bw_fn 或 false_bw_fn
        return (None, None, None) + tuple(
            cond_op(ctx._pred, ctx._true_bw_fn, ctx._false_bw_fn, flat_grads + saved))
```

反向时的 ATen 依赖完全取决于 `true_fn`/`false_fn` 内部使用的算子，无法静态确定。

**反向 ATen 依赖**：取决于用户提供的分支函数，无固定依赖。

---

### 24. torch.conj

**反向来源类型**：CIA 前向分解 → `_conj` 的 inline 公式

**Forward 分解路径**（`UnaryOps.cpp:660`）：
- 非复数 tensor：`conj` → `self`（直接返回，alias-like），不产生 autograd 节点
- 稠密复数 tensor：`conj` → `self.conj()` → `_conj` → 创建 alias 并翻转 conj bit
- 稀疏复数 tensor：`conj` → `conj_physical` → `_conj_physical`

**`_conj` 的 derivatives.yaml 条目**：
```yaml
- name: _conj(Tensor(a) self) -> Tensor(a)
  self: grad.conj()
  result: self_t.conj()
```

**Backward Node**：`ConjBackward0`

**生成代码**（`Functions.cpp` `ConjBackward0`）：
```cpp
auto grad_result = any_grad_defined ? (grad.conj()) : Tensor();
```

**反向 ATen 依赖**：
- `aten::conj` — `grad.conj()`（实数时为 no-op view，复数时设置 conj bit）

`conj` 的反向与前向形成对称：前向做共轭，反向也做共轭。对实数 tensor，`conj` 和其反向均为 identity（no-op view）。

**注意**：`conj` 本身是 CIA + `manual_cpp_binding`，在 `derivatives.yaml` 中无条目。autograd 录制在内层 `_conj` 上。

---

### 25. torch.copysign

**反向来源类型**：helper 函数

**Forward 路径**：
- Tensor 重载：`aten::copysign.Tensor` → `structured_delegate: copysign.out` → CPU/CUDA/MPS: `copysign_out` backend kernel
- Scalar 重载：`aten::copysign.Scalar` → `CompositeExplicitAutograd` → `at::native::copysign` → 将 Scalar 包为 wrapped tensor → 重入 `aten::copysign.Tensor`

**derivatives.yaml 条目**：
```yaml
- name: copysign.Tensor(Tensor self, Tensor other) -> Tensor
  self: copysign_tensor_self_backward(grad, self, result)
  other: zeros_like(other)

- name: copysign.Scalar(Tensor self, Scalar other) -> Tensor
  self: copysign_tensor_self_backward(grad, self, result)
```

**Backward Node**：`CopysignBackward0`（Tensor 重载）/ `CopysignBackward1`（Scalar 重载）

**`copysign_tensor_self_backward` 实现**（`FunctionsManual.cpp:103`）：
```cpp
Tensor copysign_tensor_self_backward(
    const Tensor& grad, const Tensor& self, const Tensor& result) {
  auto ratio = result / self;                 // aten::div.Tensor
  ratio.masked_fill_(self == 0, 0);           // aten::eq.Scalar, aten::masked_fill_.Scalar
  return grad * ratio;                        // aten::mul.Tensor
}
```

**数学含义**：`copysign(self, other) = |self| * sign(other)`，对 self 的梯度为 `grad * sign(other) * sign(self) = grad * result / self`（self≠0 时），self==0 时梯度为 0。

**反向 ATen 依赖（Tensor 重载）**：
- `aten::div.Tensor` — `result / self`
- `aten::eq.Scalar` — `self == 0`
- `aten::masked_fill_.Scalar` — 对 self==0 处置零
- `aten::mul.Tensor` — `grad * ratio`
- `aten::zeros_like` — other 的梯度为全零

**反向 ATen 依赖（Scalar 重载）**：
- `aten::div.Tensor` — `result / self`
- `aten::eq.Scalar` — `self == 0`
- `aten::masked_fill_.Scalar` — 对 self==0 处置零
- `aten::mul.Tensor` — `grad * ratio`
