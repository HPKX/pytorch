# torch API 16-20 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | `torch.block_diag` | helper 函数 | `BlockDiagBackward0` | `tensors: block_diag_backward(grad, to_args_sizes(tensors), to_args_scalartypes(tensors))` | `aten::slice.Tensor`、`aten::squeeze.dim`、`aten::zeros`；复数→实数时额外 `aten::real` | `block_diag_backward` 是手写 helper（`FunctionsManual.cpp:1117`），通过 slice 从 grad 中切出各输入对应的对角块 |
| 17 | `torch.bucketize` | 不可微 | 无 | 无条目 | 无 | 返回整数索引张量；autograd wrapper 不建立梯度图（无 backward node，输出为 fresh tensor 无 autograd 历史） |
| 18 | `torch.cholesky` | helper 函数 | `CholeskyBackward0` | `self: cholesky_backward(grad, upper, result)` | `aten::mH`（view）、`aten::matmul`、`aten::tril`、`aten::mul.Scalar`、`aten::add.Tensor`、`aten::linalg_solve_triangular` | `cholesky_backward` 是手写 helper（`FunctionsManual.cpp:1983`）；`cholesky` 已 deprecated，推荐 `torch.linalg.cholesky` |
| 19 | `torch.cholesky_solve` | helper 函数 | `CholeskySolveBackward0` | `self, input2: cholesky_solve_backward(grad, self, input2, result, upper, grad_input_mask)` | `aten::cholesky_solve`、`aten::matmul`、`aten::mH`（view）、`aten::add.Tensor`、`aten::neg` | `cholesky_solve_backward` 是手写 helper（`FunctionsManual.cpp:4599`）；self 梯度通过递归调用 `cholesky_solve` 求解 |
| 20 | `torch.clip` | CIA 前向分解 → `aten::clamp` 的 helper 函数 | `ClampBackward0`（Scalar 重载）/ `ClampBackward1`（Tensor 重载） | Scalar: `self: clamp_backward(grad, self, min, max)`；Tensor: `self: clamp_backward(grad, self, min, max)` + `min, max: clamp_backward_min_max(...)` | `aten::scalar_tensor`、`aten::ge.Scalar`/`aten::ge.Tensor`、`aten::le.Scalar`/`aten::le.Tensor`、`aten::logical_and_`、`aten::where` | `clip` 是 CIA 别名→ `aten::clamp`；backward 挂在 `clamp` 上；`clamp_backward` 手写 helper（`FunctionsManual.cpp:1171`） |

## 详细分析

### 16. torch.block_diag

**反向来源类型**：helper 函数

**Forward 路径**：`aten::block_diag` → `CompositeExplicitAutograd` → `at::native::block_diag` → `aten::zeros` → 循环中 `aten::slice.Tensor` → `aten::copy_`

**derivatives.yaml 条目**：
```yaml
- name: block_diag(Tensor[] tensors) -> Tensor
  tensors: block_diag_backward(grad, to_args_sizes(tensors), to_args_scalartypes(tensors))
  result: block_diag_jvp(tensors)
```

**Backward Node**：`BlockDiagBackward0`

**`block_diag_backward` 实现**（`FunctionsManual.cpp:1117`）：
```cpp
std::vector<Tensor> block_diag_backward(
    const Tensor& grad,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<ScalarType>& dtypes) {
  std::vector<Tensor> grad_inputs(sizes.size());
  if (!grad.defined()) return grad_inputs;

  Tensor real_view_of_grad;
  bool grad_is_complex = grad.is_complex();
  if (grad_is_complex) {
    real_view_of_grad = at::real(grad);           // aten::real
  }

  int64_t cur_dim0 = 0, cur_dim1 = 0;
  for (const auto i : c10::irange(sizes.size())) {
    Tensor grad_val = (!at::isComplexType(dtypes[i]) && grad_is_complex)
        ? real_view_of_grad : grad;
    auto& shape = sizes[i];
    if (shape.size() == 1 && shape[0] == 0) {
      grad_inputs[i] = at::zeros({0}, grad_val.options());  // aten::zeros
      continue;
    }
    int64_t dim0 = 1, dim1 = 1;
    if (shape.size() == 2) { dim0 = shape[0]; dim1 = shape[1]; }
    else if (shape.size() == 1) { dim1 = shape[0]; }

    auto slice = grad_val.slice(0, cur_dim0, cur_dim0 + dim0)  // aten::slice.Tensor
                         .slice(1, cur_dim1, cur_dim1 + dim1);  // aten::slice.Tensor
    if (shape.size() == 1) {
      slice = slice.squeeze(-1);                   // aten::squeeze.dim
    } else if (shape.empty()) {
      slice = slice.squeeze(-1).squeeze(-1);       // aten::squeeze.dim ×2
    }
    grad_inputs[i] = slice;
    cur_dim0 += dim0; cur_dim1 += dim1;
  }
  return grad_inputs;
}
```

**反向 ATen 依赖**：
- `aten::slice.Tensor` — 从 grad 矩阵中切出各输入对应的对角块
- `aten::squeeze.dim` — 0-D/1-D 输入需要降维
- `aten::zeros` — 空输入返回空梯度

**条件依赖**：
- `aten::real` — 复数 grad + 实数输入场景，取 grad 的实部视图

---

### 17. torch.bucketize

**反向来源类型**：不可微

**Forward 路径**：`aten::bucketize.Tensor` → CPU/CUDA/MPS backend kernel → `at::native::bucketize_{cpu/cuda/mps}`

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp** 中的 autograd wrapper 直接 redispatch，不建立 backward node：
```cpp
at::Tensor bucketize_Tensor(...) {
  // ...
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::bucketize(ks & c10::after_autograd_keyset, self_, boundaries_, out_int32, right);
  })();
  auto result = std::move(_tmp);
  // 无 set_history、无 SavedVariable、无 backward node
  return result;
}
```

`bucketize` 返回整数索引张量（`int32` 或 `int64`），不存在有意义的梯度。输出 tensor 不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 18. torch.cholesky

**反向来源类型**：helper 函数

**Forward 路径**：`aten::cholesky` → CPU/CUDA/MPS backend kernel → `at::native::cholesky` → `cholesky_stub` → `aten::tril_`（upper=False）或 `aten::triu_`（upper=True）

**derivatives.yaml 条目**：
```yaml
- name: cholesky(Tensor self, bool upper=False) -> Tensor
  self: cholesky_backward(grad, upper, result)
```

**Backward Node**：`CholeskyBackward0`

**`cholesky_backward` 实现**（`FunctionsManual.cpp:1983`）：
```cpp
Tensor cholesky_backward(const Tensor& gL, bool upper, const Tensor& L) {
  at::NoTF32Guard disable_tf32;
  // gA = L^{-H} π*((L^H gL).tril()) L^{-1}
  // where π*(X) = 0.5 * (X + X^H - diag(X))
  auto L_ = upper ? L.mH() : L;                    // aten::mH (view: adjoint)
  auto gL_ = upper ? gL.mH() : gL;                 // aten::mH (view: adjoint)

  auto gA = L_.mH().matmul(gL_).tril();             // aten::mH, aten::matmul, aten::tril
  gA = 0.5 * (gA + gA.tril(-1).mH());              // aten::mul.Scalar, aten::tril, aten::mH, aten::add.Tensor
  gA = at::linalg_solve_triangular(L_.mH(), gA, /*upper=*/true, /*left=*/true);   // aten::linalg_solve_triangular
  gA = at::linalg_solve_triangular(L_, gA, /*upper=*/false, /*left=*/false);       // aten::linalg_solve_triangular
  return gA;
}
```

**反向 ATen 依赖**：
- `aten::mH`（→ view 操作：`aten::adjoint` → `aten::transpose.int` → `aten::as_strided`，复数额外 `aten::conj`/`aten::_conj`/`aten::alias`）
- `aten::matmul` — 矩阵乘法 L^H × gL
- `aten::tril` — 下三角提取
- `aten::mul.Scalar` — `0.5 * ...`
- `aten::add.Tensor` — `gA + gA.tril(-1).mH()`
- `aten::linalg_solve_triangular` — 三角求解（×2）

**注意**：`torch.cholesky` 已 deprecated，推荐使用 `torch.linalg.cholesky`。

---

### 19. torch.cholesky_solve

**反向来源类型**：helper 函数

**Forward 路径**：`aten::cholesky_solve` → `CompositeExplicitAutograd` → `at::native::cholesky_solve` → `aten::_cholesky_solve_helper` → CPU/CUDA backend kernel

**derivatives.yaml 条目**：
```yaml
- name: cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
  self, input2: cholesky_solve_backward(grad, self, input2, result, upper, grad_input_mask)
  result: cholesky_solve_jvp(result, input2_p, input2_t, self_t, upper)
```

**Backward Node**：`CholeskySolveBackward0`

**`cholesky_solve_backward` 实现**（`FunctionsManual.cpp:4599`）：
```cpp
std::tuple<Tensor, Tensor> cholesky_solve_backward(
    const Tensor& grad_x, const Tensor& self, const Tensor& input2,
    const Tensor& result, const bool upper, std::array<bool, 2> output_mask) {
  at::NoTF32Guard disable_tf32;
  Tensor grad_self, grad_input2;
  if (grad_x.defined()) {
    // self 梯度：递归调用 cholesky_solve
    grad_self = grad_x.cholesky_solve(input2, upper);  // aten::cholesky_solve

    if (output_mask[1]) {
      // input2 梯度
      Tensor common_term = at::matmul(grad_self, result.mH());  // aten::matmul, aten::mH
      common_term = common_term + common_term.mH();               // aten::add.Tensor, aten::mH
      if (upper) {
        grad_input2 = -at::matmul(input2, common_term);           // aten::neg, aten::matmul
      } else {
        grad_input2 = -at::matmul(common_term, input2);           // aten::neg, aten::matmul
      }
    }
  }
  return {grad_self, grad_input2};
}
```

**反向 ATen 依赖**：
- `aten::cholesky_solve` — 递归求解 self 梯度
- `aten::matmul` — 矩阵乘法（grad_self × result^H、input2/common_term 相乘）
- `aten::mH`（→ view 操作）— 共轭转置
- `aten::add.Tensor` — `common_term + common_term.mH()`
- `aten::neg` — 取负（`-at::matmul(...)`）

---

### 20. torch.clip

**反向来源类型**：CIA 前向分解 → `aten::clamp` 的 helper 函数

**Forward 分解路径**（`TensorCompare.cpp:907`）：
- `clip(self, Scalar? min, Scalar? max)` → `at::clamp(self, min, max)` → `aten::clamp`
- `clip(self, Tensor? min, Tensor? max)` → `at::clamp(self, min, max)` → `aten::clamp.Tensor`

```cpp
Tensor clip(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  return at::clamp(self, min, max);
}
Tensor clip(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
  return at::clamp(self, min, max);
}
```

`clip` 是 CIA 别名，backward 挂在 `clamp` 上。

**derivatives.yaml 条目**（`clamp`）：
```yaml
- name: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  self: clamp_backward(grad, self, min, max)
  result: auto_element_wise

- name: clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
  self: clamp_backward(grad, self, min, max)
  min, max: clamp_backward_min_max(grad, self, min, max, grad_input_mask)
  result: clamp_jvp(self_p, self_t, min_p, min_t, max_p, max_t)
```

**Backward Node**：`ClampBackward0`（Scalar 重载）/ `ClampBackward1`（Tensor 重载）

**`clamp_backward` 实现（Scalar 重载）**（`FunctionsManual.cpp:1171`）：
```cpp
Tensor clamp_backward(const Tensor& grad, const Tensor& self,
    const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  if (max && min) {
    auto zero = at::scalar_tensor(0., grad.options());           // aten::scalar_tensor
    return where((self >= *min).logical_and_(self <= *max), grad, zero);
    // aten::ge.Scalar, aten::le.Scalar, aten::logical_and_, aten::where
  } else if (min) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where(self >= *min, grad, zero);                       // aten::ge.Scalar, aten::where
  } else if (max) {
    auto zero = at::scalar_tensor(0., grad.options());
    return where(self <= *max, grad, zero);                       // aten::le.Scalar, aten::where
  } else {
    return grad;                                                  // 无 clamp 时直接传递
  }
}
```

**`clamp_backward` 实现（Tensor 重载）**（`FunctionsManual.cpp:1192`）：
```cpp
Tensor clamp_backward(const Tensor& grad, const Tensor& self,
    const Tensor& min, const Tensor& max) {
  if (max.defined() && min.defined()) {
    auto zero = at::scalar_tensor(0., grad.options());
    const auto self_ge_min = self >= min;                         // aten::ge.Tensor
    const auto self_le_max = self <= max;                         // aten::le.Tensor
    // logical_and_ 或 logical_and 取决于是否有 tensor subclass
    return where(pred, grad, zero);                               // aten::where
  } else if (min.defined()) {
    return where(self >= min, grad, zero);                        // aten::ge.Tensor, aten::where
  } else if (max.defined()) {
    return where(self <= max, grad, zero);                        // aten::le.Tensor, aten::where
  } else {
    return grad;
  }
}
```

**`clamp_backward_min_max`**（Tensor 重载中 min/max 的梯度，`FunctionsManual.cpp:1218`）：
```cpp
// min 梯度: where(self < min && min < max, grad, 0)
// max 梯度: where(self > max || max < min, grad, 0)
```
额外依赖：`aten::lt.Tensor`、`aten::gt.Tensor`、`aten::logical_and_`/`aten::logical_and`、`aten::logical_or_`/`aten::logical_or`、`aten::where`

**反向 ATen 依赖（Scalar 重载）**：
- `aten::scalar_tensor` — 创建零标量
- `aten::ge.Scalar` — `self >= min`
- `aten::le.Scalar` — `self <= max`
- `aten::logical_and_` — 组合条件
- `aten::where` — 条件选择梯度

**反向 ATen 依赖（Tensor 重载，self 梯度）**：
- `aten::scalar_tensor` — 创建零标量
- `aten::ge.Tensor` — `self >= min`
- `aten::le.Tensor` — `self <= max`
- `aten::logical_and_` / `aten::logical_and` — 组合条件
- `aten::where` — 条件选择梯度

**反向 ATen 依赖（Tensor 重载，min/max 梯度）**：
- `aten::lt.Tensor` — `self < min`、`min < max`、`max < min`
- `aten::gt.Tensor` — `self > max`
- `aten::logical_and_` / `aten::logical_and` — min 条件组合
- `aten::logical_or_` / `aten::logical_or` — max 条件组合
- `aten::where` — 条件选择梯度
