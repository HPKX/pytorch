# torch API 71-75 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 71 | `torch.logcumsumexp` | helper 函数 | `LogcumsumexpBackward0` | `self: logcumsumexp_backward(grad, self, result, dim)` | `aten::flip`、`aten::logcumsumexp`、`aten::abs`、`aten::log`、`aten::where`、`aten::scalar_tensor`、`aten::sub.Tensor`、`aten::add.Tensor`、`aten::exp`、`aten::conj`（复数）、`aten::gt.Scalar`、`aten::lt.Scalar` | `logcumsumexp_backward` 是手写 helper（`FunctionsManual.cpp:890`）；实数路径分离正负梯度避免数值问题 |
| 72 | `torch.logdet` | CIA 前向分解 → `aten::_linalg_slogdet` 的 helper 函数 | `LinalgSlogdetBackward0`（挂在 `_linalg_slogdet` 上） | `A: slogdet_backward(grad_sign, grad_logabsdet, A, sign, LU, pivots)` | `aten::imag`、`aten::conj`、`aten::mul.Tensor`、`aten::diag_embed`、`aten::unsqueeze`、`aten::expand_as`、`aten::mT`（view）、`aten::linalg_lu_solve`、`aten::linalg_solve`、`aten::mH`（view） | `logdet` 是 CIA 分解为 `slogdet`→ `_linalg_slogdet`；backward 挂在 `_linalg_slogdet` 上 |
| 73 | `torch.logit` | inline + helper（条件分支） | `LogitBackward0` | `self: GradMode::is_enabled() ? infinitely_differentiable_logit_backward(grad, self, eps) : logit_backward(grad, self, eps)` | GradMode on: `aten::logical_and`、`aten::ge.Scalar`、`aten::le.Scalar`、`aten::where`、`aten::div.Tensor`、`aten::mul.Tensor`、`aten::sub.Scalar`、`aten::zeros`/`aten::empty`；GradMode off: `aten::logit_backward`（native kernel） | 高阶求导走手写 helper `infinitely_differentiable_logit_backward`；一阶求导走原生 `logit_backward` kernel |
| 74 | `torch.logspace` | 不可微（工厂函数） | 无 | 无条目 | 无 | CEA 工厂函数；`autogradNotImplementedFallback` |
| 75 | `torch.lu_solve` | CIA 前向分解 → `aten::linalg_lu_solve` 的 inline + helper | `LinalgLuSolveBackward0`（挂在 `linalg_lu_solve` 上） | `LU: linalg_lu_solve_LU(grad, LU, pivots, result, left, adjoint)`；`B: linalg_lu_solve(LU, pivots, grad, left, !adjoint)` | `aten::linalg_lu_solve`、`aten::lu_unpack`、`aten::matmul`、`aten::mH`（view）、`aten::mT`（view）、`aten::neg`、`aten::linalg_solve_triangular`、`aten::tril`、`aten::triu`、`aten::add.Tensor` | `lu_solve` 是 CIA deprecated wrapper → `linalg_lu_solve`；`linalg_lu_solve_LU` 是手写 helper（`FunctionsManual.cpp:5914`） |

## 详细分析

### 71. torch.logcumsumexp

**反向来源类型**：helper 函数

**Forward 路径**：`aten::logcumsumexp`（CEA）→ `aten::_logcumsumexp` → CPU/CUDA/MPS backend kernel

**derivatives.yaml 条目**：
```yaml
- name: logcumsumexp(Tensor self, int dim) -> Tensor
  self: logcumsumexp_backward(grad, self, result, dim)
```

**Backward Node**：`LogcumsumexpBackward0`

**`logcumsumexp_backward` 实现**（`FunctionsManual.cpp:890`）：
```cpp
Tensor logcumsumexp_backward(Tensor grad, const Tensor& self, const Tensor& result, int64_t dim) {
  if (grad.dim() == 0 || grad.sym_numel() == 0) {
    return grad;
  }

  auto reverse_logcumsumexp = [dim](const auto& x) {
    return at::flip(at::logcumsumexp(at::flip(x, {dim}), dim), {dim});
  };

  if (!at::is_complex(grad)) {
    auto grad_min = at::scalar_tensor(scalar_min, grad.options());
    auto log_abs_grad = grad.abs().log();
    auto log_grad_positive = at::where(grad > 0, log_abs_grad, grad_min);
    auto log_grad_negative = at::where(grad < 0, log_abs_grad, grad_min);

    auto output_pos = (reverse_logcumsumexp(log_grad_positive - result) + self).exp();
    auto output_neg = (reverse_logcumsumexp(log_grad_negative - result) + self).exp();
    return output_pos - output_neg;
  } else {
    auto log_grad = grad.conj().log();
    auto output = (reverse_logcumsumexp(log_grad - result) + self).exp();
    return output.conj();
  }
}
```

**反向 ATen 依赖**（实数路径）：
- `aten::scalar_tensor` — 创建 `grad_min` 标量
- `aten::abs` — `grad.abs()`
- `aten::log` — `grad.abs().log()`
- `aten::gt.Scalar` — `grad > 0`
- `aten::lt.Scalar` — `grad < 0`
- `aten::where` — 条件选择（×2）
- `aten::sub.Tensor` — `log_grad_positive - result`、`output_pos - output_neg`
- `aten::flip` — 翻转张量（`reverse_logcumsumexp` 内部，×4）
- `aten::logcumsumexp` — 递归调用自身（`reverse_logcumsumexp` 内部，×2）
- `aten::add.Tensor` — `... + self`（×2）
- `aten::exp` — `.exp()`（×2）

**反向 ATen 依赖**（复数路径）：
- `aten::conj` — `grad.conj()`、`output.conj()`
- `aten::log` — `grad.conj().log()`
- `aten::sub.Tensor` — `log_grad - result`
- `aten::flip`、`aten::logcumsumexp`（`reverse_logcumsumexp` 内部，×2 each）
- `aten::add.Tensor` — `... + self`
- `aten::exp` — `.exp()`

---

### 72. torch.logdet

**反向来源类型**：CIA 前向分解 → `aten::_linalg_slogdet` 的 helper 函数

**Forward 分解路径**：
`aten::logdet`（CIA）→ `at::linalg_slogdet(A)` → `aten::linalg_slogdet`（CIA）→ `aten::_linalg_slogdet`（autograd 注册点）→ backend kernel

`logdet` 的 CIA 实现：
```cpp
Tensor logdet(const Tensor& A) {
  auto [sign, logabsdet] = at::linalg_slogdet(A);
  if (A.is_complex()) {
    return sign.log() + logabsdet;    // aten::log + aten::add.Tensor
  } else {
    return at::where(sign == -1., NAN, logabsdet);  // aten::where
  }
}
```

**derivatives.yaml 条目**（挂在 `_linalg_slogdet` 上）：
```yaml
- name: _linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)
  A: slogdet_backward(grad_sign, grad_logabsdet, A, sign, LU, pivots)
  output_differentiability: [True, True, False, False]
```

**Backward Node**：`LinalgSlogdetBackward0`

**`slogdet_backward` 实现**（`FunctionsManual.cpp:4396`）：
```cpp
Tensor slogdet_backward(const Tensor& grad_sign, const Tensor& grad_logabsdet,
    const Tensor& A, const Tensor& signdet, const Tensor& LU, const Tensor& pivots) {
  auto is_complex = A.is_complex();

  auto g = grad_logabsdet;
  if (is_complex) {
    if (grad_sign.defined()) {
      auto i = c10::complex<double>{0.0, 1.0};
      g = g - i * at::imag(grad_sign.conj() * signdet);  // aten::imag, aten::conj, aten::mul
    }
  }

  // (g_abs - g_sign.conj() * sgn) * A^{-H}
  auto d = at::diag_embed(g.unsqueeze(-1).expand_as(pivots)).mT();  // aten::diag_embed, aten::unsqueeze, aten::expand_as, aten::mT
  if (!at::GradMode::is_enabled()) {
    return at::linalg_lu_solve(LU, pivots, d, /*left=*/true, /*adjoint=*/...);  // aten::linalg_lu_solve
  } else {
    return at::linalg_solve(A.mH(), d);  // aten::linalg_solve, aten::mH
  }
}
```

**反向 ATen 依赖**：
- `aten::unsqueeze` — `g.unsqueeze(-1)`
- `aten::expand_as` — `.expand_as(pivots)`
- `aten::diag_embed` — `at::diag_embed(...)`
- `aten::mT`（→ view 操作：`aten::transpose.int`）— `.mT()`

条件分支：
- GradMode off: `aten::linalg_lu_solve` — 使用 LU 分解直接求解
- GradMode on: `aten::linalg_solve` — 重新计算求解（支持高阶导数）
- `aten::mH`（→ view 操作）— `A.mH()`

复数额外依赖：
- `aten::imag` — `at::imag(grad_sign.conj() * signdet)`
- `aten::conj` — `grad_sign.conj()`
- `aten::mul.Tensor` — `grad_sign.conj() * signdet`
- `aten::sub.Tensor` — `g - i * ...`

---

### 73. torch.logit

**反向来源类型**：inline + helper（条件分支）

**Forward 路径**：`aten::logit` → CPU/CUDA/MPS backend kernel（`logit_stub`）

**derivatives.yaml 条目**：
```yaml
- name: logit(Tensor self, float? eps=None) -> Tensor
  self: "GradMode::is_enabled() ? infinitely_differentiable_logit_backward(grad, self, eps) : logit_backward(grad, self, eps)"
```

**Backward Node**：`LogitBackward0`

**生成代码**（`Functions.cpp` `LogitBackward0`）：
```cpp
auto grad_result = any_grad_defined ?
    (GradMode::is_enabled() ?
        infinitely_differentiable_logit_backward(grad, self, eps) :
        logit_backward(grad, self, eps)) : Tensor();
```

**路径 1：GradMode off → `logit_backward`（原生 ATen kernel）**

`logit_backward` 是在 `native_functions.yaml` 中注册的原生 op：
```yaml
- func: logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor
  structured_delegate: logit_backward.grad_input
```
直接调用 CPU/CUDA/MPS 的 `logit_backward_out` kernel。

反向 ATen 依赖：
- `aten::logit_backward` — 单一原生 op 调用

**路径 2：GradMode on → `infinitely_differentiable_logit_backward`（手写 helper）**

**`infinitely_differentiable_logit_backward` 实现**（`FunctionsManual.cpp:2313`）：
```cpp
Tensor infinitely_differentiable_logit_backward(
    const Tensor& grad, const Tensor& self, std::optional<double> eps) {
  if (eps) {
    const double lo = eps.value();
    const double hi = 1.0 - lo;
    return at::where(
        at::logical_and(self >= lo, self <= hi),
        grad / (self * (1.0 - self)),
        at::zeros({}, self.options()));
  } else {
    return at::where(
        at::logical_and(self >= 0.0, self <= 1.0),
        grad / (self * (1.0 - self)),
        at::empty({}, self.options()).fill_(std::numeric_limits<double>::quiet_NaN()));
  }
}
```

反向 ATen 依赖（GradMode on 路径）：
- `aten::ge.Scalar` — `self >= lo` 或 `self >= 0.0`
- `aten::le.Scalar` — `self <= hi` 或 `self <= 1.0`
- `aten::logical_and` — 组合条件
- `aten::mul.Tensor` — `self * (1.0 - self)`
- `aten::sub.Scalar` — `1.0 - self`
- `aten::div.Tensor` — `grad / (...)`
- `aten::where` — 条件选择
- `aten::zeros` — 创建零标量（有 eps 时）
- `aten::empty` + `aten::fill_` — 创建 NaN 标量（无 eps 时）

---

### 74. torch.logspace

**反向来源类型**：不可微（工厂函数）

**Forward 路径**：`aten::logspace`（CEA）→ `aten::empty` → `aten::logspace.out` → CPU/CUDA backend kernel

**native_functions.yaml**：
```yaml
- func: logspace(Scalar start, Scalar end, int steps, float base=10.0, ...) -> Tensor
  dispatch:
    CompositeExplicitAutograd: logspace
```

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp**：
```cpp
m.impl("logspace", torch::autograd::autogradNotImplementedFallback());
m.impl("logspace.Tensor_Tensor", torch::autograd::autogradNotImplementedFallback());
m.impl("logspace.Tensor_Scalar", torch::autograd::autogradNotImplementedFallback());
m.impl("logspace.Scalar_Tensor", torch::autograd::autogradNotImplementedFallback());
```

`logspace` 是工厂函数（Scalar/Tensor 参数重载均走 `autogradNotImplementedFallback`），输出为 fresh tensor 不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 75. torch.lu_solve

**反向来源类型**：CIA 前向分解 → `aten::linalg_lu_solve` 的 inline + helper

**Forward 分解路径**：
`aten::lu_solve`（CIA，deprecated wrapper）→ `at::linalg_lu_solve(LU_data, LU_pivots, self)`

```cpp
Tensor lu_solve(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots) {
  TORCH_WARN_ONCE("torch.lu_solve is deprecated in favor of torch.linalg.lu_solve...");
  return at::linalg_lu_solve(LU_data, LU_pivots, self);
}
```

**derivatives.yaml 条目**（挂在 `linalg_lu_solve` 上）：
```yaml
- name: linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor
  LU: linalg_lu_solve_LU(grad, LU, pivots, result, left, adjoint)
  B: "at::linalg_lu_solve(LU, pivots, grad, left, !adjoint)"
```

注意：`lu_solve(self, LU_data, LU_pivots)` 映射到 `linalg_lu_solve(LU=LU_data, pivots=LU_pivots, B=self)`，所以 `self` 的梯度对应 `B` 的公式，`LU_data` 的梯度对应 `LU` 的公式。

**Backward Node**：`LinalgLuSolveBackward0`

**B 的梯度**（inline 公式）：
```cpp
at::linalg_lu_solve(LU, pivots, grad, left, !adjoint)
```
反向 ATen 依赖：
- `aten::linalg_lu_solve` — 直接调用

**LU 的梯度 — `linalg_lu_solve_LU` 实现**（`FunctionsManual.cpp:5914`）：
```cpp
Tensor linalg_lu_solve_LU(const Tensor& gX, const Tensor& LU, const Tensor& pivots,
    const Tensor& X, const bool left, const bool adjoint) {
  auto [P, L, U] = at::lu_unpack(LU, pivots, true, left == adjoint);  // aten::lu_unpack

  if (left != adjoint) {
    // gR = U^{-H} op_2(-gX) op_2(X)^H
    auto gR = at::linalg_solve_triangular(
        U.mH(), -(left ? gX : gX.mH()).matmul(left ? X.mH() : X), false);
    // gL = (L^{-H} gR U^H).tril(-1)
    auto gL = at::linalg_solve_triangular(
        L.mH(), gR.matmul(U.mH()), true, true, true).tril(-1);
    return gL + gR.triu();
  } else {
    // gR = -P^T op_3(X) op_1(op_2(gX)) P
    auto gR = -P.mT().matmul(...).matmul(...).matmul(P);
    gR = at::linalg_solve_triangular(L.mH(), gR, true, false, true);
    auto gU = at::linalg_solve_triangular(
        U.mH(), L.mH().matmul(gR), false, false).triu();
    return gR.tril(-1) + gU;
  }
}
```

**反向 ATen 依赖（LU 梯度路径）**：
- `aten::lu_unpack` — 解包 LU 分解得到 P、L、U
- `aten::mH`（→ view 操作）— `U.mH()`、`L.mH()`、`X.mH()`、`gX.mH()`
- `aten::mT`（→ view 操作）— `P.mT()`
- `aten::neg` — 取负 `-gX`、`-P.mT().matmul(...)`
- `aten::matmul` — 多处矩阵乘法
- `aten::linalg_solve_triangular` — 三角求解（×2 或 ×3）
- `aten::tril` — 下三角提取（`.tril(-1)`）
- `aten::triu` — 上三角提取（`.triu()`）
- `aten::add.Tensor` — `gL + gR.triu()` / `gR.tril(-1) + gU`

**综合反向 ATen 依赖（B + LU 梯度）**：
- `aten::linalg_lu_solve` — B 梯度
- `aten::lu_unpack` — LU 解包
- `aten::linalg_solve_triangular` — 三角求解
- `aten::matmul` — 矩阵乘法
- `aten::mH`、`aten::mT`（view 操作）
- `aten::neg`、`aten::tril`、`aten::triu`、`aten::add.Tensor`
