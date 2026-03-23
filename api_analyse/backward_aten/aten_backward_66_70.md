# torch API 66-70 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 66 | `torch.lcm` | 不可微 | 无 | 无条目 | 无 | 整数专用 op；`autogradNotImplementedFallback` |
| 67 | `torch.ldexp` | inline 公式 | `LdexpBackward0` | `self: grad * pow(2, other).conj()`；`other: grad * result.conj() * M_LN2` | `aten::pow.Scalar`、`aten::conj`、`aten::mul.Tensor`、`aten::mul.Scalar` | `M_LN2` 是 `ln(2)` 常量 |
| 68 | `torch.linalg.eigvals` | CIA 前向分解 → `aten::linalg_eig` 的 helper 函数 | `LinalgEigBackward0`（挂在 `linalg_eig` 上） | `self: handle_r_to_c(self.scalar_type(), linalg_eig_backward(grads[0], grads[1], eigenvalues, eigenvectors, false))` | `aten::linalg_solve`（非 Hermitian）、`aten::matmul`、`aten::mH`（view）、`aten::conj`、`aten::unsqueeze`、`aten::div.Tensor`、`aten::diagonal`、`aten::copy_`、`aten::real` | `eigvals` 有 grad 时前向改调 `linalg_eig`，backward 挂在 `linalg_eig` 上；仅传 eigenvalues 梯度时走简化路径 `V^{-H} gL V^H` |
| 69 | `torch.linalg.pinv` | helper 函数 | `LinalgPinvBackward0`（挂在 `linalg_pinv.atol_rtol_tensor` 上） | `self: pinv_backward(grad, result, self)` | `aten::mH`（view）、`aten::matmul`、`aten::neg` | `pinv_backward` 是手写 helper（`FunctionsManual.cpp:2110`）；外层 rcond/float 重载是 CIA 外壳，backward 挂在内部 `atol_rtol_tensor` 重载上 |
| 70 | `torch.linalg.vecdot` | CIA 前向分解，无专属 backward | 无 | 无 derivatives.yaml 条目 | `aten::conj`、`aten::mul.Tensor`、`aten::expand` | 1D 路径先经 `vdot` 再递归展开到 `conj/mul`；高维路径的 `sum` backward 回落到 `expand` |

## 详细分析

### 66. torch.lcm

**反向来源类型**：不可微

**Forward 路径**：`aten::lcm`（`structured_delegate: lcm.out`）→ CPU/CUDA backend kernel

**native_functions.yaml**：
```yaml
- func: lcm(Tensor self, Tensor other) -> Tensor
  structured_delegate: lcm.out
  variants: function, method
  tags: pointwise
```

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp**：
```cpp
m.impl("lcm", torch::autograd::autogradNotImplementedFallback());
```

`lcm` 是整数专用操作（最小公倍数），不存在有意义的梯度。输出 tensor 不带 autograd 历史。

**反向 ATen 依赖**：无。

---

### 67. torch.ldexp

**反向来源类型**：inline 公式

**Forward 路径**：`aten::ldexp.Tensor`（CEA）→ 整数指数时 `_ldexp_int_exponent` → `ldexp_stub` → backend kernel；非整数指数时 → `aten::pow` → `aten::mul`

**derivatives.yaml 条目**：
```yaml
- name: ldexp.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad * at::pow(2, other).conj()
  other: grad * result.conj() * M_LN2
  result: self_t * at::pow(2, other_p) + other_t * result * M_LN2
```

**Backward Node**：`LdexpBackward0`

**生成代码**（`Functions.cpp` `LdexpBackward0`）：
```cpp
// self 梯度
auto grad_result = any_grad_defined ? (grad * at::pow(2, other).conj()) : Tensor();
// other 梯度
auto grad_result = any_grad_defined ? (grad * result.conj() * M_LN2) : Tensor();
```

**反向 ATen 依赖**：
- `aten::pow.Scalar` — `at::pow(2, other)`（Scalar base, Tensor exp）
- `aten::conj` — 复数共轭（`.conj()`）
- `aten::mul.Tensor` — `grad * ...`
- `aten::mul.Scalar` — `... * M_LN2`（M_LN2 = ln(2) ≈ 0.6931，C++ double 常量作为 Scalar）

---

### 68. torch.linalg.eigvals

**反向来源类型**：CIA 前向分解 → `aten::linalg_eig` 的 helper 函数

**Forward 分解路径**：
- `requires_grad=True`：`aten::linalg_eigvals`（CIA）→ `at::linalg_eig(input)` → 返回 `std::get<0>(...)` 即 eigenvalues
- `requires_grad=False`：`aten::linalg_eigvals`（CIA）→ `at::_linalg_eigvals(input)` → 不经 autograd

```cpp
Tensor linalg_eigvals(const Tensor& input) {
  if (_may_require_fw_or_bw_grad(input)) {
    return std::get<0>(at::linalg_eig(input));
  }
  return at::_linalg_eigvals(input);
}
```

**derivatives.yaml 条目**（挂在 `linalg_eig` 上）：
```yaml
- name: linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
  self: handle_r_to_c(self.scalar_type(), linalg_eig_backward(grads[0], grads[1], eigenvalues, eigenvectors, /*is_hermitian=*/false))
```

**Backward Node**：`LinalgEigBackward0`

**`linalg_eig_backward` 实现**（`FunctionsManual.cpp:3755`）：

当 `eigvals` 使用时仅有 eigenvalues 的梯度（`gV` 未定义），走简化路径：
```cpp
if (!gV.defined()) {
  // 非 Hermitian: V^{-H} gL V^H = linalg_solve(V^H, gL.unsqueeze(-1) * V^H)
  return at::linalg_solve(V.mH(), gL.unsqueeze(-1) * V.mH());
}
```

**反向 ATen 依赖**（eigvals 场景，简化路径）：
- `aten::mH`（→ view 操作：`aten::transpose.int` → `aten::as_strided`，复数额外 `aten::conj`）— `V.mH()`
- `aten::unsqueeze` — `gL.unsqueeze(-1)`
- `aten::mul.Tensor` — `gL.unsqueeze(-1) * V.mH()`
- `aten::linalg_solve` — 求解线性系统

**条件依赖**：`handle_r_to_c` 在复数梯度 + 实数输入场景调用 `aten::real`。

---

### 69. torch.linalg.pinv

**反向来源类型**：helper 函数

**Forward 路径**：
`aten::linalg_pinv(Tensor, float, bool)`（CIA）→ `aten::linalg_pinv.atol_rtol_float`（CIA）→ `aten::linalg_pinv.atol_rtol_tensor`（CEANonFunctional，autograd 注册点）→ `at::native::linalg_pinv`（非 Hermitian: SVD 路径 / Hermitian: eigh 路径）

**derivatives.yaml 条目**（挂在 `linalg_pinv.atol_rtol_tensor` 上）：
```yaml
- name: linalg_pinv.atol_rtol_tensor(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
  self: pinv_backward(grad, result, self)
```

**Backward Node**：`LinalgPinvBackward0`

**`pinv_backward` 实现**（`FunctionsManual.cpp:2110`）：
```cpp
Tensor pinv_backward(const Tensor& grad, const Tensor& pinvA, const Tensor& A) {
  at::NoTF32Guard disable_tf32;
  auto m = A.sym_size(-2);
  auto n = A.sym_size(-1);
  auto pinvAh = pinvA.mH();
  auto gradh = grad.mH();
  if (m <= n) {
    auto K = gradh.matmul(pinvA);
    auto KpinvAh = K.matmul(pinvAh);
    return -(pinvA.matmul(K)).mH() + KpinvAh -
        (A.matmul(pinvA)).matmul(KpinvAh) +
        (pinvAh.matmul(pinvA)).matmul(gradh - K.matmul(A));
  } else {
    auto K = pinvA.matmul(gradh);
    auto pinvAhK = pinvAh.matmul(K);
    return -(K.matmul(pinvA)).mH() +
        (gradh - A.matmul(K)).matmul(pinvA).matmul(pinvAh) + pinvAhK -
        pinvAhK.matmul(pinvA).matmul(A);
  }
}
```

**反向 ATen 依赖**：
- `aten::mH`（→ view 操作：`aten::transpose.int` → `aten::as_strided`，复数额外 `aten::conj`/`aten::_conj`/`aten::alias`）— `pinvA.mH()`、`grad.mH()`、`(...).mH()`
- `aten::matmul` — 多处矩阵乘法（`gradh.matmul(pinvA)` 等）
- `aten::neg` — 取负 `-(...)`
- `aten::add.Tensor` — 各项相加（由 `+` 运算符触发）
- `aten::sub.Tensor` — `gradh - K.matmul(A)` / `gradh - A.matmul(K)`

---

### 70. torch.linalg.vecdot

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- 1D 向量：`aten::linalg_vecdot`（CIA）→ `aten::vdot`
- 高维张量：`aten::linalg_vecdot`（CIA）→ `aten::conj(x)` → `aten::mul(conj_x, y)` → `aten::sum(result, dim)`

**native_functions.yaml**：
```yaml
- func: linalg_vecdot(Tensor x, Tensor y, *, int dim=-1) -> Tensor
  python_module: linalg
  variants: function
```
无 dispatch key，为 CompositeImplicitAutograd。

**derivatives.yaml**：无 `linalg_vecdot` 条目。`vdot` 有独立条目：
```yaml
- name: vdot(Tensor self, Tensor other) -> Tensor
  self: grad.conj() * other
  other: grad * self
```

由于 `linalg_vecdot` 是 CIA，autograd 引擎对每个子 op 独立录制。

**1D 向量路径反向 ATen 依赖**（通过 `vdot` 的 backward）：
- `aten::conj` — `grad.conj()`
- `aten::mul.Tensor` — `grad.conj() * other`、`grad * self`

**高维路径反向 ATen 依赖**（通过 `conj` + `mul` + `sum` 各自的 backward）：
- `aten::conj` — `conj` 的反向
- `aten::mul.Tensor` — `mul` 的反向
- `aten::sum` 的反向 → `aten::expand`（沿求和维度展开梯度）
