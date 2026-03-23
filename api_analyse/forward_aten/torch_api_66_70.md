# torch API 66-70 ATen Dispatch 分析

| 接口 | 场景 | ATen Dispatch 调用链 | 正向 Dispatch 依赖的 ATen 接口 |
| --- | --- | --- | --- |
| `torch.lcm` | dense 整数 `Tensor / Tensor` | Python 绑定 -> `aten::lcm` -> `CPU/CUDA` structured wrapper（`meta -> impl`）-> `native::lcm_out` -> `lcm_stub` -> `{cpu,cuda}::lcm_kernel` | `aten::lcm` |
| `torch.ldexp` | 浮点尾数 + 整数指数（常见 fast path；非整数指数会分支） | Python 绑定 -> `aten::ldexp.Tensor` -> `AutogradCPU/CUDA` 的 `LdexpBackward0`（若需要梯度）-> `CompositeExplicitAutograd::native::ldexp` -> 整数指数时走 `native::_ldexp_int_exponent -> ldexp_stub -> {cpu,cuda}::ldexp_kernel`；否则改发 `aten::pow -> aten::mul` | `aten::ldexp.Tensor`, `aten::pow`, `aten::mul` |
| `torch.linalg.eigvals` | 方阵；`requires_grad=False` 与 `True` 路径不同 | Python 绑定 -> `aten::linalg_eigvals` -> `CompositeImplicitAutograd::native::linalg_eigvals` -> 无 grad: `aten::_linalg_eigvals` -> `CPU/CUDA::native::_linalg_eigvals`；有 grad: 内部改调 `aten::linalg_eig` -> `AutogradCPU/CUDA` -> `CPU/CUDA::linalg_eig` | `aten::linalg_eigvals`, `aten::_linalg_eigvals`, `aten::linalg_eig` |
| `torch.linalg.pinv` | 默认 `rcond` 为 `float`，`hermitian=False`；`hermitian=True` 改走 Hermitian 分支 | Python 绑定 -> `aten::linalg_pinv(Tensor,float,bool)` -> `CompositeImplicitAutograd::native::linalg_pinv(rcond)` -> `aten::linalg_pinv.atol_rtol_float` -> `CompositeImplicitAutograd::native::linalg_pinv(atol/rtol float)` -> `aten::linalg_pinv.atol_rtol_tensor` -> `AutogradCPU/CUDA` 的 `LinalgPinvBackward0`（若需要梯度）-> `CompositeExplicitAutogradNonFunctional::native::linalg_pinv` -> 非 Hermitian: `aten::svd + aten::narrow + aten::where + aten::matmul`；Hermitian: `aten::linalg_eigh + aten::amax + aten::where + aten::matmul` | `aten::linalg_pinv`, `aten::linalg_pinv.atol_rtol_float`, `aten::linalg_pinv.atol_rtol_tensor`, `aten::svd`, `aten::narrow`, `aten::where`, `aten::matmul`, `aten::linalg_eigh`, `aten::amax` |
| `torch.linalg.vecdot` | 1D 向量与高维 batch 路径不同 | Python 绑定 -> `aten::linalg_vecdot` -> `CompositeImplicitAutograd::native::linalg_vecdot` -> 1D 向量时走 `aten::vdot`；更高维时走 `aten::conj -> aten::mul -> aten::sum` | `aten::linalg_vecdot`, `aten::vdot`, `aten::conj`, `aten::mul`, `aten::sum` |

## 备注

- `torch.lcm` 在 `VariableTypeEverything.cpp` 上注册的是 `autogradNotImplementedFallback()`，没有可微 backward 链。
- `torch.linalg.eigvals` 自己没有单独的 autograd wrapper；可微场景是它在 native 实现里显式跳到 `aten::linalg_eig`。
- `torch.linalg.pinv` 真正承接 backward 的是内部 `aten::linalg_pinv.atol_rtol_tensor`；外层 `rcond`/float overload 只是 composite 外壳。
