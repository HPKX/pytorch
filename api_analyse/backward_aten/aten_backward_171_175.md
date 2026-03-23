# torch API 171-175 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 171 | `torch.optim.Rprop` | Python 组合更新；默认 `no_grad`，无统一 backward | 无统一 node | 无单一条目 | 默认无；`differentiable=True` 时由内部原语各自求导：`aten::neg`、`aten::mul`、`aten::sign`、`aten::mul_`、`aten::clamp_`、`aten::addcmul_`、`aten::copy_` | 不是单一 ATen schema；`foreach=True` 也只是切换到 `_foreach_*` 原语 |
| 172 | `torch.optim.adadelta.Adadelta` | Python 组合更新；默认 `no_grad`，无统一 backward | 无统一 node | 无单一条目 | 默认无；`differentiable=True` 时由内部原语各自求导：`aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::addcmul_`、`aten::sqrt_`、`aten::div_` | `foreach` 分支同理，无独立 `AdadeltaBackward` |
| 173 | `torch.optim.adagrad.Adagrad` | Python 组合更新；默认 `no_grad`，无统一 backward | 无统一 node | 无单一条目 | 默认无；dense 路径涉及 `aten::add_`、`aten::neg`、`aten::add`、`aten::addcmul_`、`aten::sqrt` / `aten::sqrt_`、`aten::addcdiv_`；sparse 路径还涉及 `aten::pow`、`aten::sparse_mask`；`fused=True` 为 `aten::_fused_adagrad_` | `foreach` / `fused` 只改变内部原语，不形成单独 schema |
| 174 | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` | 纯 Python 调度逻辑，不对应固定 ATen backward | 无 | 无条目 | 无 | `step()` / `get_lr()` 只更新 Python `param_groups["lr"]`；不存在稳定 dispatcher schema |
| 175 | `torch.orgqr` | 前端别名；真实反向挂在 `aten::linalg_householder_product` | `LinalgHouseholderProductBackward0` | `input, tau: householder_product_backward(grad, result, input, tau)` | 直接依赖：`aten::linalg_householder_product`；展开 helper 后依赖 `aten::tril`、`aten::diagonal`、`aten::fill_`、`aten::sum`、`aten::matmul`、`aten::mH`、`aten::narrow`、`aten::zeros_like`、`aten::cat`、`aten::copy_` | `orgqr` 本身只是 `linalg_householder_product` 别名；没有独立 `OrgqrBackward` |

## 详细分析

### 171. torch.optim.Rprop

**反向来源类型**：Python 组合更新，无统一 backward

`Rprop.step()` 在 `torch/optim/rprop.py` 中由 `_single_tensor_rprop` / `_multi_tensor_rprop` 组合基础原语实现。仓库里不存在独立 `rprop` ATen schema，也没有 `derivatives.yaml` 对应条目。

**默认行为**：
- 常规训练配置下 optimizer step 不参与 autograd。

**若 `differentiable=True`，内部原语各自求导**：
- `aten::neg`
- `aten::mul`
- `aten::sign`
- `aten::mul_`
- `aten::clamp_`
- `aten::addcmul_`
- `aten::copy_`

`foreach=True` 只把这些原语替换为 `_foreach_*` 批量版本，不改变“无统一 backward schema”的结论。

---

### 172. torch.optim.adadelta.Adadelta

**反向来源类型**：Python 组合更新，无统一 backward

`Adadelta.step()` → `adadelta(...)` → `_single_tensor_adadelta / _multi_tensor_adadelta`。不存在独立 `adadelta` dispatcher schema。

**若显式要求可微**，梯度仅沿内部原语传播：
- `aten::add_`
- `aten::neg`
- `aten::add`
- `aten::mul_`
- `aten::addcmul_`
- `aten::sqrt_`
- `aten::div_`

没有统一 `AdadeltaBackward` node。

---

### 173. torch.optim.adagrad.Adagrad

**反向来源类型**：Python 组合更新，无统一 backward

`Adagrad.step()` → `adagrad(...)`，再分为 dense / sparse / foreach / fused 多个实现。仓库中不存在独立 `adagrad` schema，因此不能像普通 ATen op 一样锁定单一 `derivatives.yaml` 条目。

**内部可能参与反向的原语**：
- dense：`aten::add_`、`aten::neg`、`aten::add`、`aten::addcmul_`、`aten::sqrt`、`aten::sqrt_`、`aten::addcdiv_`
- sparse：`aten::pow`、`aten::add_`、`aten::sparse_mask`、`aten::sqrt_`
- fused：`aten::_fused_adagrad_`

**备注**：
- 默认优化步骤依然不建立统一 backward 图。

---

### 174. torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

**反向来源类型**：纯 Python 调度逻辑，不对应固定 ATen backward

这个接口的核心是：
- Python 维护 `T_cur` / `T_i` / `last_epoch`
- `get_lr()` 使用 Python `math.cos` 和标量运算生成新的学习率
- 然后写回 optimizer `param_groups`

不存在稳定的 ATen schema、`derivatives.yaml` 条目、`VariableTypeEverything.cpp` wrapper 或 `Functions.cpp` backward node。

**反向 ATen 依赖**：无。

---

### 175. torch.orgqr

**反向来源类型**：前端别名，真实反向挂在 `aten::linalg_householder_product`

**锁定 schema**：
- 前端入口：`aten::orgqr(Tensor self, Tensor input2) -> Tensor`
- 实际可微 schema：`aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor`

`native_functions.yaml` 明确注释：
```yaml
# orgqr, alias for linalg_householder_product
```

`BatchLinearAlgebra.cpp` 中的实现也是：
```cpp
Tensor orgqr(const Tensor& input, const Tensor& tau) {
  return at::linalg_householder_product(input, tau);
}
```

因此 `torch.orgqr` 本身没有独立 backward 定义；实际 backward 来自 `linalg_householder_product`。

**derivatives.yaml 条目**：
```yaml
- name: linalg_householder_product(Tensor input, Tensor tau) -> Tensor
  input, tau: householder_product_backward(grad, result, input, tau)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `LinalgHouseholderProductBackward0`
- `Functions.cpp` 调用 `householder_product_backward(...)`

**直接反向依赖**：
- `aten::linalg_householder_product`
- helper：`householder_product_backward`

**helper 展开后的主要 ATen 依赖**：
- `aten::tril`
- `aten::diagonal`
- `aten::fill_`
- `aten::sum`
- `aten::matmul`
- `aten::mH`
- `aten::narrow`
- `aten::zeros_like`
- `aten::cat`
- `aten::copy_`

**备注**：
- `orgqr` 前向虽然最终落到 `orgqr_stub` kernel，但 autograd 层面分析应锁定到真正带导数定义的 `linalg_householder_product`。
