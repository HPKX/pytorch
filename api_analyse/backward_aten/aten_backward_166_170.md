# torch API 166-170 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 166 | `torch.nn.modules.flatten.Unflatten` | CIA 前向分解到 `aten::view`，无专属 backward | 无统一 node；主要为内层 `ViewBackward0` | 无条目 | `aten::view` | `aten::unflatten.int` / `aten::unflatten.Dimname` 都会先做 shape 计算，再转到内层 `aten::view`；NamedTensor 路径额外写回 names，但不新增 ATen backward 依赖 |
| 167 | `torch.numel` | 不可微 | 无 | 无条目 | 无 | eager 路径直接读取张量形状元信息，JIT 路径是 prim `aten::numel(Tensor)->int`；返回 `int/SymInt`，不进入 autograd 图 |
| 168 | `torch.optim.ASGD` | Python 组合更新；默认 `no_grad`，无统一 backward | 无统一 node | 无单一条目 | 默认无；`differentiable=True` 时仅由内部原语各自求导：`aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::sub`、`aten::copy_` | 不是单一 ATen schema；`foreach=True` 路径显式走 `_foreach_*` 原语，常规优化步骤通常不建立梯度图 |
| 169 | `torch.optim.Adamax` | Python 组合更新；默认 `no_grad`，无统一 backward | 无统一 node | 无单一条目 | 默认无；`differentiable=True` 时由内部原语各自求导：`aten::add_`、`aten::add`、`aten::lerp_`、`aten::mul_`、`aten::abs`、`aten::maximum.out`、`aten::addcdiv_` | `foreach` 快路径同样不是单一 schema；常见训练配置下 optimizer step 不参与 autograd |
| 170 | `torch.optim.RMSprop` | Python 组合更新；默认 `no_grad`，无统一 backward | 无统一 node | 无单一条目 | 默认无；`differentiable=True` 时由内部原语各自求导：`aten::add_`、`aten::add`、`aten::mul_`、`aten::addcmul_`、`aten::lerp_`、`aten::sqrt` / `aten::sqrt_`、`aten::addcdiv_` | `centered` / `momentum` / `foreach` 分支只改变内部原语组合，不产生单独 optimizer backward schema |

## 详细分析

### 166. torch.nn.modules.flatten.Unflatten

**反向来源类型**：`CompositeImplicitAutograd` 前向分解（view）

**锁定 schema**：
- `aten::unflatten.int(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)`
- `aten::unflatten.Dimname(Tensor(a) self, Dimname dim, SymInt[] sizes, Dimname[] names) -> Tensor(a)`

两者在 `native_functions.yaml` 中都注册为 `CompositeImplicitAutograd`，没有 `derivatives.yaml` 条目。

**Forward 路径**：
- `Unflatten.forward` → `aten::unflatten.int / aten::unflatten.Dimname`
- 前向实现先做 shape 计算
- 再转到内层 `aten::view`

`TensorShape.cpp` 里的核心实现是：
```cpp
result = self.view_symint(shape);
```

因此 `unflatten` 自己不生成专属 backward，autograd 实际记录的是内层 `aten::view` 的 view backward。

**反向 ATen 依赖**：
- `aten::view` - `ViewBackward0` 负责把梯度 replay 回原 shape

**备注**：
- `Dimname` overload 只是额外调用 `internal_set_names_inplace` 写回 names，不引入新的可微 ATen op。
- 这是标准 view 语义，不是 inplace。

---

### 167. torch.numel

**反向来源类型**：不可微

`torch.numel` 没有稳定的 eager dispatcher schema。Python eager 路径直接读取张量形状元信息；JIT tracing 时插入的是 prim `aten::numel(Tensor)->int` 节点，也不走 `native_functions.yaml` 那套 autograd wrapper。

返回值是 `int` / `SymInt`，不是 Tensor，因此不会创建 backward node，也没有 `derivatives.yaml` 可分析条目。

**反向 ATen 依赖**：无。

---

### 168. torch.optim.ASGD

**反向来源类型**：Python 组合更新，无统一 backward

`ASGD.step()` 是 Python 逻辑：
- `ASGD.step` → `asgd(...)`
- 再进入 `_single_tensor_asgd` 或 `_multi_tensor_asgd`
- 内部串联 `add_`、`neg`、`add`、`mul_`、`sub`、`copy_` 或 `_foreach_*`

仓库里不存在名为 `asgd` 的 ATen schema，也没有 `derivatives.yaml` 条目、`VariableTypeEverything.cpp` wrapper 或 `Functions.cpp` node。

**结论**：
- 默认 `differentiable=False` 时，优化器 step 通常在 `no_grad` 语义下执行，不建立 backward 图。
- 若显式 `differentiable=True` 且走单张量路径，梯度只会沿内部原语各自传播，没有统一的 `ASGDBackward`。

**可能参与反向的内部原语**：
- `aten::add_`
- `aten::neg`
- `aten::add`
- `aten::mul_`
- `aten::sub`
- `aten::copy_`

**备注**：
- `foreach=True` 分支使用 `_foreach_add_`、`_foreach_neg`、`_foreach_addcmul_` 等批量原语；这些路径通常不作为常规 autograd 分析对象。

---

### 169. torch.optim.Adamax

**反向来源类型**：Python 组合更新，无统一 backward

与 `ASGD` 相同，`Adamax.step()` 是 Python 级组合：
- `Adamax.step` → `adamax(...)`
- 再进入 `_single_tensor_adamax` 或 `_multi_tensor_adamax`
- 内部使用 `aten::add_`、`aten::add`、`aten::lerp_`、`aten::mul_`、`aten::abs`、`aten::maximum.out`、`aten::addcdiv_` 等原语

没有独立的 `adamax` ATen schema，也没有 `derivatives.yaml` / generated autograd node。

**结论**：
- 默认优化步骤无统一 backward。
- `differentiable=True` 时，只有内部基础 ATen 原语分别参与求导。

**可能参与反向的内部原语**：
- `aten::add_`
- `aten::add`
- `aten::lerp_`
- `aten::mul_`
- `aten::abs`
- `aten::maximum.out`
- `aten::addcdiv_`

---

### 170. torch.optim.RMSprop

**反向来源类型**：Python 组合更新，无统一 backward

`RMSprop.step()` 也是 Python 组合逻辑：
- `RMSprop.step` → `rmsprop(...)`
- 再进入 `_single_tensor_rmsprop` 或 `_multi_tensor_rmsprop`
- `centered` / `momentum` / `foreach` 仅改变内部 ATen 原语的排列

仓库中没有独立 `rmsprop` ATen schema；因此不存在统一的 backward node。

**若 `differentiable=True`，内部可能参与反向的原语**：
- `aten::add_`
- `aten::add`
- `aten::mul_`
- `aten::addcmul_`
- `aten::lerp_`
- `aten::sqrt`
- `aten::sqrt_`
- `aten::addcdiv_`

**备注**：
- 默认训练流程把 optimizer step 当作参数更新副作用，不作为需要 backward 分析的单独可微接口。
