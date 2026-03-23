# torch API 176-180 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 176 | `torch.ormqr` | helper 公式 | `OrmqrBackward0` | `self, input2, input3: ormqr_backward(grad, result, self, input2, input3, left, transpose, grad_input_mask)` | 直接依赖：`aten::ormqr`；展开 helper 后依赖 `aten::ormqr`、`aten::tril`、`aten::diagonal`、`aten::fill_`、`aten::sum`、`aten::matmul`、`aten::mH`、`aten::narrow`、`aten::zeros_like`、`aten::cat`、`aten::copy_` | `input3` 的梯度直接通过再次调用 `aten::ormqr(..., !transpose)` 得到；`self/input2` 梯度复用 `householder_product_backward` |
| 177 | `torch.pdist` | backward ATen op | `PdistBackward0` | `_pdist_forward: self: _pdist_backward(grad, self, p, result)` | `aten::_pdist_backward`；展开后为 `aten::empty_like`，CUDA 还会走 `aten::empty`、`aten::sum.IntList_out` | `aten::pdist` 自身无条目；CIA wrapper 先做 `contiguous()` 后进入 `_pdist_forward`，真正 backward 挂在 `_pdist_forward` 上 |
| 178 | `torch.poisson` | inline 公式 | `PoissonBackward0` | `self: zeros_like(self)` | `aten::zeros_like` | `Generator` 不可微；随机采样值对输入率参数的导数在当前定义里直接置零 |
| 179 | `torch.polygamma` | inline 公式（递归调用同名 ATen op） | `PolygammaBackward0` | `self: grad * polygamma(n + 1, self)` | `aten::mul.Tensor`、`aten::polygamma` | backward 再次调用 `aten::polygamma` 的更高阶版本；若继续高阶求导会递归扩展 |
| 180 | `torch.positive` | identity / alias，无专属 backward | 无 | 无条目 | 无 | `aten::positive` 只做非 `bool` 检查后直接返回 `self`；没有新的 autograd node，梯度历史保持不变 |

## 详细分析

### 176. torch.ormqr

**反向来源类型**：helper 公式

**锁定 schema**：
`aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
  self, input2, input3: ormqr_backward(grad, result, self, input2, input3, left, transpose, grad_input_mask)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `OrmqrBackward0`
- `Functions.cpp` 调用 `ormqr_backward(...)`

**helper 展开**：
`ormqr_backward` 的逻辑分两部分：
- `input3` 梯度：`other_grad = at::ormqr(self, tau, grad, left, !transpose)`
- `self` / `tau` 梯度：复用 `householder_product_backward(...)`

因此它的**直接反向依赖**是：
- `aten::ormqr`
- helper：`householder_product_backward`

`householder_product_backward` 在 `FunctionsManual.cpp` 中进一步使用的主要 ATen op 有：
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
- `left` / `transpose` 只决定 helper 里是否交换 `grad/result` 与是否 `flip_order=true`，不改变依赖集合的大类。

---

### 177. torch.pdist

**反向来源类型**：backward ATen op

**锁定 schema**：
- 前端 wrapper：`aten::pdist(Tensor self, float p=2) -> Tensor`
- 真正承载求导：`aten::_pdist_forward(Tensor self, float p=2) -> Tensor`

`pdist` 在 `native_functions.yaml` 中没有导数定义；`derivatives.yaml` 给的是 `_pdist_forward`：
```yaml
- name: _pdist_forward(Tensor self, float p=2) -> Tensor
  self: _pdist_backward(grad, self, p, result)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `PdistBackward0`
- `Functions.cpp` 中公式为：
```cpp
_pdist_backward(grad, self, p, result)
```

**反向 ATen 依赖**：
- 直接依赖：`aten::_pdist_backward`
- 展开 `_pdist_backward` backend 实现：
  - `aten::empty_like`
  - CUDA 内部还会额外用到 `aten::empty`
  - CUDA reduction 路径还会用到 `aten::sum.IntList_out`

**备注**：
- `_pdist_backward` 自己在 `derivatives.yaml` 中被标记为 `not_implemented("_pdist_backward")`，因此这是“一阶 backward ATen op”，不再继续支持该 backward 本身的高阶导数。

---

### 178. torch.poisson

**反向来源类型**：inline 公式

**锁定 schema**：
`aten::poisson(Tensor self, Generator? generator=None) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: poisson(Tensor self, Generator? generator=None) -> Tensor
  self: zeros_like(self)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `PoissonBackward0`
- `Functions.cpp` 中为：
```cpp
auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
```

`self_info.zeros()` 是 codegen 对 `zeros_like(self)` 的类型尺寸缓存形式。

**反向 ATen 依赖**：
- `aten::zeros_like`

**备注**：
- 随机数生成器参数不参与求导。
- 当前定义里，采样输出对 rate tensor 的梯度恒为零。

---

### 179. torch.polygamma

**反向来源类型**：inline 公式（递归调用同名 ATen op）

**锁定 schema**：
`aten::polygamma(int n, Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: polygamma(int n, Tensor self) -> Tensor
  self: grad * polygamma(n + 1, self)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `PolygammaBackward0`
- `Functions.cpp` 中为：
```cpp
auto grad_result = any_grad_defined ? (grad * polygamma(n + 1, self)) : Tensor();
```

**反向 ATen 依赖**：
- `aten::polygamma` - 计算更高阶 polygamma
- `aten::mul.Tensor` - `grad * ...`

**备注**：
- 这是递归定义；若继续做高阶导，会继续沿 `polygamma(n+2, self)` 展开。
- `polygamma_` 另有 inplace 版本条目，但 `torch.polygamma` 前端对应的是 functional schema。

---

### 180. torch.positive

**反向来源类型**：identity / alias，无专属 backward

**锁定 schema**：
`aten::positive(Tensor(a) self) -> Tensor(a)`

`derivatives.yaml` 中没有 `positive` 条目，`VariableTypeEverything.cpp` 里也没有专门的 autograd wrapper。native 实现只有：
```cpp
Tensor positive(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool, ...);
  return self;
}
```

**结论**：
- `positive` 对非 `bool` Tensor 是纯 identity alias。
- 不创建新的 backward node，也不引入额外的 ATen backward 依赖。
- 输入若已有 autograd history，结果直接沿用该 history。

**反向 ATen 依赖**：无。
