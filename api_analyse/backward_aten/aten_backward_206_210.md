# torch API 206-210 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 206 | `torch.triu_indices` | 不可微（factory / index tensor） | 无 | 无条目 | 无 | `aten::triu_indices` 在 `VariableTypeEverything.cpp` 注册为 `autogradNotImplementedFallback()`；返回的是整型索引张量，不建立梯度链 |
| 207 | `torch.true_divide` | CIA 别名前向；反向落到内层 `aten::div.*` | `DivBackward0` / `DivBackward1` | `true_divide` 无条目；继承 `div.Tensor: self=div_tensor_self_backward(...) , other=div_tensor_other_backward(...)`、`div.Scalar: self=div_tensor_self_backward(...)` | `aten::div.Tensor`、`aten::div.Scalar`、`aten::mul.Tensor`、`aten::neg`、`aten::conj` | `true_divide.Tensor` 直接改调 `aten::div.Tensor`；`true_divide.Scalar` 改调 `aten::div.Scalar`，其 backward 仍由 `DivBackward1` 处理 |
| 208 | `torch.vander` | CIA 前向分解，无专属 backward | 无统一 node；主要为 `CumprodBackward0` + `SliceBackward0` + `UnsqueezeBackward0` + `FlipBackward0` | `vander` 无条目；继承 `cumprod: cumprod_backward(...)`、`slice.Tensor: slice_backward_wrapper(...)`、`unsqueeze: grad.squeeze(dim)`、`flip: grad.flip(dims)` | `aten::cumprod_backward`、`aten::slice_backward`、`aten::squeeze.dim`、`aten::flip`、`aten::zeros_like` | `aten::vander` 前向组合的是 `aten::empty`、`aten::select`、`aten::fill_`、`aten::slice`、`aten::unsqueeze`、`aten::copy_`、`aten::cumprod`，以及可选的 `aten::flip`；autograd 只在真正可微的内层 op 上挂节点 |
| 209 | `torch.view_as_real` | inline 公式 + view autograd 包装 | `ViewAsRealBackward0` | `self: view_as_complex(grad.contiguous())` | `aten::contiguous`、`aten::view_as_complex` | 前向本身还是 view op，`ADInplaceOrView` 用 `as_view` 包装；反向公式显式把实部/虚部最后一维重新拼回复数 |
| 210 | `torch.vsplit` | CIA 前向分解，无专属 backward | 无统一 node；各输出各自挂 `SliceBackward0` | `vsplit.*` / `tensor_split.*` 无条目；继承 `slice.Tensor: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)` | `aten::slice_backward` | `aten::vsplit.int/array` 都是 `CompositeImplicitAutograd`，最终分解成若干个 `aten::slice.Tensor`；反向时每个分片独立 scatter 回输入 |

## 详细分析

### 206. torch.triu_indices

**对应前向 schema**：
```cpp
aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
```

**反向来源类型**：不可微 factory op

`derivatives.yaml` 中没有 `triu_indices` 条目；generated autograd 注册里直接是：
```cpp
m.impl("triu_indices", torch::autograd::autogradNotImplementedFallback());
m.impl("triu_indices.out", torch::autograd::autogradNotImplementedFallback());
```

因此 `torch.triu_indices` 不会生成 backward node，也没有反向 ATen 依赖。它返回的是整型索引张量，本身不参与梯度传播。

---

### 207. torch.true_divide

**对应前向 schema**：
```cpp
aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
```

**反向来源类型**：`CompositeImplicitAutograd` 别名前向，真实反向落到内层 `aten::div.*`

前向 dispatch 已锁定：
- `true_divide.Tensor` → `native::true_divide` → `aten::div.Tensor`
- `true_divide.Scalar` → `native::true_divide` → `aten::div.Scalar`

`true_divide` 自己在 `derivatives.yaml` 中**无条目**，所以 backward 不会挂在 `aten::true_divide.*` 名义层，而是继承内层 `div` 的公式。

**`derivatives.yaml` 条目**：
```yaml
- name: div.Tensor(Tensor self, Tensor other) -> Tensor
  self: div_tensor_self_backward(grad, other, self.scalar_type())
  other: div_tensor_other_backward(grad, self, other)

- name: div.Scalar(Tensor self, Scalar other) -> Tensor
  self: div_tensor_self_backward(grad, other, self.scalar_type())
```

**生成 node**：
- `aten::div.Tensor` → `DivBackward0`
- `aten::div.Scalar` → `DivBackward1`

`Functions.cpp` 中对应实现分别调用：
```cpp
div_tensor_self_backward(grad, other, self_scalar_type)
div_tensor_other_backward(grad, self, other)
```

`FunctionsManual.cpp` 进一步展开为：
```cpp
// self 梯度
auto result = grad / other.conj();

// other 梯度（Tensor/Tensor 才有）
auto result = -grad * ((self / other) / other).conj();
```

**反向 ATen 依赖**：
- `aten::div.Tensor` / `aten::div.Scalar`：计算 `grad / other.conj()` 与 `self / other / other`
- `aten::conj`：复数除法对分母取共轭
- `aten::mul.Tensor`：`-grad * (...)`
- `aten::neg`：other 梯度取负

**条件依赖**：
- `handle_r_to_c` 在“复数梯度回传到实数输入”场景会额外调用 `aten::real`。

**备注**：
- 对用户可见 API 来说，`torch.true_divide` 的 backward 结论应写在 `aten::div.*` 上，而不是写成 `true_divide` 自己有单独 node。

---

### 208. torch.vander

**对应前向 schema**：
```cpp
aten::vander(Tensor x, int? N=None, bool increasing=False) -> Tensor
```

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

`aten::vander` 在 `RegisterCompositeImplicitAutogradEverything.cpp` 中注册为：
```cpp
m.impl("vander", TORCH_FN(wrapper_CompositeImplicitAutograd__vander));
```

其前向实现（`TensorFactories.cpp`）直接组合内层 ATen：
```cpp
auto result = at::empty(...);
result.select(1, 0).fill_(1);
result.slice(1, 1).copy_(x.unsqueeze(1));
result.slice(1, 1).copy_(at::cumprod(result.slice(1, 1), 1));
if (!increasing) {
  return at::flip(result, {1});
}
return result;
```

因此 eager autograd 不会为 `aten::vander` 单独生成一个专属 backward node，而是把梯度挂在实际参与梯度图的子 op 上。

**主要子 op 的反向来源**：

| 子 op | backward 来源 | 反向依赖 |
| --- | --- | --- |
| `aten::cumprod` | `CumprodBackward0` | `aten::cumprod_backward` |
| `aten::slice.Tensor` | `SliceBackward0` | `aten::slice_backward` |
| `aten::unsqueeze` | `UnsqueezeBackward0` | `aten::squeeze.dim` |
| `aten::flip` | `FlipBackward0` | `aten::flip` |
| `aten::fill_.Scalar` | `zeros_like(grad)` | `aten::zeros_like` |

**对应 `derivatives.yaml` 条目**：
```yaml
- name: cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  self: cumprod_backward(grad.to(self.scalar_type()), self, dim, result)

- name: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)

- name: unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  self: grad.squeeze(dim)

- name: flip(Tensor self, int[] dims) -> Tensor
  self: grad.flip(dims)
```

**反向 ATen 依赖**：
- `aten::cumprod_backward`
- `aten::slice_backward`
- `aten::squeeze.dim`
- `aten::flip`
- `aten::zeros_like`

**备注**：
- `select + fill_` 那条把第一列写成常数 1 的路径对输入 `x` 不产生有效梯度贡献。
- 文档层面更稳定的结论是：`vander` 的 backward 由内层 `cumprod`、view/slice、以及可选的 `flip` 共同承担，而不是 `aten::vander` 自己有单独导数公式。

---

### 209. torch.view_as_real

**对应前向 schema**：
```cpp
aten::view_as_real(Tensor(a) self) -> Tensor(a)
```

**反向来源类型**：inline 公式 + view autograd 包装

`derivatives.yaml` 条目：
```yaml
- name: view_as_real(Tensor(a) self) -> Tensor(a)
  self: at::view_as_complex(grad.contiguous()) # gx0 + 1j * gx1
  result: at::view_as_real(self_t)
```

`VariableTypeEverything.cpp` 为它生成 `ViewAsRealBackward0`：
```cpp
std::shared_ptr<ViewAsRealBackward0> grad_fn;
...
return at::redispatch::view_as_real(...);
```

`Functions.cpp` 中对应 backward 实现是：
```cpp
auto grad_result = any_grad_defined ? (at::view_as_complex(grad.contiguous())) : Tensor();
```

同时，`ADInplaceOrViewTypeEverything.cpp` 说明前向本身按 view 语义包装：
```cpp
auto result = as_view(/* base */ self, /* output */ _tmp, ...);
```

**反向 ATen 依赖**：
- `aten::contiguous`：把最后一维的实/虚对变成 `view_as_complex` 需要的连续布局
- `aten::view_as_complex`：把 `[..., 2]` 的实张量重新解释成复数张量

**备注**：
- 这里虽然是 view op，但 backward 不是通用 `as_strided_backward` 口径，而是有显式的专用导数公式：`contiguous -> view_as_complex`。

---

### 210. torch.vsplit

**对应前向 schema**：
```cpp
aten::vsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
aten::vsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
```

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

`aten::vsplit.int/array` 在 `RegisterCompositeImplicitAutogradEverything.cpp` 中分别注册为：
```cpp
m.impl("vsplit.int", TORCH_FN(wrapper_CompositeImplicitAutograd_int_vsplit));
m.impl("vsplit.array", TORCH_FN(wrapper_CompositeImplicitAutograd_array_vsplit));
```

`native::vsplit` 本身只是把维度固定为 `0` 后转调 `tensor_split(...)`：
```cpp
return at::tensor_split(self, split_size, 0);
return at::tensor_split(self, split_sizes, 0);
```

再往下，`tensor_split` 最终把每个输出实现成若干个 `aten::slice.Tensor`。因此 `vsplit` 不会有统一 backward node，反向由每个 slice 各自负责。

**`derivatives.yaml` 条目**：
```yaml
- name: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  self: slice_backward_wrapper(grad, self.sym_sizes(), dim, start, end, step)
```

`VariableTypeEverything.cpp` 对应 node 是 `SliceBackward0`，`FunctionsManual.cpp` 中 helper 最终落到：
```cpp
return slice_backward_symint(grad, input_sizes, dim, start_val, end_val, step);
```

**反向 ATen 依赖**：
- `aten::slice_backward`

**备注**：
- 从 API 视角看，`torch.vsplit` 的 backward 本质就是“把各个分片梯度 scatter 回原 tensor 的第 0 维对应区间”。
