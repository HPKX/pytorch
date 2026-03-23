# torch API 211 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 211 | `torch.vstack` | CIA 前向分解，无专属 backward | 无统一 node；主要为 `CatBackward0`，并可能带 `UnsqueezeBackward0` / reshape view backward | `vstack` 无条目；继承 `cat: cat_tensors_backward(...)`、`unsqueeze: grad.squeeze(dim)`；`reshape` 本身无直接条目，view 情况走 `_reshape_alias: grad.reshape_symint(self.sym_sizes())` | `aten::narrow`、`aten::squeeze.dim`、`aten::reshape` | `aten::vstack` / `aten::vstack.out` 都是 `CompositeImplicitAutograd`；真实可微主链是 `atleast_2d.Sequence` 内部产生的 `unsqueeze/reshape` 与最后的 `aten::cat` |

## 详细分析

### 211. torch.vstack

**对应前向 schema**：
```cpp
aten::vstack(Tensor[] tensors) -> Tensor
aten::vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
```

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

`aten::vstack` / `aten::vstack.out` 在 `RegisterCompositeImplicitAutogradEverything.cpp` 中注册为：
```cpp
m.impl("vstack", TORCH_FN(wrapper_CompositeImplicitAutograd__vstack));
m.impl("vstack.out", TORCH_FN(wrapper_CompositeImplicitAutograd_out_vstack_out));
```

其原生实现（`TensorShape.cpp`）很直接：
```cpp
Tensor vstack(TensorList tensors) {
  auto rep = at::atleast_2d(tensors);
  return at::cat(rep, 0);
}

Tensor& vstack_out(TensorList tensors, Tensor& result) {
  auto rep = at::atleast_2d(tensors);
  return at::cat_out(result, rep, 0);
}
```

而 `atleast_2d(TensorList)`（`TensorTransformations.cpp`）会对每个输入做：
```cpp
case 0: return self.reshape({1, 1});
case 1: return self.unsqueeze(0);
default: return self;
```

因此 eager backward 的实际挂点是：
- `aten::cat` → `CatBackward0`
- 若输入有 1D tensor：`aten::unsqueeze` → `UnsqueezeBackward0`
- 若输入有 0D tensor：`reshape` 若走 view 分支，则表现为 reshape/view backward（`_reshape_alias` / `ViewBackward0` 一类）

**`derivatives.yaml` 条目**：
```yaml
- name: cat(Tensor[] tensors, int dim=0) -> Tensor
  tensors: cat_tensors_backward(grad, to_args_sizes_symint(tensors), to_args_scalartypes(tensors), dim)

- name: unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  self: grad.squeeze(dim)

- name: _reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)
  self: grad.reshape_symint(self.sym_sizes())
```

`Functions.cpp` 中 `CatBackward0` 调用：
```cpp
auto grad_result = cat_tensors_backward(
    grad, tensors_args_sizes_symint, tensors_args_scalartypes, dim);
```

`FunctionsManual.cpp` 里 `cat_tensors_backward(...)` 的核心实现是按输入大小切回各段：
```cpp
grad_inputs[i] = grad_val.narrow_symint(dim, accumulate - size, size);
```

所以 `cat` 的 backward 主要依赖 `aten::narrow`；而 `unsqueeze` 的 backward 依赖 `aten::squeeze.dim`；0D 输入经 `reshape({1,1})` 进入 `vstack` 时，反向再用 `aten::reshape` 回原形状。

**反向 ATen 依赖**：
- `aten::narrow`
- `aten::squeeze.dim`
- `aten::reshape`

**`out=` 情况**：
- `torch.vstack(..., out=out)` 最终走的是 `aten::cat.out`。
- 当任一输入或 `out` 需要 grad 时，`cat.out` 属于不可微 out 变体，会直接报错；只有无梯度场景才会继续 redispatch 到 backend。

**备注**：
- 文档口径上，`torch.vstack` 应归为“CIA 组合接口，主 backward 是 `CatBackward0`，再叠加 `atleast_2d` 内部 view 变换的 backward”，而不是单独给 `vstack` 发明一个不存在的专属 node。
