# torch API 191-195 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 191 | `torch.special.bessel_j1` | 不可微 | 无 | `self: non_differentiable` | 无 | wrapper 只 redispatch，不设置 history |
| 192 | `torch.special.bessel_y0` | 不可微 | 无 | `self: non_differentiable` | 无 | 同 191 |
| 193 | `torch.special.bessel_y1` | 不可微 | 无 | `self: non_differentiable` | 无 | 同 191 |
| 194 | `torch.special.i0` | 前端别名，真实反向挂在 `aten::i0` | `I0Backward0`（来自内层 `aten::i0`） | `i0: self: grad * special_i1(self)` | `aten::mul.Tensor`、`aten::special_i1` | `special_i0` 在 `UnaryOps.cpp` 中只是转调内层 `aten::i0`；它自己没有专属 backward 条目 |
| 195 | `torch.special.i1` | helper 公式 | `SpecialI1Backward0` | `self: i1_backward(grad, self, result)` | helper 展开后主要依赖 `aten::abs`、`aten::where`、`aten::scalar_tensor`、`aten::i0`、`aten::reciprocal`、`aten::sub`、`aten::mul.Tensor` | helper 对 `x==0` 特判，把梯度修正成 `0.5` |

## 详细分析

### 191. torch.special.bessel_j1

**反向来源类型**：不可微

**锁定 schema**：
`aten::special_bessel_j1(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: special_bessel_j1(Tensor self) -> Tensor
  self: non_differentiable
```

`VariableTypeEverything.cpp` 中 wrapper 只 redispatch 并直接返回结果，没有 `grad_fn`。

**反向 ATen 依赖**：无。

---

### 192. torch.special.bessel_y0

**反向来源类型**：不可微

**锁定 schema**：
`aten::special_bessel_y0(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: special_bessel_y0(Tensor self) -> Tensor
  self: non_differentiable
```

与 `special_bessel_j1` 相同，generated wrapper 只 redispatch，不创建 backward node。

**反向 ATen 依赖**：无。

---

### 193. torch.special.bessel_y1

**反向来源类型**：不可微

**锁定 schema**：
`aten::special_bessel_y1(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: special_bessel_y1(Tensor self) -> Tensor
  self: non_differentiable
```

**反向 ATen 依赖**：无。

---

### 194. torch.special.i0

**反向来源类型**：前端别名，真实反向挂在 `aten::i0`

**锁定 schema**：
- 前端入口：`aten::special_i0(Tensor self) -> Tensor`
- 实际可微 schema：`aten::i0(Tensor self) -> Tensor`

`UnaryOps.cpp` 中实现明确是别名：
```cpp
Tensor special_i0(const Tensor& self) { return self.i0(); }
```

因此 `torch.special.i0` 自己没有独立 `derivatives.yaml` 条目；真正有导数定义的是 `aten::i0`：
```yaml
- name: i0(Tensor self) -> Tensor
  self: grad * at::special_i1(self)
```

**生成代码**：
- `VariableTypeEverything.cpp` 为 `aten::i0` 生成 `I0Backward0`
- forward AD 公式同样显式调用 `at::special_i1(self_p)`

**反向 ATen 依赖**：
- `aten::special_i1`
- `aten::mul.Tensor`

**备注**：
- 分析 `torch.special.i0` 的 backward 时，必须把名字 canonicalize 到内层 `aten::i0`。

---

### 195. torch.special.i1

**反向来源类型**：helper 公式

**锁定 schema**：
`aten::special_i1(Tensor self) -> Tensor`

**derivatives.yaml 条目**：
```yaml
- name: special_i1(Tensor self) -> Tensor
  self: i1_backward(grad, self, result)
```

**生成代码**：
- `VariableTypeEverything.cpp` 生成 `SpecialI1Backward0`
- `Functions.cpp` 调用 `i1_backward(...)`

**helper 展开**（`FunctionsManual.cpp`）：
```cpp
auto self_is_not_tiny = self.abs() > eps;
auto safe_self = at::where(
    self_is_not_tiny, self, at::scalar_tensor(eps, self.options()));
auto gradx = (safe_self.i0() - (result * safe_self.reciprocal()));
return grad *
    at::where(self_is_not_tiny, gradx, at::scalar_tensor(0.5, self.options()));
```

**反向 ATen 依赖**：
- `aten::abs`
- `aten::where`
- `aten::scalar_tensor`
- `aten::i0`
- `aten::reciprocal`
- `aten::sub`
- `aten::mul.Tensor`

**备注**：
- `x == 0` 时真实导数是 `0.5`，helper 专门用 `where` 避开 NaN。
