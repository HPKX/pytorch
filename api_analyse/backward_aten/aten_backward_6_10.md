# torch API 6-10 反向 ATen 依赖分析

| 序号 | API | 反向来源 | Backward Node | derivatives.yaml 公式 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | `torch.aminmax` | 不可微 | 无 | 无条目；使用 `autogradNotImplementedFallback` | 无 | `aminmax` 返回 min/max 值，autograd 层注册为 not-implemented fallback；对 `requires_grad=True` 输入会报错 |
| 7 | `torch.angle` | helper 函数 | `AngleBackward0` | `self: angle_backward(grad, self)` | 复数：`aten::where`、`aten::eq.Scalar`、`aten::zeros`、`aten::mul.Tensor`、`aten::div.Tensor`、`aten::abs`、`aten::pow.Tensor_Scalar`；实数：`aten::zeros_like` | `angle_backward` 是手写 helper（`FunctionsManual.cpp`），对复数/实数分支产生不同 ATen 依赖 |
| 8 | `torch.argwhere` | 不可微 | 无 | 无条目（CIA 分解到 `nonzero`，后者标记 `output_differentiability: [False]`） | 无 | 返回整数索引张量，无梯度流；`nonzero` 在 derivatives.yaml 中显式标记输出不可微 |
| 9 | `torch.atleast_1d` | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无条目 | 前向分解产生的各 op 各自反向：dim==0 时 `aten::reshape`→`aten::view` 反向为 `aten::reshape`；dim≥1 时返回 self 无额外 op | CIA；dim==0 分解为 `aten::reshape`（→ `view` 或 `_reshape_alias`），dim≥1 直接返回 self |
| 10 | `torch.atleast_2d` | CIA 前向分解，无专属 backward | 无（各子 op 独立记录 autograd） | 无条目 | 前向分解产生的各 op 各自反向：dim==0 时同 `atleast_1d`；dim==1 时 `aten::unsqueeze` 反向为 `aten::squeeze.dim`；dim≥2 返回 self | CIA；dim==0 分解为 `aten::reshape`，dim==1 分解为 `aten::unsqueeze`，dim≥2 直接返回 self |

## 详细分析

### 6. torch.aminmax

**反向来源类型**：不可微（`autogradNotImplementedFallback`）

**Forward 路径**：`aten::aminmax` → structured backend kernel（CPU: `aminmax_out` → `aminmax_stub`；CUDA 类似）

**derivatives.yaml**：无条目。

**VariableTypeEverything.cpp** 中注册为：
```cpp
m.impl("aminmax", torch::autograd::autogradNotImplementedFallback());
```

对 `requires_grad=True` 的输入，autograd 引擎会在反向时报错（因为没有注册 backward 公式，fallback 不保留 autograd 信息）。

**反向 ATen 依赖**：无。

---

### 7. torch.angle

**反向来源类型**：helper 函数

**Forward 路径**：`aten::angle` → CPU/CUDA backend kernel

**derivatives.yaml 条目**：
```yaml
- name: angle(Tensor self) -> Tensor
  self: angle_backward(grad, self)
```

**Backward Node**：`AngleBackward0`

**`angle_backward` 实现**（`FunctionsManual.cpp:560`）：
```cpp
Tensor angle_backward(const Tensor& grad, const Tensor& self) {
  if (self.is_complex()) {
    return at::where(
        self == 0.0,
        at::zeros({}, self.options()),
        grad * self / self.abs().pow(2) *
            Scalar(c10::complex<double>{0.0, 1.0}));
  } else {
    return at::zeros_like(self, at::MemoryFormat::Preserve);
  }
}
```

**反向 ATen 依赖（复数分支）**：
- `aten::eq.Scalar` — `self == 0.0`
- `aten::zeros` — 零标量
- `aten::where` — 条件选择
- `aten::mul.Tensor` — `grad * self`
- `aten::abs` — `self.abs()`
- `aten::pow.Tensor_Scalar` — `.pow(2)`
- `aten::div.Tensor` — `… / self.abs().pow(2)`
- `aten::mul.Scalar` — 乘以复数标量 `{0, 1}`

**反向 ATen 依赖（实数分支）**：
- `aten::zeros_like` — 返回全零梯度（实数 angle 梯度恒为零）

---

### 8. torch.argwhere

**反向来源类型**：不可微

**Forward 路径**：`aten::argwhere`（CIA）→ `self.nonzero()` → `aten::nonzero` → backend kernel

**derivatives.yaml**：
- `argwhere`：无条目（CIA，分解到 `nonzero`）
- `nonzero`：显式标记 `output_differentiability: [False]`

```yaml
- name: nonzero(Tensor self) -> Tensor
  output_differentiability: [False]
```

`argwhere` 返回非零元素的索引（整数张量），不存在有意义的梯度。autograd 引擎对该 op 不建立梯度图。

**反向 ATen 依赖**：无。

---

### 9. torch.atleast_1d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- `dim == 0`：`atleast_1d` → `self.reshape({1})` → `aten::reshape`
  - `reshape` 本身也是 CIA → contiguous 时调用 `self.view(shape)` → `aten::view`；non-contiguous 时 `self._contiguous() + view` 或 `_reshape_alias`
- `dim >= 1`：`atleast_1d` → 直接返回 `self`（alias-like）

**无 derivatives.yaml 条目**。

**各子 op 反向**：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `view` | `self: grad.reshape_symint(self.sym_sizes())` | `aten::reshape` |
| `_reshape_alias` | `self: grad.reshape_symint(self.sym_sizes())` | `aten::reshape` |

dim==0 时，反向本质是把 grad 从 shape `[1]` reshape 回原始 shape `[]`（标量）。dim≥1 时无额外 op，梯度直接传递。

---

### 10. torch.atleast_2d

**反向来源类型**：`CompositeImplicitAutograd` 前向分解，无专属 backward

**Forward 分解路径**：
- `dim == 0`：`atleast_2d` → `self.reshape({1, 1})` → `aten::reshape`（同 `atleast_1d` 的 reshape 路径）
- `dim == 1`：`atleast_2d` → `self.unsqueeze(0)` → `aten::unsqueeze`
- `dim >= 2`：`atleast_2d` → 直接返回 `self`（alias-like）

**无 derivatives.yaml 条目**。

**各子 op 反向**：

| 子 op | derivatives.yaml 公式 | 反向依赖 |
| --- | --- | --- |
| `reshape`/`view` | `self: grad.reshape_symint(self.sym_sizes())` | `aten::reshape` |
| `unsqueeze` | `self: grad.squeeze(dim)` | `aten::squeeze.dim` |

dim==0 反向：grad shape `[1,1]` → reshape 回 `[]`。dim==1 反向：`grad.squeeze(0)` 去掉第 0 维。dim≥2 无额外 op。
