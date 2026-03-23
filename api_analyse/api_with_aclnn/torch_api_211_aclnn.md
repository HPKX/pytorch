# torch API 211 aclnn 接入分析

## 接口 211：`torch.vstack`

**A5 支持结论：✅ 可在 A5 上支持**

### 正向依赖

| ATen 接口 | 分类 | aclnn 接入 | aclnn 接口名 | 备注 |
|-----------|------|-----------|-------------|------|
| `aten::vstack` | 计算接口 | **否** | - | op_plugin 中未找到实现，cat(dim=0) 的封装 |
| `aten::vstack.out` | 计算接口 | **否** | - | op_plugin 中未找到实现 |
| `aten::atleast_2d.Sequence` | view | N/A | - | composite_view |
| `aten::unsqueeze` | view | N/A | - | view 类接口 |
| `aten::reshape` | view | N/A | - | composite_view |
| `aten::cat` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |
| `aten::cat.out` | 计算接口 | 是 | `aclnnCat` | opapi/CatKernelNpuOpApi.cpp |

### 反向依赖

| ATen 接口 | 分类 | aclnn 接入 | 备注 |
|-----------|------|-----------|------|
| `aten::narrow` | view | N/A | composite_view |
| `aten::squeeze.dim` | view | N/A | view 类接口 |
| `aten::reshape` | view | N/A | composite_view |

> `aten::vstack` 和 `aten::vstack.out` 是 CIA composite 函数，前向实际分解为 atleast_2d（view）、unsqueeze（view）、reshape（view）、cat（aclnn=是）。composite 函数不需要独立 aclnn 实现，实际计算由 cat 承担。反向依赖全部为 view 操作。

---

## 汇总

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 211 | `torch.vstack` | ✅ | `aclnnCat` | 全为 view，无需 aclnn | - |
