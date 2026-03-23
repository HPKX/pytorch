# ATen 接口汇总

按每 5 个接口为一个任务分批整理 `api_analyse/forward_aten/*.md` 与 `api_analyse/backward_aten/*.md` 中的结果。

当前已完成任务 1-43。

## 任务进度

| 任务 | 接口序号 | 状态 | 来源文件 |
| --- | --- | --- | --- |
| 任务 1 | 1-5 | 已完成 | `forward_aten/torch_api_1_5.md`、`backward_aten/aten_backward_1_5.md` |
| 任务 2 | 6-10 | 已完成 | `forward_aten/torch_api_6_10.md`、`backward_aten/aten_backward_6_10.md` |
| 任务 3 | 11-15 | 已完成 | `forward_aten/torch_api_11_15.md`、`backward_aten/aten_backward_11_15.md` |
| 任务 4 | 16-20 | 已完成 | `forward_aten/torch_api_16_20.md`、`backward_aten/aten_backward_16_20.md` |
| 任务 5 | 21-25 | 已完成 | `forward_aten/torch_api_21_25.md`、`backward_aten/aten_backward_21_25.md` |
| 任务 6 | 26-30 | 已完成 | `forward_aten/torch_api_26_30.md`、`backward_aten/aten_backward_26_30.md` |
| 任务 7 | 31-35 | 已完成 | `forward_aten/torch_api_31_35.md`、`backward_aten/aten_backward_31_35.md` |
| 任务 8 | 36-40 | 已完成 | `forward_aten/torch_api_36_40.md`、`backward_aten/aten_backward_36_40.md` |
| 任务 9 | 41-45 | 已完成 | `forward_aten/torch_api_41_45.md`、`backward_aten/aten_backward_41_45.md` |
| 任务 10 | 46-50 | 已完成 | `forward_aten/torch_api_46_50.md`、`backward_aten/aten_backward_46_50.md` |
| 任务 11 | 51-55 | 已完成 | `forward_aten/torch_api_51_55.md`、`backward_aten/aten_backward_51_55.md` |
| 任务 12 | 56-60 | 已完成 | `forward_aten/torch_api_56_60.md`、`backward_aten/aten_backward_56_60.md` |
| 任务 13 | 61-65 | 已完成 | `forward_aten/torch_api_61_65.md`、`backward_aten/aten_backward_61_65.md` |
| 任务 14 | 66-70 | 已完成 | `forward_aten/torch_api_66_70.md`、`backward_aten/aten_backward_66_70.md` |
| 任务 15 | 71-75 | 已完成 | `forward_aten/torch_api_71_75.md`、`backward_aten/aten_backward_71_75.md` |
| 任务 16 | 76-80 | 已完成 | `forward_aten/torch_api_76_80.md`、`backward_aten/aten_backward_76_80.md` |
| 任务 17 | 81-85 | 已完成 | `forward_aten/torch_api_81_85.md`、`backward_aten/aten_backward_81_85.md` |
| 任务 18 | 86-90 | 已完成 | `forward_aten/torch_api_86_90.md`、`backward_aten/aten_backward_86_90.md` |
| 任务 19 | 91-95 | 已完成 | `forward_aten/torch_api_91_95.md`、`backward_aten/aten_backward_91_95.md` |
| 任务 20 | 96-100 | 已完成 | `forward_aten/torch_api_96_100.md`、`backward_aten/aten_backward_96_100.md` |
| 任务 21 | 101-105 | 已完成 | `forward_aten/torch_api_101_105.md`、`backward_aten/aten_backward_101_105.md` |
| 任务 22 | 106-110 | 已完成 | `forward_aten/torch_api_106_110.md`、`backward_aten/aten_backward_106_110.md` |
| 任务 23 | 111-115 | 已完成 | `forward_aten/torch_api_111_115.md`、`backward_aten/aten_backward_111_115.md` |
| 任务 24 | 116-120 | 已完成 | `forward_aten/torch_api_116_120.md`、`backward_aten/aten_backward_116_120.md` |
| 任务 25 | 121-125 | 已完成 | `forward_aten/torch_api_121_125.md`、`backward_aten/aten_backward_121_125.md` |
| 任务 26 | 126-130 | 已完成 | `forward_aten/torch_api_126_130.md`、`backward_aten/aten_backward_126_130.md` |
| 任务 27 | 131-135 | 已完成 | `forward_aten/torch_api_131_135.md`、`backward_aten/aten_backward_131_135.md` |
| 任务 28 | 136-140 | 已完成 | `forward_aten/torch_api_136_140.md`、`backward_aten/aten_backward_136_140.md` |
| 任务 29 | 141-145 | 已完成 | `forward_aten/torch_api_141_145.md`、`backward_aten/aten_backward_141_145.md` |
| 任务 30 | 146-150 | 已完成 | `forward_aten/torch_api_146_150.md`、`backward_aten/aten_backward_146_150.md` |
| 任务 31 | 151-155 | 已完成 | `forward_aten/torch_api_151_155.md`、`backward_aten/aten_backward_151_155.md` |
| 任务 32 | 156-160 | 已完成 | `forward_aten/torch_api_156_160.md`、`backward_aten/aten_backward_156_160.md` |
| 任务 33 | 161-165 | 已完成 | `forward_aten/torch_api_161_165.md`、`backward_aten/aten_backward_161_165.md` |
| 任务 34 | 166-170 | 已完成 | `forward_aten/torch_api_166_170.md`、`backward_aten/aten_backward_166_170.md` |
| 任务 35 | 171-175 | 已完成 | `forward_aten/torch_api_171_175.md`、`backward_aten/aten_backward_171_175.md` |
| 任务 36 | 176-180 | 已完成 | `forward_aten/torch_api_176_180.md`、`backward_aten/aten_backward_176_180.md` |
| 任务 37 | 181-185 | 已完成 | `forward_aten/torch_api_181_185.md`、`backward_aten/aten_backward_181_185.md` |
| 任务 38 | 186-190 | 已完成 | `forward_aten/torch_api_186_190.md`、`backward_aten/aten_backward_186_190.md` |
| 任务 39 | 191-195 | 已完成 | `forward_aten/torch_api_191_195.md`、`backward_aten/aten_backward_191_195.md` |
| 任务 40 | 196-200 | 已完成 | `forward_aten/torch_api_196_200.md`、`backward_aten/aten_backward_196_200.md` |
| 任务 41 | 201-205 | 已完成 | `forward_aten/torch_api_201_205.md`、`backward_aten/aten_backward_201_205.md` |
| 任务 42 | 206-210 | 已完成 | `forward_aten/torch_api_206_210.md`、`backward_aten/aten_backward_206_210.md` |
| 任务 43 | 211 | 已完成 | `forward_aten/torch_api_211.md`、`backward_aten/aten_backward_211.md` |

## 任务 1：接口 1-5

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | `torch.absolute` | `aten::absolute`、`aten::absolute.out`、`aten::abs`、`aten::abs.out` | `aten::mul.Tensor`、`aten::sgn` | `absolute` 前向主体分解到 `aten::abs` |
| 2 | `torch.addcdiv` | `aten::addcdiv`、`aten::addcdiv.out` | `aten::mul.Tensor`、`aten::mul.Scalar`、`aten::div.Tensor`、`aten::neg`、`aten::conj` |  |
| 3 | `torch.addcmul` | `aten::addcmul`、`aten::addcmul.out` | `aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj` |  |
| 4 | `torch.addr` | `aten::addr`、`aten::addr.out`、`aten::outer`、`aten::mul`、`aten::add`、`aten::to` | `aten::mul.Scalar`、`aten::mv`、`aten::conj`、`aten::t` | `maybe_multiply` 在 scalar 为 1 时不会产生 `aten::mul.Scalar` |
| 5 | `torch.adjoint` | `aten::adjoint`、`aten::transpose.int`、`aten::as_strided`、`aten::conj`、`aten::_conj`、`aten::alias` | `aten::transpose`、`aten::conj`、`as_strided_backward` | `adjoint` 是 `CompositeImplicitAutograd`；反向由分解后的子 op 独立承担 |

## 任务 2：接口 6-10

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 6 | `torch.aminmax` | `aten::aminmax` | 无 | 不可微 |
| 7 | `torch.angle` | `aten::angle` | `aten::where`、`aten::eq.Scalar`、`aten::zeros`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::div.Tensor`、`aten::abs`、`aten::pow.Tensor_Scalar`、`aten::zeros_like` | 复数与实数分支的反向依赖不同 |
| 8 | `torch.argwhere` | `aten::argwhere`、`aten::nonzero` | 无 | 不可微 |
| 9 | `torch.atleast_1d` | `aten::atleast_1d`、`aten::reshape` | `aten::reshape` | `CompositeImplicitAutograd` 分解；`dim>=1` 时直接返回 self |
| 10 | `torch.atleast_2d` | `aten::atleast_2d`、`aten::reshape`、`aten::unsqueeze` | `aten::reshape`、`aten::squeeze.dim` | `CompositeImplicitAutograd` 分解；不同维度走不同子 op |

## 任务 3：接口 11-15

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 11 | `torch.atleast_3d` | `aten::atleast_3d`、`aten::reshape`、`aten::unsqueeze` | `aten::reshape`、`aten::squeeze.dim` | `CompositeImplicitAutograd` 分解；`dim==1` 走两次 `unsqueeze`，反向对应两次 `squeeze.dim` |
| 12 | `torch.bartlett_window` | `aten::bartlett_window`、`aten::bartlett_window.periodic`、`aten::empty`、`aten::ones`、`aten::arange`、`aten::mul_`、`aten::narrow`、`aten::add_` | 无 | 工厂函数，不可微 |
| 13 | `torch.bitwise_left_shift` | `aten::bitwise_left_shift.Tensor`、`aten::bitwise_left_shift.Tensor_Scalar`、`aten::bitwise_left_shift.Scalar_Tensor` | 无 | 整数位运算，不可微 |
| 14 | `torch.bitwise_right_shift` | `aten::bitwise_right_shift.Tensor`、`aten::bitwise_right_shift.Tensor_Scalar`、`aten::bitwise_right_shift.Scalar_Tensor` | 无 | 整数位运算，不可微 |
| 15 | `torch.blackman_window` | `aten::blackman_window`、`aten::blackman_window.periodic`、`aten::empty`、`aten::ones`、`aten::arange`、`aten::mul`、`aten::mul_`、`aten::cos_`、`aten::narrow` | 无 | 工厂函数，不可微 |

## 任务 4：接口 16-20

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 16 | `torch.block_diag` | `aten::block_diag`、`aten::expand`、`aten::zeros`、`aten::slice.Tensor`、`aten::copy_` | `aten::slice.Tensor`、`aten::squeeze.dim`、`aten::zeros`、`aten::real` | 0D/1D 输入会先 expand 成 2D |
| 17 | `torch.bucketize` | `aten::bucketize.Tensor`、`aten::empty` | 无 | 不可微；backend 内部复用 `searchsorted_out_*`，但不是新的 dispatcher hop |
| 18 | `torch.cholesky` | `aten::cholesky`、`aten::empty`、`aten::_linalg_check_errors`、`aten::tril_`、`aten::triu_` | `aten::mH`、`aten::matmul`、`aten::tril`、`aten::mul.Scalar`、`aten::add.Tensor`、`aten::linalg_solve_triangular` | 已 deprecated，推荐 `torch.linalg.cholesky` |
| 19 | `torch.cholesky_solve` | `aten::cholesky_solve`、`aten::_cholesky_solve_helper` | `aten::cholesky_solve`、`aten::matmul`、`aten::mH`、`aten::add.Tensor`、`aten::neg` | 反向里 self 梯度递归调用 `cholesky_solve` |
| 20 | `torch.clip` | `aten::clip`、`aten::clip.Tensor`、`aten::clamp`、`aten::clamp.Tensor` | `aten::scalar_tensor`、`aten::ge.Scalar`、`aten::le.Scalar`、`aten::ge.Tensor`、`aten::le.Tensor`、`aten::logical_and_`、`aten::where` | CIA 别名，backward 挂在 `clamp` 上 |

## 任务 5：接口 21-25

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 21 | `torch.column_stack` | `aten::column_stack`、`aten::reshape`、`aten::hstack`、`aten::atleast_1d.Sequence`、`aten::cat` | `aten::reshape`、`aten::narrow`、`aten::slice.Tensor` | `column_stack` 是 CIA 分解，前向由 `reshape`、`hstack`、`cat` 组成 |
| 22 | `torch.combinations` | `aten::combinations`、`aten::empty`、`aten::meshgrid`、`aten::arange`、`aten::full`、`aten::masked_select`、`aten::stack` | `aten::masked_select_backward`、`aten::zeros_like`、`aten::expand`、`aten::masked_scatter_`、`aten::select.int` | `combinations` 是 CIA 分解，梯度主要来自 `masked_select` 和 `stack` |
| 23 | `torch.cond` | 无 | 无 | 不是传统 ATen 算子；属于 Python 层 `HigherOrderOperator`，正反向依赖取决于 `true_fn` / `false_fn` 内部算子 |
| 24 | `torch.conj` | `aten::conj`、`aten::conj_physical`、`aten::_conj`、`aten::_conj_physical` | `aten::conj` | `conj` 是 CIA 分解，复数张量会进入 `_conj` 路径 |
| 25 | `torch.copysign` | `aten::copysign.Tensor`、`aten::copysign.Scalar` | `aten::div.Tensor`、`aten::eq.Scalar`、`aten::masked_fill_.Scalar`、`aten::mul.Tensor`、`aten::zeros_like` | Tensor 重载与 Scalar 重载的反向实现不同，`other` 的梯度为零 |

## 任务 6：接口 26-30

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 26 | `torch.cov` | `aten::cov`、`aten::view`、`aten::mul`、`aten::sum`、`aten::scalar_tensor`、`aten::unsqueeze`、`aten::sub`、`aten::t`、`aten::conj`、`aten::mm`、`aten::real`、`aten::zeros_like`、`aten::complex`、`aten::true_divide`、`aten::squeeze` | `aten::reshape`、`aten::mul.Tensor`、`aten::conj`、`aten::expand`、`aten::squeeze.dim`、`aten::neg`、`aten::t`、`aten::mm`、`aten::div.Tensor`、`aten::unsqueeze`、`aten::real`、`aten::complex`、`aten::zeros_like` | `CompositeImplicitAutograd`，梯度穿透到基础算术和线性代数子 op |
| 27 | `torch.deg2rad` | `aten::deg2rad`、`aten::empty_like`、`aten::deg2rad.out`、`aten::mul.out` | `aten::mul.Scalar` | `requires_grad=True` 时走专门 backward helper |
| 28 | `torch.diag_embed` | `aten::diag_embed`、`aten::zeros`、`aten::diagonal`、`aten::as_strided`、`aten::copy_` | `aten::diagonal`、`aten::as_strided` | 反向就是从对角线视图中提取梯度 |
| 29 | `torch.diagflat` | `aten::diagflat`、`aten::contiguous`、`aten::view`、`aten::diag`、`aten::diag_embed`、`aten::zeros`、`aten::diagonal`、`aten::as_strided`、`aten::copy_` | `aten::reshape`、`aten::diagonal` | `contiguous` 反向为 identity，主体梯度由 `view` 与 `diag_embed` 传递 |
| 30 | `torch.diagonal` | `aten::diagonal`、`aten::as_strided` | `aten::diagonal_backward`、`aten::zeros`、`aten::diagonal`、`aten::copy_` | `view` op，反向通过 `diagonal_backward` 创建零矩阵再写回对角线 |

## 任务 7：接口 31-35

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 31 | `torch.diagonal_scatter` | `aten::diagonal_scatter`、`aten::diagonal`、`aten::copy_` | `aten::diagonal_scatter`、`aten::zeros_like`、`aten::diagonal` | `CEA non-functional`，self 梯度用零张量 scatter，src 梯度取对角线 |
| 32 | `torch.digamma` | `aten::digamma` | `aten::mul.Tensor`、`aten::polygamma` | 导数是 `polygamma(1, self)` |
| 33 | `torch.dist` | `aten::dist`、`aten::sub`、`aten::norm` | `aten::sub.Tensor`、`aten::neg`、`aten::sgn`、`aten::mul.Tensor`、`aten::div.Tensor`、`aten::masked_fill_`、`aten::eq.Scalar`、`aten::eq.Tensor`、`aten::abs`、`aten::isnan`、`aten::logical_or`、`aten::sum` | `p` 分支不同，`norm_backward` 会走不同 helper 路径 |
| 34 | `torch.distribution.gamma.Gamma` | `aten::broadcast_tensors`、`aten::expand`、`aten::_standard_gamma`、`aten::div`、`aten::xlogy`、`aten::sub`、`aten::mul`、`aten::lgamma` | `aten::mul.Tensor`、`aten::_standard_gamma_grad`、`aten::div.Tensor`、`aten::neg` | Python 分布类；`rsample()` 的核心梯度挂在 `_standard_gamma` 上，`log_prob()` 路径由内部基础 ATen op 各自反向承担 |
| 35 | `torch.distribution.laplace.Laplace` | `aten::broadcast_tensors`、`aten::uniform_`、`aten::rand`、`aten::sign`、`aten::abs`、`aten::clamp`、`aten::neg`、`aten::log1p`、`aten::mul`、`aten::sub`、`aten::log`、`aten::div` | `aten::sgn`、`aten::mul.Tensor`、`aten::neg`、`aten::add.Scalar`、`aten::div.Tensor`、`aten::mul.Scalar` | Python 分布类；梯度通过重参数化路径上的基础 ATen op 传播 |

## 任务 8：接口 36-40

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 36 | `torch.distribution.uniform.Uniform` | `aten::rand`、`aten::uniform_`、`aten::mul`、`aten::sub`、`aten::add`、`aten::le.Tensor`、`aten::gt.Tensor`、`aten::mul.Tensor`、`aten::log`、`aten::sub.Tensor`、`aten::div.Tensor`、`aten::clamp` | `aten::mul.Tensor`、`aten::neg`、`aten::div.Tensor` | Python 分布类；`rand`/`uniform_` 采样本身不可微，梯度通过 `low + rand * (high - low)` 传播 |
| 37 | `torch.dsplit` | `aten::dsplit.int`、`aten::dsplit.array`、`aten::tensor_split.sections`、`aten::tensor_split.indices`、`aten::slice.Tensor` | `aten::zeros`、`aten::slice.Tensor`、`aten::copy_` | `CompositeImplicitAutograd`，autograd 穿透到 `slice` 的 backward |
| 38 | `torch.dstack` | `aten::dstack`、`aten::dstack.out`、`aten::atleast_3d`、`aten::cat`、`aten::cat.out` | `aten::narrow`、`aten::slice.Tensor`、`aten::zeros`、`aten::real`、`aten::squeeze.dim`、`aten::reshape` | `cat` 负责主梯度切分，`atleast_3d` 的 `unsqueeze`/`reshape` 反向对应 `squeeze.dim`/`reshape`，复数场景可能额外走 `real` |
| 39 | `torch.fliplr` | `aten::fliplr`、`aten::flip` | `aten::flip` | `flip` 自身是逆操作 |
| 40 | `torch.flipud` | `aten::flipud`、`aten::flip` | `aten::flip` | `flip` 自身是逆操作 |

## 任务 9：接口 41-45

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 41 | `torch.fmax` | `aten::fmax` | `aten::ge.Tensor`、`aten::isnan`、`aten::logical_or_`、`aten::logical_not_`、`aten::masked_fill_.Scalar` | NaN 语义参与梯度 mask 选择 |
| 42 | `torch.fmod` | `aten::fmod.Scalar`、`aten::fmod.Tensor` | `aten::div.Tensor_mode`、`aten::mul.Tensor`、`aten::neg` | Scalar 重载 self 梯度直通；Tensor 重载的 other 梯度需要 trunc 除法 |
| 43 | `torch.gather` | `aten::gather` | `aten::gather_backward`、`aten::new_zeros`、`aten::scatter_add_`、`aten::scatter_add`、`aten::_gather_sparse_backward` | 默认路径用 `scatter_add_`，`sparse_grad=True` 走专门 sparse backward |
| 44 | `torch.gcd` | `aten::gcd` | 无 | 整数专用，不可微 |
| 45 | `torch.ge` | `aten::ge.Tensor`、`aten::ge.Scalar` | 无 | `output_differentiability: [False]`，结果为 bool 张量 |

## 任务 10：接口 46-50

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 46 | `torch.geqrf` | `aten::geqrf`、`aten::empty`、`aten::geqrf.a` | 无 | `not_implemented("geqrf")`，backward 调用会抛异常 |
| 47 | `torch.ger` | `aten::ger`、`aten::ger.out`、`aten::outer`、`aten::outer.out`、`aten::reshape`、`aten::mul.Tensor`、`aten::mul.out` | `aten::reshape`、`aten::mul.Tensor`、`aten::conj` | 详细分析补出了 `reshape`，以 CIA 分解后的子 op 反向为准 |
| 48 | `torch.hamming_window` | `aten::hamming_window`、`aten::hamming_window.periodic_alpha_beta`、`aten::empty`、`aten::ones`、`aten::arange`、`aten::mul_.Scalar`、`aten::cos_`、`aten::add_.Scalar`、`aten::narrow` | 无 | 工厂函数，不可微 |
| 49 | `torch.hann_window` | `aten::hann_window`、`aten::hann_window.periodic`、`aten::empty`、`aten::ones`、`aten::arange`、`aten::mul_.Scalar`、`aten::cos_`、`aten::add_.Scalar`、`aten::narrow` | 无 | 工厂函数，不可微 |
| 50 | `torch.heaviside` | `aten::heaviside`、`aten::heaviside.out`、`aten::heaviside_` | 无 | structured op，`autogradNotImplementedFallback` |

## 任务 11：接口 51-55

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 51 | `torch.hsplit` | `aten::hsplit.int`、`aten::hsplit.array`、`aten::tensor_split.sections`、`aten::tensor_split.indices`、`aten::slice.Tensor` | `aten::zeros`、`aten::slice.Tensor`、`aten::copy_` | CIA 前向分解，反向由 `slice` 承担 |
| 52 | `torch.hstack` | `aten::hstack`、`aten::hstack.out`、`aten::atleast_1d.Sequence`、`aten::cat`、`aten::cat.out` | `aten::narrow`、`aten::squeeze.dim`、`aten::zeros`、`aten::real` | 详细分析补出了 `squeeze.dim` 和 `zeros`，复数路径还会走 `real` |
| 53 | `torch.hypot` | `aten::hypot` | `aten::mul.Tensor`、`aten::div.Tensor` | inline 公式，梯度直接按 `self / result`、`other / result` 计算 |
| 54 | `torch.i0` | `aten::i0` | `aten::mul.Tensor`、`aten::special_i1` | 反向使用 `I1` |
| 55 | `torch.igamma` | `aten::igamma` | `aten::sub.Scalar`、`aten::log`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::lgamma`、`aten::exp` | `self` 梯度为 `not_implemented("igamma: input")` |

## 任务 12：接口 56-60

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 56 | `torch.igammac` | `aten::igammac`、`aten::igammac.out` | `aten::neg`、`aten::mul.Tensor`、`aten::exp`、`aten::sub.Scalar`、`aten::log`、`aten::sub.Tensor`、`aten::lgamma` | `self` 梯度为 `not_implemented("igammac: input")` |
| 57 | `torch.inner` | `aten::inner`、`aten::mul.Tensor`、`aten::tensordot`、`aten::permute`、`aten::reshape`、`aten::mm`、`aten::dot` | `aten::mul.Tensor`、`aten::mm`、`aten::dot`、`aten::permute`、`aten::reshape` | 无专属 backward，反向由子 op 分别承担 |
| 58 | `torch.is_complex` | `aten::is_complex` | 无 | 返回 `bool`，不可微 |
| 59 | `torch.is_floating_point` | `aten::is_floating_point` | 无 | 返回 `bool`，不可微 |
| 60 | `torch.is_nonzero` | `aten::is_nonzero`、`aten::item`、`aten::_local_scalar_dense` | 无 | 返回 `bool`，不可微 |

## 任务 13：接口 61-65

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 61 | `torch.isnan` | `aten::isnan`、`aten::ne.Tensor` | 无 | `non_differentiable` |
| 62 | `torch.isposinf` | `aten::isposinf` | 无 | 返回 `bool`，不可微 |
| 63 | `torch.isreal` | `aten::isreal`、`aten::ones_like`、`aten::imag`、`aten::eq.Scalar` | 无 | 返回 `bool`，不可微 |
| 64 | `torch.kaiser_window` | `aten::kaiser_window`、`aten::kaiser_window.periodic`、`aten::kaiser_window.beta`、`aten::empty`、`aten::ones`、`aten::arange`、`aten::narrow` | 无 | 工厂函数，不可微 |
| 65 | `torch.kron` | `aten::kron`、`aten::kron.out`、`aten::_unsafe_view`、`aten::mul.Tensor`、`aten::mul.out` | `aten::reshape`、`aten::mul.Tensor`、`aten::conj` | CIA 分解，反向由 `_unsafe_view` 和 `mul.Tensor` 子 op 承担 |

## 任务 14：接口 66-70

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 66 | `torch.lcm` | `aten::lcm` | 无 | 整数专用 op，不可微 |
| 67 | `torch.ldexp` | `aten::ldexp.Tensor`、`aten::pow`、`aten::mul` | `aten::pow.Scalar`、`aten::conj`、`aten::mul.Tensor`、`aten::mul.Scalar` | 非整数指数会改走 `pow + mul` 路径 |
| 68 | `torch.linalg.eigvals` | `aten::linalg_eigvals`、`aten::_linalg_eigvals`、`aten::linalg_eig` | `aten::linalg_solve`、`aten::matmul`、`aten::mH`、`aten::conj`、`aten::unsqueeze`、`aten::div.Tensor`、`aten::diagonal`、`aten::copy_`、`aten::real` | `requires_grad=True` 时前向会显式改调 `linalg_eig` |
| 69 | `torch.linalg.pinv` | `aten::linalg_pinv`、`aten::linalg_pinv.atol_rtol_float`、`aten::linalg_pinv.atol_rtol_tensor`、`aten::svd`、`aten::narrow`、`aten::where`、`aten::matmul`、`aten::linalg_eigh`、`aten::amax` | `aten::mH`、`aten::matmul`、`aten::neg`、`aten::add.Tensor`、`aten::sub.Tensor` | 反向挂在内部 `atol_rtol_tensor` 重载上 |
| 70 | `torch.linalg.vecdot` | `aten::linalg_vecdot`、`aten::vdot`、`aten::conj`、`aten::mul`、`aten::sum` | `aten::conj`、`aten::mul.Tensor`、`aten::expand` | 1D 走 `vdot`，高维路径还会经过 `sum` 的 backward |

## 任务 15：接口 71-75

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 71 | `torch.logcumsumexp` | `aten::logcumsumexp`、`aten::_logcumsumexp` | `aten::flip`、`aten::logcumsumexp`、`aten::abs`、`aten::log`、`aten::where`、`aten::scalar_tensor`、`aten::sub.Tensor`、`aten::add.Tensor`、`aten::exp`、`aten::conj`、`aten::gt.Scalar`、`aten::lt.Scalar` | 实数和复数分支的 backward 公式不同 |
| 72 | `torch.logdet` | `aten::logdet`、`aten::linalg_slogdet`、`aten::_linalg_slogdet`、`aten::linalg_lu_factor_ex.out`、`aten::sgn`、`aten::prod`、`aten::mul.out`、`aten::abs`、`aten::log_`、`aten::sum.IntList_out`、`aten::where`、`aten::log`、`aten::add` | `aten::imag`、`aten::conj`、`aten::mul.Tensor`、`aten::diag_embed`、`aten::unsqueeze`、`aten::expand_as`、`aten::mT`、`aten::linalg_lu_solve`、`aten::linalg_solve`、`aten::mH`、`aten::sub.Tensor` | `logdet` 的可导路径实际落在 `_linalg_slogdet` 上 |
| 73 | `torch.logit` | `aten::logit` | `aten::logit_backward`、`aten::logical_and`、`aten::ge.Scalar`、`aten::le.Scalar`、`aten::where`、`aten::div.Tensor`、`aten::mul.Tensor`、`aten::sub.Scalar`、`aten::zeros`、`aten::empty`、`aten::fill_` | `GradMode` 开启与关闭时走不同 backward 路径 |
| 74 | `torch.logspace` | `aten::logspace`、`aten::empty`、`aten::logspace.out` | 无 | 工厂函数，不可微 |
| 75 | `torch.lu_solve` | `aten::lu_solve`、`aten::linalg_lu_solve`、`aten::linalg_lu_solve.out` | `aten::linalg_lu_solve`、`aten::lu_unpack`、`aten::matmul`、`aten::mH`、`aten::mT`、`aten::neg`、`aten::linalg_solve_triangular`、`aten::tril`、`aten::triu`、`aten::add.Tensor` | 旧接口包装，真正计算落到 `linalg_lu_solve` |

## 任务 16：接口 76-80

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 76 | `torch.lu_unpack` | `aten::lu_unpack`、`aten::triu.out`、`aten::tril.out`、`aten::diagonal`、`aten::fill_.Scalar`、`aten::arange`、`aten::expand`、`aten::contiguous`、`aten::zero_`、`aten::unsqueeze`、`aten::scatter_.value` | `aten::tril`、`aten::triu`、`aten::narrow`、`aten::add.Tensor`、`aten::cat`、`aten::zeros` | `P` 不可微，`LU_pivots` 也不参与梯度 |
| 77 | `torch.moveaxis` | `aten::moveaxis`、`aten::movedim`、`aten::permute` | `aten::permute` | `moveaxis` 是 `movedim` 的别名 |
| 78 | `torch.movedim` | `aten::movedim`、`aten::permute` | `aten::permute` | `CompositeImplicitAutograd` 直接落到 `permute` |
| 79 | `torch.msort` | `aten::msort`、`aten::sort`、`aten::sort.stable` | `aten::zeros`、`aten::scatter_`、`aten::scatter`、`aten::unsqueeze` | 实际是 `sort(self, 0, false)` 的封装 |
| 80 | `torch.multiply` | `aten::multiply.Tensor`、`aten::mul.Tensor`、`aten::multiply.Scalar`、`aten::mul.Scalar` | `aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj` | `multiply` 是 `mul` 的完全别名 |

## 任务 17：接口 81-85

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 81 | `torch.mvlgamma` | `aten::mvlgamma`、`aten::arange.start_step`、`aten::unsqueeze`、`aten::add.Tensor`、`aten::lgamma_`、`aten::sum.dim_IntList`、`aten::add_.Scalar` | `aten::arange`、`aten::unsqueeze`、`aten::add.Tensor`、`aten::digamma_`、`aten::sum`、`aten::mul.Tensor` | CEA op，backward 是手写 helper |
| 82 | `torch.nanmean` | `aten::nanmean`、`aten::detach`、`aten::isnan`、`aten::logical_not_`、`aten::sum.dim_IntList`、`aten::nansum`、`aten::div.Tensor` | `aten::unsqueeze`、`aten::expand`、`aten::isnan`、`aten::logical_not`、`aten::mul.Tensor`、`aten::div.Tensor`、`aten::neg` | `nanmean` 是 CIA 外壳，梯度实际来自内部 `nansum` 和 `div` |
| 83 | `torch.nanmedian` | `aten::nanmedian`、`aten::empty`、`aten::nanmedian.dim`、`aten::nanmedian.dim_values` | `aten::isnan`、`aten::logical_and_`、`aten::logical_or_`、`aten::eq.Tensor`、`aten::sum`、`aten::div.Scalar`、`aten::mul.Tensor`、`aten::new_zeros`、`aten::masked_fill_`、`aten::zeros`、`aten::unsqueeze`、`aten::scatter_` | 全归约和按维归约走不同 backward 公式 |
| 84 | `torch.nextafter` | `aten::nextafter` | 无 | `not_implemented`，反向会抛异常 |
| 85 | `torch.nn.AdaptiveMaxPool3d` | `aten::adaptive_max_pool3d` | `aten::adaptive_max_pool3d_backward` | backward 本身就是 structured ATen op |

## 任务 18：接口 86-90

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 86 | `torch.nn.AvgPool1d` | `aten::avg_pool1d`、`aten::unsqueeze`、`aten::avg_pool2d`、`aten::squeeze` | `aten::avg_pool2d_backward`、`aten::squeeze`、`aten::unsqueeze` | `avg_pool1d` 是 CIA 分解，反向挂在内部 `avg_pool2d` 上 |
| 87 | `torch.nn.CELU` | `aten::celu`、`aten::elu` | `aten::elu_backward` | `celu` 前向分解到 `elu`，但 backward 直接注册在 `celu` 上 |
| 88 | `torch.nn.CTCLoss` | `aten::ctc_loss.IntList`、`aten::ctc_loss.Tensor`、`aten::_ctc_loss`、`aten::_use_cudnn_ctc_loss`、`aten::_use_miopen_ctc_loss`、`aten::_cudnn_ctc_loss`、`aten::miopen_ctc_loss` | `aten::_ctc_loss_backward`、`aten::_cudnn_ctc_loss_backward`、`aten::_miopen_ctc_loss_backward` | `ctc_loss` 走 CIA 主链，不同加速后端有各自 backward |
| 89 | `torch.nn.ChannelShuffle` | `aten::channel_shuffle`、`aten::native_channel_shuffle`、`aten::view`、`aten::permute`、`aten::contiguous`、`aten::reshape` | `aten::channel_shuffle`、`aten::native_channel_shuffle`、`aten::view`、`aten::permute`、`aten::contiguous`、`aten::reshape` | `channel_shuffle` 是 self-inverse，反向再次调用自身 |
| 90 | `torch.nn.ConvTranspose1d` | `aten::conv_transpose1d`、`aten::convolution`、`aten::_convolution` | `aten::convolution_backward` | `conv_transpose1d` 的 backward 挂在内部 `convolution` 上 |

## 任务 19：接口 91-95

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 91 | `torch.nn.ConvTranspose3d` | `aten::conv_transpose3d.input`、`aten::convolution`、`aten::cudnn_convolution_transpose`、`aten::miopen_convolution_transpose`、`aten::slow_conv_transpose3d` | `aten::convolution_backward` | 与 1d 版本同机制，backward 仍挂在 `convolution` 上 |
| 92 | `torch.nn.Dropout1d` | `aten::feature_dropout`、`aten::new_empty`、`aten::bernoulli_`、`aten::div_`、`aten::mul.Tensor`、`aten::unsqueeze`、`aten::squeeze` | `aten::mul.Tensor`、`aten::conj`、`aten::unsqueeze`、`aten::squeeze` | 非 batched 输入会额外引入一对 `unsqueeze/squeeze` |
| 93 | `torch.nn.Dropout2d` | `aten::feature_dropout`、`aten::new_empty`、`aten::bernoulli_`、`aten::div_`、`aten::mul.Tensor` | `aten::mul.Tensor`、`aten::conj` | 2D 版本没有额外的 view 维度补齐 |
| 94 | `torch.nn.Dropout3d` | `aten::feature_dropout`、`aten::new_empty`、`aten::bernoulli_`、`aten::div_`、`aten::mul.Tensor`、`aten::unsqueeze`、`aten::squeeze` | `aten::mul.Tensor`、`aten::conj`、`aten::unsqueeze`、`aten::squeeze` | 非 batched 输入同样会补 batch 维再还原 |
| 95 | `torch.nn.GRU` | `aten::gru.input`、`aten::gru.data`、`aten::_cudnn_rnn`、`aten::miopen_rnn`、`aten::linear`、`aten::_thnn_fused_gru_cell`、`aten::sigmoid_`、`aten::tanh_`、`aten::mul_`、`aten::add_` | `aten::_cudnn_rnn_backward`、`aten::_thnn_fused_gru_cell_backward`、`aten::_thnn_differentiable_gru_cell_backward`、`aten::sigmoid_backward`、`aten::tanh_backward`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::cat`、`aten::sum` | GRU 有 cuDNN、fused 和 CPU fallback 三条反向路径 |

## 任务 20：接口 96-100

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 96 | `torch.nn.GRUCell` | `aten::gru_cell`、`aten::linear`、`aten::unsafe_chunk`、`aten::add_`、`aten::sigmoid_`、`aten::mul_`、`aten::tanh_`、`aten::sub`、`aten::t`、`aten::matmul`、`aten::_thnn_fused_gru_cell` | `aten::_thnn_fused_gru_cell_backward`、`aten::_thnn_differentiable_gru_cell_backward`、`aten::sigmoid_backward`、`aten::tanh_backward`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::cat`、`aten::sum` | CPU fallback 由基础算子独立反向，CUDA fused 路径走 GRU cell 专用 backward |
| 97 | `torch.nn.GaussianNLLLoss` | `aten::lt.Scalar`、`aten::any`、`aten::unsqueeze`、`aten::clone`、`aten::clamp_`、`aten::sub`、`aten::pow.Tensor_Scalar`、`aten::div.Tensor`、`aten::log`、`aten::add.Tensor`、`aten::mul.Scalar`、`aten::mean` | `aten::neg`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::div.Tensor`、`aten::conj`、`aten::pow.Tensor_Scalar`、`aten::expand`、`aten::div.Scalar` | 纯 Python composite，没有独立的 `aten::gaussian_nll_loss` schema |
| 98 | `torch.nn.HardTanh` | `aten::hardtanh`、`aten::empty_like`、`aten::hardtanh.out`、`aten::clamp.out` | `aten::hardtanh_backward` | backward 是独立 backend kernel |
| 99 | `torch.nn.HingeEmbeddingLoss` | `aten::hinge_embedding_loss`、`aten::zeros_like`、`aten::rsub.Scalar`、`aten::clamp_min_`、`aten::clamp_min`、`aten::where`、`aten::add`、`aten::mean` | `aten::neg`、`aten::ge.Scalar`、`aten::where`、`aten::expand`、`aten::div.Scalar` | Python composite，无专属 backward node |
| 100 | `torch.nn.HuberLoss` | `aten::broadcast_tensors`、`aten::huber_loss`、`aten::empty_like`、`aten::mean` | `aten::huber_loss_backward` | self 和 target 共享同一 backward op，只是参数顺序交换 |

## 任务 21：接口 101-105

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 101 | `torch.nn.InstanceNorm1d` | `aten::instance_norm`、`aten::contiguous`、`aten::view`、`aten::batch_norm`、`aten::_batch_norm_impl_index`、`aten::native_batch_norm`、`aten::cudnn_batch_norm`、`aten::miopen_batch_norm` | `aten::native_batch_norm_backward`、`aten::cudnn_batch_norm_backward`、`aten::miopen_batch_norm_backward`、`as_strided_backward` | `InstanceNorm1d/2d/3d` 主链相同，仅输入 rank 不同 |
| 102 | `torch.nn.InstanceNorm2d` | `aten::instance_norm`、`aten::contiguous`、`aten::view`、`aten::batch_norm`、`aten::_batch_norm_impl_index`、`aten::native_batch_norm`、`aten::cudnn_batch_norm`、`aten::miopen_batch_norm` | `aten::native_batch_norm_backward`、`aten::cudnn_batch_norm_backward`、`aten::miopen_batch_norm_backward`、`as_strided_backward` | 与 1d 版本 backward 机制一致 |
| 103 | `torch.nn.InstanceNorm3d` | `aten::instance_norm`、`aten::contiguous`、`aten::view`、`aten::batch_norm`、`aten::_batch_norm_impl_index`、`aten::native_batch_norm`、`aten::cudnn_batch_norm`、`aten::miopen_batch_norm` | `aten::native_batch_norm_backward`、`aten::cudnn_batch_norm_backward`、`aten::miopen_batch_norm_backward`、`as_strided_backward` | 与 1d 版本 backward 机制一致 |
| 104 | `torch.nn.LPPool1d` | `aten::pow.Tensor_Scalar`、`aten::avg_pool1d`、`aten::unsqueeze`、`aten::avg_pool2d`、`aten::squeeze`、`aten::sign`、`aten::abs`、`aten::relu`、`aten::mul.Tensor`、`aten::mul.Scalar` | `aten::pow.Tensor_Scalar`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj`、`aten::zeros_like`、`aten::avg_pool2d_backward`、`aten::threshold_backward`、`aten::sgn`、`as_strided_backward` | `LPPool1d` 是 Python composite，`avg_pool1d` 还会多一层 view backward |
| 105 | `torch.nn.LPPool2d` | `aten::pow.Tensor_Scalar`、`aten::avg_pool2d`、`aten::sign`、`aten::abs`、`aten::relu`、`aten::mul.Tensor`、`aten::mul.Scalar` | `aten::pow.Tensor_Scalar`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::conj`、`aten::zeros_like`、`aten::avg_pool2d_backward`、`aten::threshold_backward`、`aten::sgn` | 与 1d 版本相比，没有 `unsqueeze/squeeze` 这层 view backward |

## 任务 22：接口 106-110

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 106 | `torch.nn.LSTM` | `aten::lstm.input`、`aten::_cudnn_rnn` | `aten::_cudnn_rnn_backward` | 外层 `lstm.input` 无专属 backward，当前区间锁定 cuDNN fast path |
| 107 | `torch.nn.LSTMCell` | `aten::lstm_cell`、`aten::t`、`aten::matmul`、`aten::_thnn_fused_lstm_cell` | `aten::_thnn_differentiable_lstm_cell_backward`、`aten::_thnn_fused_lstm_cell_backward`、`aten::matmul`、`aten::t` | 上游 gate GEMM 也各自带梯度 |
| 108 | `torch.nn.LeakyReLU` | `aten::leaky_relu` | `aten::leaky_relu_backward` | 非 inplace 模块路径 |
| 109 | `torch.nn.MarginRankingLoss` | `aten::margin_ranking_loss`、`aten::sub`、`aten::neg`、`aten::mul`、`aten::add.Scalar`、`aten::clamp_min_`、`aten::mean` | `aten::sub.Tensor`、`aten::neg`、`aten::mul.Tensor`、`aten::clamp_min`、`aten::mean`、`aten::sum` | CIA 分解，反向由基础 op 共同承担 |
| 110 | `torch.nn.MaxPool1d` | `aten::max_pool1d`、`aten::max_pool1d_with_indices`、`aten::unsqueeze`、`aten::max_pool2d_with_indices`、`aten::squeeze` | `aten::max_pool2d_with_indices_backward`、`as_strided_backward` | 1d 通过 2d 实现，view backward 也要算上 |

## 任务 23：接口 111-115

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 111 | `torch.nn.MaxPool2d` | `aten::max_pool2d`、`aten::max_pool2d_with_indices` | `aten::max_pool2d_backward` | 外层 `max_pool2d` 挂 `MaxPool2DBackward0`，前向实际委托到 `with_indices` |
| 112 | `torch.nn.MaxPool3d` | `aten::max_pool3d`、`aten::max_pool3d_with_indices` | `aten::max_pool3d_with_indices_backward` | 外层无统一 backward node，实际靠 `with_indices` |
| 113 | `torch.nn.MaxUnpool1d` | `aten::unsqueeze`、`aten::max_unpool2d`、`aten::squeeze` | `aten::contiguous`、`aten::view`、`aten::gather`、`aten::empty_like`、`as_strided_backward` | 通过 `max_unpool2d` 实现，view backward 也存在 |
| 114 | `torch.nn.MaxUnpool3d` | `aten::max_unpool3d` | `aten::contiguous`、`aten::view`、`aten::gather`、`aten::empty_like` | `indices` 明确是 non_differentiable |
| 115 | `torch.nn.MultiLabelMarginLoss` | `aten::multilabel_margin_loss`、`aten::multilabel_margin_loss_forward` | `aten::multilabel_margin_loss_backward` | 真正建立 grad_fn 的是 `multilabel_margin_loss_forward` |

## 任务 24：接口 116-120

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 116 | `torch.nn.MultiLabelSoftMarginLoss` | `aten::log_sigmoid`、`aten::log_sigmoid_forward`、`aten::neg`、`aten::rsub.Scalar`、`aten::mul`、`aten::add`、`aten::sum`、`aten::div.Scalar`、`aten::mean` | `aten::log_sigmoid_backward`、`aten::neg`、`aten::rsub.Scalar`、`aten::mul.Tensor`、`aten::add.Tensor`、`aten::sum.dim_IntList`、`aten::div.Scalar`、`aten::mean` | Python 组合前向，`weight` 分支会再多一层 `mul.Tensor` |
| 117 | `torch.nn.MultiMarginLoss` | `aten::multi_margin_loss` | `aten::multi_margin_loss_backward` | backward ATen op，`target` 非可导 |
| 118 | `torch.nn.MultiheadAttention` | `aten::_native_multi_head_attention`、`aten::linear`、`aten::scaled_dot_product_attention`、`aten::_scaled_dot_product_flash_attention`、`aten::_scaled_dot_product_efficient_attention`、`aten::_scaled_dot_product_cudnn_attention`、`aten::_scaled_dot_product_attention_math`、`aten::matmul`、`aten::_safe_softmax`、`aten::_softmax`、`aten::bmm`、`aten::baddbmm`、`aten::dropout`、`aten::softmax`、`aten::mean` | `aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::matmul`、`aten::bmm`、`aten::baddbmm`、`aten::addmm`、`aten::mm`、`aten::t` | 无统一 module backward node，按具体 attention backend 分流 |
| 119 | `torch.nn.PixelUnshuffle` | `aten::pixel_unshuffle` | `aten::pixel_shuffle` | 逆变换即反向 |
| 120 | `torch.nn.PoissonNLLLoss` | `aten::poisson_nll_loss`、`aten::exp`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::add.Scalar`、`aten::log`、`aten::le`、`aten::masked_fill`、`aten::add_`、`aten::mean`、`aten::sum` | `aten::exp`、`aten::mul.Tensor`、`aten::sub.Tensor`、`aten::add.Scalar`、`aten::log`、`aten::le`、`aten::masked_fill`、`aten::add_`、`aten::mean`、`aten::sum` | 不同参数分支走不同 composite 展开 |

## 任务 25：接口 121-125

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 121 | `torch.nn.RNN` | `aten::rnn_tanh.input`、`aten::rnn_relu.input`、`aten::rnn_tanh.data`、`aten::rnn_relu.data`、`aten::_cudnn_rnn`、`aten::linear`、`aten::add_`、`aten::tanh`、`aten::relu`、`aten::dropout`、`aten::cat` | `aten::_cudnn_rnn_backward`、`aten::addmm`、`aten::mm`、`aten::t`、`aten::tanh_backward`、`aten::threshold_backward`、`aten::native_dropout_backward`、`aten::cat` | cuDNN 和 native 两条常见路径都要覆盖 |
| 122 | `torch.nn.RNNCell` | `aten::rnn_tanh_cell`、`aten::rnn_relu_cell`、`aten::linear`、`aten::add_`、`aten::tanh`、`aten::relu` | `aten::addmm`、`aten::mm`、`aten::t`、`aten::add.Tensor`、`aten::tanh_backward`、`aten::threshold_backward` | 无统一 backward node |
| 123 | `torch.nn.RReLU` | `aten::rrelu`、`aten::rrelu_with_noise`、`aten::leaky_relu.out` | `aten::rrelu_with_noise_backward`、`aten::leaky_relu_backward` | 训练态走 `rrelu_with_noise`，`eval()` 退化为 `leaky_relu` |
| 124 | `torch.nn.SoftMarginLoss` | `aten::soft_margin_loss`、`aten::neg.out`、`aten::mul_`、`aten::exp_`、`aten::log1p_`、`aten::mean` | `aten::soft_margin_loss_backward`、`aten::exp`、`aten::mul`、`aten::add`、`aten::div` | backward ATen op + helper 展开 |
| 125 | `torch.nn.Softmax2d` | `aten::softmax.int`、`aten::_softmax` | `aten::_softmax_backward_data` | 只是把 `dim=-3` 固定住的 wrapper |

## 任务 26：接口 126-130

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 126 | `torch.nn.Softmin` | `aten::neg`、`aten::softmax.int`、`aten::_softmax` | `aten::_softmax_backward_data`、`aten::neg` | `softmin` 是 Python 组合包装，不是独立 native schema |
| 127 | `torch.nn.Softsign` | `aten::abs`、`aten::add.Scalar`、`aten::div.Tensor` | `aten::div.Tensor`、`aten::abs`、`aten::sgn`、`aten::add.Scalar`、`aten::mul.Tensor`、`aten::neg` | 反向由 `abs` 和 `div` 等子 op 组合而来 |
| 128 | `torch.nn.Tanhshrink` | `aten::tanh`、`aten::sub.Tensor` | `aten::tanh_backward`、`aten::neg` | `forward = input - input.tanh()` |
| 129 | `torch.nn.Transformer` | `aten::linear`、`aten::addmm`、`aten::scaled_dot_product_attention`、`aten::_scaled_dot_product_flash_attention_for_cpu`、`aten::_scaled_dot_product_flash_attention`、`aten::_scaled_dot_product_efficient_attention`、`aten::_scaled_dot_product_cudnn_attention`、`aten::_scaled_dot_product_attention_math`、`aten::matmul`、`aten::_safe_softmax`、`aten::softmax.int`、`aten::_softmax`、`aten::dropout`、`aten::layer_norm`、`aten::native_layer_norm`、`aten::relu`、`aten::gelu`、`aten::_transformer_encoder_layer_fwd`、`aten::_native_multi_head_attention`、`aten::_transform_bias_rescale_qkv`、`aten::bmm`、`aten::_addmm_activation`、`aten::add_`、`aten::_masked_softmax`、`aten::_nested_tensor_from_mask`、`aten::to_padded_tensor` | `aten::native_layer_norm_backward`、`aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward`、`aten::gelu_backward`、`aten::add.Tensor`、`aten::matmul` | 训练态 backward 不会落到 fused fastpath |
| 130 | `torch.nn.TransformerDecoder` | `aten::linear`、`aten::addmm`、`aten::scaled_dot_product_attention`、`aten::_scaled_dot_product_flash_attention_for_cpu`、`aten::_scaled_dot_product_flash_attention`、`aten::_scaled_dot_product_efficient_attention`、`aten::_scaled_dot_product_cudnn_attention`、`aten::_scaled_dot_product_attention_math`、`aten::matmul`、`aten::_safe_softmax`、`aten::softmax.int`、`aten::_softmax`、`aten::dropout`、`aten::relu`、`aten::gelu`、`aten::layer_norm`、`aten::native_layer_norm`、`aten::_native_multi_head_attention`、`aten::_transform_bias_rescale_qkv`、`aten::bmm`、`aten::_masked_softmax` | `aten::native_layer_norm_backward`、`aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward`、`aten::gelu_backward`、`aten::add.Tensor`、`aten::matmul` | self-attn fastpath 不支持 autograd，训练时会回到常规子图 |

## 任务 27：接口 131-135

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 131 | `torch.nn.TransformerDecoderLayer` | `aten::linear`、`aten::scaled_dot_product_attention`、`aten::_scaled_dot_product_flash_attention_for_cpu`、`aten::_scaled_dot_product_flash_attention`、`aten::_scaled_dot_product_efficient_attention`、`aten::_scaled_dot_product_cudnn_attention`、`aten::_scaled_dot_product_attention_math`、`aten::dropout`、`aten::layer_norm`、`aten::native_layer_norm`、`aten::relu`、`aten::gelu` | `aten::native_layer_norm_backward`、`aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward`、`aten::gelu_backward`、`aten::add.Tensor`、`aten::matmul` | 整层没有 fused backward，依赖内部 self-attn / cross-attn / FFN 子图 |
| 132 | `torch.nn.TransformerEncoder` | `aten::_transformer_encoder_layer_fwd`、`aten::scaled_dot_product_attention`、`aten::layer_norm`、`aten::native_layer_norm`、`aten::_nested_tensor_from_mask`、`aten::to_padded_tensor` | `aten::native_layer_norm_backward`、`aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward`、`aten::gelu_backward`、`aten::add.Tensor`、`aten::matmul` | 有梯度时不走 `_transformer_encoder_layer_fwd` fastpath |
| 133 | `torch.nn.TransformerEncoderLayer` | `aten::_transformer_encoder_layer_fwd`、`aten::linear`、`aten::scaled_dot_product_attention`、`aten::_scaled_dot_product_flash_attention`、`aten::_scaled_dot_product_efficient_attention`、`aten::_scaled_dot_product_cudnn_attention`、`aten::_scaled_dot_product_attention_math`、`aten::dropout`、`aten::add`、`aten::layer_norm`、`aten::native_layer_norm`、`aten::relu`、`aten::gelu`、`aten::_native_multi_head_attention`、`aten::add_`、`aten::_addmm_activation`、`aten::addmm` | `aten::native_layer_norm_backward`、`aten::_scaled_dot_product_flash_attention_backward`、`aten::_scaled_dot_product_efficient_attention_backward`、`aten::_scaled_dot_product_cudnn_attention_backward`、`aten::_softmax_backward_data`、`aten::native_dropout_backward`、`aten::addmm`、`aten::threshold_backward`、`aten::gelu_backward`、`aten::add.Tensor`、`aten::matmul` | 推理 fastpath 无 autograd，训练态才有 backward |
| 134 | `torch.nn.TripletMarginLoss` | `aten::triplet_margin_loss`、`aten::pairwise_distance`、`aten::minimum`、`aten::clamp_min`、`aten::mean`、`aten::sum`、`aten::norm` | `aten::pairwise_distance`、`aten::norm`、`aten::minimum`、`aten::clamp_min`、`aten::mean`、`aten::sum` | `swap=True` 时才多一条 `minimum` 支路 |
| 135 | `torch.nn.functional.affine_grid` | `aten::affine_grid_generator`、`aten::empty`、`aten::linspace`、`aten::copy_`、`aten::fill_`、`aten::transpose`、`aten::bmm` | `aten::affine_grid_generator_backward`、`aten::empty`、`aten::linspace`、`aten::copy_`、`aten::fill_`、`aten::transpose`、`aten::bmm` | 反向 schema 规范化后对应 `aten::affine_grid_generator_backward` |

## 任务 28：接口 136-140

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 136 | `torch.nn.functional.celu` | `aten::celu`、`aten::elu`、`aten::celu_`、`aten::elu_` | `aten::elu_backward` | 前向是 `CompositeExplicitAutograd`，反向直接复用 ELU backward |
| 137 | `torch.nn.functional.cosine_similarity` | `aten::cosine_similarity`、`aten::to`、`aten::expand`、`aten::linalg_vector_norm`、`aten::clone`、`aten::scalar_tensor`、`aten::clamp_min_`、`aten::div`、`aten::mul`、`aten::sum` | `aten::linalg_vector_norm_backward`、`aten::div.Tensor`、`aten::mul.Tensor`、`aten::sum.dim_IntList`、`aten::expand` | 梯度来自内部展开的 norm / div / mul / sum 子图 |
| 138 | `torch.nn.functional.ctc_loss` | `aten::ctc_loss.Tensor`、`aten::_ctc_loss`、`aten::_use_cudnn_ctc_loss`、`aten::_use_miopen_ctc_loss`、`aten::_cudnn_ctc_loss`、`aten::miopen_ctc_loss` | `aten::_ctc_loss_backward`、`aten::_cudnn_ctc_loss_backward`、`aten::_miopen_ctc_loss_backward` | `ctc_loss.Tensor` 本身是 wrapper，真正承接 backward 的是底层 loss op |
| 139 | `torch.nn.functional.dropout1d` | `aten::unsqueeze`、`aten::feature_dropout`、`aten::feature_dropout_`、`aten::new_empty`、`aten::bernoulli_`、`aten::div_`、`aten::mul`、`aten::mul_`、`aten::squeeze` | `aten::mul.Tensor`、`as_strided_backward` | 2D 非 batched 路径才会多出 view backward |
| 140 | `torch.nn.functional.dropout2d` | `aten::feature_dropout`、`aten::feature_dropout_`、`aten::new_empty`、`aten::bernoulli_`、`aten::div_`、`aten::mul`、`aten::mul_` | `aten::mul.Tensor` | 历史 3D 兼容路径仍复用同一 `feature_dropout` 机制 |

## 任务 29：接口 141-145

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 141 | `torch.nn.functional.dropout3d` | `aten::unsqueeze`、`aten::feature_dropout`、`aten::new_empty`、`aten::bernoulli_`、`aten::div_`、`aten::mul`、`aten::squeeze.dim` | `aten::mul.Tensor`、`as_strided_backward` | 4D 无 batch 输入会多出 `unsqueeze/squeeze` |
| 142 | `torch.nn.functional.gaussian_nll_loss` | `aten::unsqueeze`、`aten::clone`、`aten::clamp_`、`aten::log`、`aten::sub.Tensor`、`aten::pow.Tensor_Scalar`、`aten::div.Tensor`、`aten::add.Tensor`、`aten::mul.Scalar`、`aten::mean` | `aten::log`、`aten::sub.Tensor`、`aten::pow.Tensor_Scalar`、`aten::div.Tensor`、`aten::add.Tensor`、`aten::mul.Scalar`、`aten::mean`、`aten::sum`、`as_strided_backward` | `clamp_` 在 `no_grad` 中执行，不进梯度图 |
| 143 | `torch.nn.functional.gumbel_softmax` | `aten::empty_like`、`aten::exponential_`、`aten::log`、`aten::neg`、`aten::add.Tensor`、`aten::div.Scalar`、`aten::softmax.int`、`aten::_softmax`、`aten::max.dim`、`aten::zeros_like`、`aten::scatter_.value`、`aten::detach`、`aten::sub.Tensor` | `aten::_softmax_backward_data`、`aten::div.Scalar`、`aten::add.Tensor`、`aten::neg`、`aten::sub.Tensor` | `hard=True` 的 straight-through 反向仍回到 softmax 分支 |
| 144 | `torch.nn.functional.hardtanh` | `aten::hardtanh`、`aten::clamp.out` | `aten::hardtanh_backward` | 前向真正算在 `clamp.out`，但 backward 是专用 schema |
| 145 | `torch.nn.functional.hinge_embedding_loss` | `aten::hinge_embedding_loss`、`aten::zeros_like`、`aten::sub.Scalar`、`aten::clamp_min_`、`aten::clamp_min`、`aten::where`、`aten::add.Tensor`、`aten::mean`、`aten::sum` | `aten::where`、`aten::clamp_min`、`aten::add.Tensor`、`aten::mean`、`aten::sum` | `target` 非可导，`zeros_like` 只构造常量分支 |

## 任务 30：接口 146-150

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 146 | `torch.nn.functional.huber_loss` | `aten::huber_loss`、`aten::mean`、`aten::sum`、`aten::mul` | `aten::huber_loss_backward`、`aten::mul.Tensor`、`aten::mean`、`aten::sum` | `weight=None` 时主链更短，`weight!=None` 会在 Python 侧再叠一层 `mul + reduction` |
| 147 | `torch.nn.functional.lp_pool1d` | `aten::pow.Tensor_Scalar`、`aten::avg_pool1d`、`aten::unsqueeze`、`aten::avg_pool2d`、`aten::squeeze.dim`、`aten::sign`、`aten::abs`、`aten::relu`、`aten::mul.Scalar` | `aten::pow.Tensor_Scalar`、`aten::avg_pool2d_backward`、`aten::sgn`、`aten::threshold_backward`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::zeros_like`、`as_strided_backward` | `lp_pool1d` 是 Python composite，`avg_pool1d` 还会多一层 view backward |
| 148 | `torch.nn.functional.lp_pool2d` | `aten::pow.Tensor_Scalar`、`aten::avg_pool2d`、`aten::sign`、`aten::abs`、`aten::relu`、`aten::mul.Scalar` | `aten::pow.Tensor_Scalar`、`aten::avg_pool2d_backward`、`aten::sgn`、`aten::threshold_backward`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::zeros_like` | 与 1d 版本相比，没有 `unsqueeze/squeeze` 这层 view backward |
| 149 | `torch.nn.functional.margin_ranking_loss` | `aten::margin_ranking_loss`、`aten::sub`、`aten::mul`、`aten::neg`、`aten::add.Scalar`、`aten::clamp_min`、`aten::mean`、`aten::sum` | `aten::sub.Tensor`、`aten::mul.Tensor`、`aten::neg`、`aten::add.Scalar`、`aten::clamp_min`、`aten::mean`、`aten::sum` | `target` 非可导，反向由 composite body 的基础算子承担 |
| 150 | `torch.nn.functional.max_pool3d` | `aten::max_pool3d`、`aten::max_pool3d_with_indices` | `aten::max_pool3d_with_indices_backward` | `return_indices=False` 的公开 API 仍会归到 `max_pool3d_with_indices` 的 backward 上 |

## 任务 31：接口 151-155

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 151 | `torch.nn.functional.max_unpool1d` | `aten::unsqueeze`、`aten::max_unpool2d`、`aten::squeeze.dim` | `aten::gather`、`as_strided_backward` | 1D 反池化本质是 `unsqueeze -> max_unpool2d -> squeeze` |
| 152 | `torch.nn.functional.max_unpool3d` | `aten::max_unpool3d` | `aten::gather` | `max_pool_double_backward` 展开后核心是 `gather` |
| 153 | `torch.nn.functional.multi_margin_loss` | `aten::multi_margin_loss` | `aten::multi_margin_loss_backward` | 专用 backward ATen op，`target/weight` 不可导 |
| 154 | `torch.nn.functional.multilabel_margin_loss` | `aten::multilabel_margin_loss`、`aten::multilabel_margin_loss_forward` | `aten::multilabel_margin_loss_backward` | 公开 `multilabel_margin_loss` 只取 `forward` tuple 的第一个输出 |
| 155 | `torch.nn.functional.multilabel_soft_margin_loss` | `aten::log_sigmoid`、`aten::log_sigmoid_forward`、`aten::neg`、`aten::rsub.Scalar`、`aten::mul`、`aten::add`、`aten::sum`、`aten::div.Scalar`、`aten::mean` | `aten::log_sigmoid_backward`、`aten::neg`、`aten::rsub.Scalar`、`aten::mul.Tensor`、`aten::add.Tensor`、`aten::sum.dim_IntList`、`aten::div.Scalar`、`aten::mean`、`aten::sum` | Python 组合前向，没有 fused loss schema |

## 任务 32：接口 156-160

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 156 | `torch.nn.functional.pixel_unshuffle` | `aten::pixel_unshuffle`、`aten::reshape`、`aten::permute`、`aten::clone`、`aten::view` | `aten::pixel_shuffle` | CPU 走原生 kernel；无专用 kernel 的后端会回退到 `reshape -> permute -> clone -> view` |
| 157 | `torch.nn.functional.rms_norm` | `aten::rms_norm`、`aten::_fused_rms_norm`、`aten::pow.Tensor_Scalar`、`aten::mean.dim`、`aten::add_`、`aten::rsqrt`、`aten::mul.Tensor` | `aten::_fused_rms_norm_backward`、`aten::pow.Tensor_Scalar`、`aten::mean.dim`、`aten::add_`、`aten::rsqrt`、`aten::mul.Tensor` | `rms_norm` 入口是 CIA 壳，梯度常挂在 `_fused_rms_norm` 或 fallback composite 链上 |
| 158 | `torch.nn.functional.rrelu` | `aten::rrelu`、`aten::rrelu_with_noise`、`aten::leaky_relu.out` | `aten::rrelu_with_noise_backward`、`aten::mul.Tensor`、`aten::leaky_relu_backward` | 训练态 helper 用 `noise * grad`，推理态退化为 `leaky_relu_backward` |
| 159 | `torch.nn.functional.soft_margin_loss` | `aten::soft_margin_loss`、`aten::neg.out`、`aten::mul_`、`aten::exp_`、`aten::log1p_`、`aten::mean` | `aten::soft_margin_loss_backward` | 前向是 `CompositeExplicitAutograd`，但 backward 直接走专用 schema |
| 160 | `torch.nn.functional.softmin` | `aten::neg`、`aten::softmax.int`、`aten::_softmax` | `aten::neg`、`aten::_softmax_backward_data` | `softmin(x)` 等价于 `softmax(-x)` |

## 任务 33：接口 161-165

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 161 | `torch.nn.functional.softsign` | `aten::abs`、`aten::add.Scalar`、`aten::div.Tensor` | `aten::div.Tensor`、`aten::sgn`、`aten::mul.Tensor`、`aten::add.Scalar` | 纯 Python 组合表达式，没有单独 `aten::softsign` |
| 162 | `torch.nn.functional.triplet_margin_loss` | `aten::triplet_margin_loss`、`aten::pairwise_distance`、`aten::sub.Tensor`、`aten::add.Scalar`、`aten::norm.ScalarOpt_dim`、`aten::min.other`、`aten::clamp_min`、`aten::mean` | `aten::sub.Tensor`、`aten::add.Scalar`、`aten::norm.ScalarOpt_dim`、`aten::min.other`、`aten::clamp_min`、`aten::mean`、`aten::sum` | `swap=True` 时会多一条 `positive-negative` 分支 |
| 163 | `torch.nn.functional.upsample` | `aten::upsample_nearest2d.vec`、`aten::upsample_nearest2d`、`aten::upsample_bilinear2d.vec`、`aten::upsample_bilinear2d` | `aten::upsample_nearest2d_backward`、`aten::upsample_bilinear2d_backward` | `upsample` 只是弃用别名，实际 backward 挂在 2D op 上 |
| 164 | `torch.nn.modules.ChannelShuffle` | `aten::channel_shuffle`、`aten::native_channel_shuffle`、`aten::view`、`aten::permute`、`aten::contiguous`、`aten::reshape` | `aten::channel_shuffle`、`aten::view`、`aten::permute`、`aten::contiguous`、`aten::reshape` | backward 公式再次调用 `channel_shuffle` 本身 |
| 165 | `torch.nn.modules.flatten.Flatten` | `aten::flatten.using_ints`、`aten::view` | `as_strided_backward` | 连续输入常规路径最终落到 `view` backward |

## 任务 34：接口 166-170

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 166 | `torch.nn.modules.flatten.Unflatten` | `aten::unflatten.int`、`aten::unflatten.Dimname`、`aten::view` | `aten::view` | NamedTensor 分支只回写 names，不新增 ATen 依赖 |
| 167 | `torch.numel` | 无 | 无 | eager 路径直接读元信息；JIT prim `aten::numel` 也不是 dispatcher schema |
| 168 | `torch.optim.ASGD` | `aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::sub`、`aten::copy_`、`aten::_foreach_add_.Scalar`、`aten::_foreach_neg`、`aten::_foreach_add.List`、`aten::_foreach_add_.List`、`aten::_foreach_addcmul_.Scalar`、`aten::_foreach_sub.List`、`aten::_foreach_copy_`、`aten::_foreach_maximum_.Scalar`、`aten::_foreach_pow_.Scalar`、`aten::_foreach_reciprocal_` | `aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::sub`、`aten::copy_` | Python 组合更新；默认 `no_grad`，`differentiable=True` 时梯度沿内部原语传播，`foreach=True` 改走 `_foreach_*` 原语 |
| 169 | `torch.optim.Adamax` | `aten::add_`、`aten::add`、`aten::lerp_`、`aten::mul_`、`aten::abs`、`aten::maximum.out`、`aten::addcdiv_`、`aten::_foreach_add_.Scalar`、`aten::_foreach_add.List`、`aten::_foreach_add_.List`、`aten::_foreach_lerp_.Scalar`、`aten::_foreach_mul_.Scalar`、`aten::_foreach_abs`、`aten::_foreach_abs_`、`aten::_foreach_maximum_.List`、`aten::_foreach_addcdiv_.ScalarList`、`aten::_foreach_pow`、`aten::_foreach_div_`、`aten::_foreach_mul`、`aten::cat`、`aten::amax` | `aten::add_`、`aten::add`、`aten::lerp_`、`aten::mul_`、`aten::abs`、`aten::maximum.out`、`aten::addcdiv_` | Python 组合更新；默认 `no_grad`，`foreach` / `capturable` 只改变内部原语排列；`differentiable=True` 的单张量路径会把 `maximum.out` 改成 `cat + amax` |
| 170 | `torch.optim.RMSprop` | `aten::add_`、`aten::add`、`aten::mul_`、`aten::addcmul_`、`aten::sqrt`、`aten::addcdiv_`、`aten::lerp_`、`aten::addcmul`、`aten::sqrt_`、`aten::_foreach_add_.Scalar`、`aten::_foreach_add.List`、`aten::_foreach_mul_.Scalar`、`aten::_foreach_addcmul_.Scalar`、`aten::_foreach_lerp_.Scalar`、`aten::_foreach_addcmul.Scalar`、`aten::_foreach_sqrt_`、`aten::_foreach_sqrt`、`aten::_foreach_addcdiv_.Scalar`、`aten::_foreach_add_` | `aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::addcmul_`、`aten::lerp_`、`aten::sqrt`、`aten::sqrt_`、`aten::addcdiv_` | Python 组合更新；默认 `no_grad`，`centered` / `momentum` / `foreach` 只改变内部原语组合，不产生统一 backward schema |

## 任务 35：接口 171-175

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 171 | `torch.optim.Rprop` | `aten::neg`、`aten::mul`、`aten::sign`、`aten::mul_`、`aten::clamp_`、`aten::addcmul_`、`aten::copy_`、`aten::_foreach_add_.Scalar`、`aten::_foreach_mul`、`aten::_foreach_neg_`、`aten::_foreach_copy_`、`aten::_foreach_sign_`、`aten::_foreach_mul_`、`aten::_foreach_addcmul_` | `aten::neg`、`aten::mul`、`aten::sign`、`aten::mul_`、`aten::clamp_`、`aten::addcmul_`、`aten::copy_` | Python 组合更新；默认 `no_grad`，`differentiable=True` 时由内部原语各自求导，`foreach=True` 改走 `_foreach_*` 原语 |
| 172 | `torch.optim.adadelta.Adadelta` | `aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::addcmul_`、`aten::sqrt_`、`aten::div_`、`aten::_foreach_add_.Scalar`、`aten::_foreach_neg`、`aten::_foreach_add`、`aten::_foreach_add_`、`aten::_foreach_mul_`、`aten::_foreach_addcmul_`、`aten::_foreach_sqrt_`、`aten::_foreach_div_` | `aten::add_`、`aten::neg`、`aten::add`、`aten::mul_`、`aten::addcmul_`、`aten::sqrt_`、`aten::div_` | Python 组合更新；默认 `no_grad`，`differentiable=True` 时梯度沿内部原语传播，`foreach=True` 只切换到批量原语 |
| 173 | `torch.optim.adagrad.Adagrad` | `aten::add_`、`aten::neg`、`aten::add`、`aten::addcmul_`、`aten::sqrt`、`aten::sqrt_`、`aten::addcdiv_`、`aten::pow`、`aten::sparse_mask`、`aten::_foreach_add_.Scalar`、`aten::_foreach_neg`、`aten::_foreach_add`、`aten::_foreach_add_`、`aten::_foreach_addcmul_`、`aten::_foreach_sqrt`、`aten::_foreach_mul`、`aten::_foreach_mul_`、`aten::_foreach_addcdiv_`、`aten::_fused_adagrad_` | `aten::add_`、`aten::neg`、`aten::add`、`aten::addcmul_`、`aten::sqrt`、`aten::sqrt_`、`aten::addcdiv_`、`aten::pow`、`aten::sparse_mask`、`aten::_fused_adagrad_` | Python 组合更新；默认 `no_grad`，`dense` / `sparse` / `foreach` / `fused` 只改变内部原语组合 |
| 174 | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` | 无 | 无 | 纯 Python 调度逻辑，不对应固定 ATen dispatch / backward schema |
| 175 | `torch.orgqr` | `aten::orgqr`、`aten::linalg_householder_product`、`aten::empty`、`aten::linalg_householder_product.out`、`aten::copy_` | `aten::linalg_householder_product`、`aten::tril`、`aten::diagonal`、`aten::fill_`、`aten::sum`、`aten::matmul`、`aten::mH`、`aten::narrow`、`aten::zeros_like`、`aten::cat`、`aten::copy_` | `orgqr` 只是 `linalg_householder_product` 别名，反向挂在后者上 |

## 任务 36：接口 176-180

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 176 | `torch.ormqr` | `aten::ormqr`、`aten::empty`、`aten::resize_as_`、`aten::transpose_`、`aten::copy_` | `aten::ormqr`、`aten::tril`、`aten::diagonal`、`aten::fill_`、`aten::sum`、`aten::matmul`、`aten::mH`、`aten::narrow`、`aten::zeros_like`、`aten::cat`、`aten::copy_` | 反向 helper 一部分会再次调用 `aten::ormqr` |
| 177 | `torch.pdist` | `aten::pdist`、`aten::contiguous`、`aten::_pdist_forward`、`aten::empty` | `aten::_pdist_backward`、`aten::empty_like`、`aten::empty`、`aten::sum.IntList_out` | 真正 backward 挂在 `_pdist_forward` 上 |
| 178 | `torch.poisson` | `aten::poisson`、`aten::zeros`、`aten::empty` | `aten::zeros_like` | 随机采样对输入率参数的梯度直接置零 |
| 179 | `torch.polygamma` | `aten::polygamma` | `aten::mul.Tensor`、`aten::polygamma` | backward 递归调用更高阶 `polygamma` |
| 180 | `torch.positive` | `aten::positive` | 无 | 纯 identity / alias，`bool` 输入会报错 |

## 任务 37：接口 181-185

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 181 | `torch.rad2deg` | `aten::rad2deg`、`aten::empty_like`、`aten::rad2deg.out`、`aten::mul.out` | `aten::mul.Scalar` | helper 只做常数缩放 |
| 182 | `torch.range` | `aten::range.step`、`aten::empty`、`aten::range.out` | 无 | 手写 Python wrapper；forward 会命中 `range.step / range.out`，但输入是标量，因此反向不可微 |
| 183 | `torch.renorm` | `aten::renorm`、`aten::renorm.out`、`aten::linalg_vector_norm`、`aten::empty`、`aten::mul.out` | `aten::linalg_vector_norm`、`aten::conj`、`aten::mul.Tensor`、`aten::real`、`aten::sum`、`aten::sgn`、`aten::div.Tensor`、`aten::masked_fill_`、`aten::eq.Scalar`、`aten::eq.Tensor`、`aten::abs`、`aten::isnan`、`aten::logical_or`、`aten::reciprocal`、`aten::add.Scalar`、`aten::mul.Scalar`、`aten::sub.Tensor`、`aten::where`、`aten::gt.Scalar` | backward 已按 `renorm_backward -> norm_backward` 递归展开 |
| 184 | `torch.rot90` | `aten::rot90`、`aten::flip`、`aten::transpose_`、`aten::clone` | `aten::rot90` | backward 就是 `grad.rot90(-k, dims)` |
| 185 | `torch.row_stack` | `aten::row_stack`、`aten::vstack`、`aten::atleast_2d.Sequence`、`aten::unsqueeze`、`aten::reshape`、`aten::cat` | `aten::narrow`、`aten::squeeze.dim`、`aten::reshape` | `row_stack` 是 `vstack` 纯别名，反向依赖与 #211 一致 |

## 任务 38：接口 186-190

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 186 | `torch.select_scatter` | `aten::select_scatter`、`aten::select.int`、`aten::copy_` | `aten::select_scatter`、`aten::zeros_like`、`aten::select.int` | backward 直接复用同一个 scatter schema 回填 `self` 梯度 |
| 187 | `torch.sgn` | `aten::sgn` | `aten::abs`、`aten::mul.Tensor`、`aten::conj`、`aten::div.Tensor`、`aten::masked_fill_.Scalar`、`aten::eq.Scalar` | 复数分支才有非零梯度，实数输入返回零梯度张量 |
| 188 | `torch.signbit` | `aten::signbit`、`aten::fill_.Scalar` | 无 | 布尔输出，不可微 |
| 189 | `torch.slogdet` | `aten::slogdet`、`aten::linalg_slogdet`、`aten::_linalg_slogdet`、`aten::linalg_lu_factor_ex.out`、`aten::sgn`、`aten::prod`、`aten::mul.out`、`aten::abs`、`aten::log_`、`aten::sum.IntList_out` | `aten::imag`、`aten::conj`、`aten::diag_embed`、`aten::unsqueeze`、`aten::expand_as`、`aten::mT`、`aten::linalg_lu_solve`、`aten::linalg_solve`、`aten::mH` | 真正可导入口是 `_linalg_slogdet`，不是外层 alias |
| 190 | `torch.special.bessel_j0` | `aten::special_bessel_j0` | 无 | 不可微，wrapper 只 redispatch，不设 history |

## 任务 39：接口 191-195

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 191 | `torch.special.bessel_j1` | `aten::special_bessel_j1` | 无 | 不可微 |
| 192 | `torch.special.bessel_y0` | `aten::special_bessel_y0` | 无 | 不可微 |
| 193 | `torch.special.bessel_y1` | `aten::special_bessel_y1` | 无 | 不可微 |
| 194 | `torch.special.i0` | `aten::special_i0`、`aten::i0` | `aten::mul.Tensor`、`aten::special_i1` | `special_i0` 是 `i0` 的别名，真实反向挂在内层 `aten::i0` |
| 195 | `torch.special.i1` | `aten::special_i1` | `aten::abs`、`aten::where`、`aten::scalar_tensor`、`aten::i0`、`aten::reciprocal`、`aten::sub`、`aten::mul.Tensor` | `x == 0` 时 helper 会把梯度修正成 `0.5` |

## 任务 40：接口 196-200

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 196 | `torch.special.i1e` | `aten::special_i1e` | `aten::special_i0e`、`aten::sgn`、`aten::reciprocal`、`aten::where`、`aten::abs`、`aten::mul.Tensor`、`aten::sub.Tensor` | helper 展开后依赖 `special_i0e` 与数值修正分支 |
| 197 | `torch.special.zeta` | `aten::special_zeta`、`aten::special_zeta.self_scalar` | `aten::special_zeta`、`aten::add.Scalar`、`aten::mul.Tensor`、`aten::neg` | 只有 `other` 方向有梯度公式，`self` 方向是 `not_implemented("zeta")` |
| 198 | `torch.subtract` | `aten::subtract.Tensor`、`aten::subtract.Scalar`、`aten::subtract.out`、`aten::sub.Tensor`、`aten::sub.Scalar`、`aten::sub.out` | `aten::neg`、`aten::mul.Scalar`、`aten::conj` | 原清单写成 `substract`，这里按真实 API `torch.subtract` 汇总；Tensor 重载的 `other` 梯度经 `sub` 的导数规则展开 |
| 199 | `torch.svd` | `aten::svd`、`aten::linalg_svd`、`aten::linalg_svdvals`、`aten::_linalg_svd`、`aten::mH` | `aten::matmul`、`aten::mH`、`aten::transpose.int`、`aten::conj`、`aten::diagonal`、`aten::diag_embed`、`aten::narrow`、`aten::div.Tensor`、`aten::add.Tensor`、`aten::sub.Tensor`、`aten::mul.Tensor` | `compute_uv=True/False` 的前向入口不同；主 backward 挂在 `_linalg_svd`，`Vh.mH()` 还会额外引入 view/conj 回传 |
| 200 | `torch.swapdims` | `aten::swapdims`、`aten::transpose.int`、`aten::as_strided` | `aten::transpose.int` | `swapdims` 只是 `transpose` 别名，反向直接按 `transpose.int` 公式回传 |

## 任务 41：接口 201-205

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 201 | `torch.tanhshrink` | `aten::tanh`、`aten::sub.Tensor` | `aten::tanh_backward`、`aten::sub.Tensor` | 前向就是 `input - input.tanh()`，没有独立 `aten::tanhshrink` schema |
| 202 | `torch.tensor_split` | `aten::tensor_split.sections`、`aten::tensor_split.indices`、`aten::tensor_split.tensor_indices_or_sections`、`aten::slice.Tensor` | `aten::slice_backward` | 返回的是切片 view 列表，梯度按各切片分别回传并累加 |
| 203 | `torch.tensordot` | `aten::tensordot`、`aten::sum`、`aten::permute`、`aten::reshape`、`aten::mm`、`aten::dot`、`aten::squeeze`、`aten::mul.Tensor` | `aten::mm`、`aten::dot`、`aten::sum`、`aten::permute`、`aten::reshape`、`aten::squeeze`、`aten::mul.Tensor` | 实际分支取决于收缩形态、连续性和设备 |
| 204 | `torch.trapezoid` | `aten::trapezoid.x`、`aten::trapezoid.dx`、`aten::view`、`aten::slice.Tensor`、`aten::sub.Tensor`、`aten::add.Tensor`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::sum`、`aten::div`、`aten::select.int` | `aten::slice_backward`、`aten::sub.Tensor`、`aten::add.Tensor`、`aten::mul.Tensor`、`aten::mul.Scalar`、`aten::sum`、`aten::div.Scalar`、`aten::view` | `x` 版与 `dx` 版 backward 结构不同 |
| 205 | `torch.tril_indices` | `aten::tril_indices` | 无 | factory op，无 Tensor 输入，不参与可导链 |

## 任务 42：接口 206-210

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 206 | `torch.triu_indices` | `aten::triu_indices` | 无 | factory / index tensor，不建立梯度链 |
| 207 | `torch.true_divide` | `aten::true_divide.Tensor`、`aten::div.Tensor`、`aten::true_divide.Scalar`、`aten::div.Scalar` | `aten::div.Tensor`、`aten::div.Scalar`、`aten::mul.Tensor`、`aten::neg`、`aten::conj` | `true_divide` 本身无独立 `derivatives.yaml`，backward 继承 `div.*` |
| 208 | `torch.vander` | `aten::vander`、`aten::empty`、`aten::select.int`、`aten::fill_.Scalar`、`aten::slice.Tensor`、`aten::unsqueeze`、`aten::copy_`、`aten::cumprod`、`aten::flip` | `aten::cumprod_backward`、`aten::slice_backward`、`aten::squeeze.dim`、`aten::flip`、`aten::zeros_like` | `increasing=False` 时最后一步会再走一次 `flip` |
| 209 | `torch.view_as_real` | `aten::view_as_real` | `aten::contiguous`、`aten::view_as_complex` | dense complex view，backward 专门把实/虚部重拼回复数 |
| 210 | `torch.vsplit` | `aten::vsplit.int`、`aten::vsplit.array`、`aten::tensor_split.sections`、`aten::tensor_split.indices`、`aten::slice.Tensor` | `aten::slice_backward` | `vsplit` 是 CIA，最终分解成若干个 `slice.Tensor` |

## 任务 43：接口 211

| 序号 | API | 正向 Dispatch 依赖的 ATen 接口 | 反向依赖的 ATen 接口 | 备注 |
| --- | --- | --- | --- | --- |
| 211 | `torch.vstack` | `aten::vstack`、`aten::vstack.out`、`aten::atleast_2d.Sequence`、`aten::unsqueeze`、`aten::reshape`、`aten::cat`、`aten::cat.out` | `aten::narrow`、`aten::squeeze.dim`、`aten::reshape` | CIA 组合接口，主 backward 来自 `cat`，再叠加 `atleast_2d` 内部 view 变换的 backward |

