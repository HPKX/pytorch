# torch API A5（昇腾 NPU）aclnn 接入汇总

## 总览

- 总接口数：211
- A5 可支持：165 个（78.2%）
- A5 不可支持：46 个（21.8%）

## 完整列表

| 序号 | API | A5 支持 | 正向 aclnn 接口 | 反向 aclnn 接口 | 未接入 aclnn 的正反向 ATen 接口 |
|------|-----|:-------:|----------------|----------------|-------------------------------|
| 1 | `torch.absolute` | ✅ | `aclnnAbs` | `aclnnMul`、`aclnnSign` | - |
| 2 | `torch.addcdiv` | ✅ | `aclnnAddcdiv` | `aclnnMul`、`aclnnMuls`、`aclnnDiv`、`aclnnNeg` | - |
| 3 | `torch.addcmul` | ✅ | `aclnnAddcmul` | `aclnnMul`、`aclnnMuls` | - |
| 4 | `torch.addr` | ✅ | `aclnnAddr` | `aclnnMuls`、`aclnnMv` | - |
| 5 | `torch.adjoint` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | 复数 tensor 场景受限（`conj_bit` 可能触发 `_conj_physical`，无 NPU 实现） |
| 6 | `torch.aminmax` | ✅ | `aclnnAminmax` | 无（不可微） | - |
| 7 | `torch.angle` | ✅ | `aclnnAngleV2` | `aclnnSWhere`、`aclnnEqScalar`、`aclnnInplaceZero`、`aclnnMul`、`aclnnMuls`、`aclnnDiv`、`aclnnAbs`、`aclnnPowTensorScalar` | - |
| 8 | `torch.argwhere` | ✅ | `aclnnNonzero` | 无（不可微） | - |
| 9 | `torch.atleast_1d` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 10 | `torch.atleast_2d` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 11 | `torch.atleast_3d` | ✅ | - | - | - |
| 12 | `torch.bartlett_window` | ✅ | `aclnnArange`、`aclnnInplaceMul`/`aclnnInplaceMuls`、`aclnnInplaceAdd`/`aclnnInplaceAdds`、`aclnnInplaceOne` | - | - |
| 13 | `torch.bitwise_left_shift` | ✅ | `aclnnLeftShift`、`aclnnLeftShifts` | - | - |
| 14 | `torch.bitwise_right_shift` | ✅ | `aclnnRightShift` | - | - |
| 15 | `torch.blackman_window` | ✅ | `aclnnArange`、`aclnnMul`/`aclnnMuls`、`aclnnInplaceMul`/`aclnnInplaceMuls`、`aclnnInplaceCos`、`aclnnInplaceOne` | - | - |
| 16 | `torch.block_diag` | ✅ | `aclnnInplaceZero`, `aclnnInplaceCopy` | `aclnnInplaceZero` | - |
| 17 | `torch.bucketize` | ✅ | `aclnnSearchSorted` | - | - |
| 18 | `torch.cholesky` | ✅ | `aclnnLinalgCholesky`, `aclnnInplaceTril`, `aclnnInplaceTriu` | `aclnnTril`, `aclnnMuls`, `aclnnAdd`, `aclnnTriangularSolve` | - |
| 19 | `torch.cholesky_solve` | ❌ | - | `aclnnAdd`, `aclnnNeg` | 正向 `aten::cholesky_solve`（否）、`aten::_cholesky_solve_helper`（否），反向 `aten::cholesky_solve`（否） |
| 20 | `torch.clip` | ✅ | `aclnnClamp`, `aclnnClampTensor` | `aclnnGeScalar`, `aclnnLeScalar`, `aclnnGeTensor`, `aclnnLeTensor`, `aclnnInplaceLogicalAnd`, `aclnnSWhere` | - |
| 21 | `torch.column_stack` | ✅ | `aclnnCat` | - | - |
| 22 | `torch.combinations` | ✅ | `aclnnArange`, `aclnnInplaceFillScalar`, `aclnnMaskedSelect`, `aclnnStack` | `aclnnInplaceZero`, `aclnnInplaceMaskedScatter` | - |
| 23 | `torch.cond` | ✅ | - | - | - |
| 24 | `torch.conj` | ✅ | - | - | - |
| 25 | `torch.copysign` | ❌ | - | `aclnnDiv`, `aclnnEqScalar`, `aclnnInplaceMaskedFillScalar`, `aclnnMul`, `aclnnInplaceZero` | 正向 `aten::copysign.Tensor`（否）、`aten::copysign.Scalar`（否） |
| 26 | `torch.cov` | ✅ | `aclnnMul`, `aclnnReduceSum`, `aclnnSub`, `aclnnMm`, `aclnnInplaceZero`, `aclnnComplex`, `aclnnDiv` | `aclnnMul`, `aclnnNeg`, `aclnnMm`, `aclnnDiv`, `aclnnComplex`, `aclnnInplaceZero` | 复数 tensor 场景受限（`conj_bit` 可能触发 `_conj_physical`，无 NPU 实现） |
| 27 | `torch.deg2rad` | ✅ | `aclnnMul` | `aclnnMuls` | - |
| 28 | `torch.diag_embed` | ✅ | `aclnnInplaceZero`, `aclnnInplaceCopy` | - | - |
| 29 | `torch.diagflat` | ✅ | `aclnnInplaceZero`, `aclnnInplaceCopy` | - | - |
| 30 | `torch.diagonal` | ✅ | - | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 31 | `torch.diagonal_scatter` | ✅ | `aclnnInplaceCopy` | `aclnnInplaceZero` | - |
| 32 | `torch.digamma` | ❌ | - | `aclnnMul` | 正向 `aten::digamma`（否），反向 `aten::polygamma`（否） |
| 33 | `torch.dist` | ✅ | `aclnnSub`, `aclnnNorm` | `aclnnSub`, `aclnnNeg`, `aclnnSign`, `aclnnMul`, `aclnnDiv`, `aclnnInplaceMaskedFillScalar`, `aclnnEqScalar`, `aclnnEqTensor`, `aclnnAbs`, `aclnnLogicalOr`, `aclnnReduceSum` | - |
| 34 | `torch.distribution.gamma.Gamma` | ❌ | `aclnnDiv`, `aclnnXLogYTensor`, `aclnnSub`, `aclnnMul` | `aclnnMul`, `aclnnDiv`, `aclnnNeg` | 正向 `aten::_standard_gamma`（否）、`aten::lgamma`（否），反向 `aten::_standard_gamma_grad`（否） |
| 35 | `torch.distribution.laplace.Laplace` | ✅ | `aclnnInplaceUniform`, `aclnnInplaceRandom`, `aclnnSign`, `aclnnAbs`, `aclnnClamp`, `aclnnNeg`, `aclnnLog1p`, `aclnnMul`, `aclnnSub`, `aclnnLog`, `aclnnDiv` | `aclnnSign`, `aclnnMul`, `aclnnNeg`, `aclnnAdds`, `aclnnDiv`, `aclnnMuls` | - |
| 36 | `torch.distribution.uniform.Uniform` | ✅ | `aclnnInplaceRandom`, `aclnnInplaceUniform`, `aclnnMul`, `aclnnSub`, `aclnnAdd`, `aclnnLeTensor`, `aclnnGtTensor`, `aclnnLog`, `aclnnDiv`, `aclnnClamp` | `aclnnMul`, `aclnnNeg`, `aclnnDiv` | - |
| 37 | `torch.dsplit` | ✅ | - | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 38 | `torch.dstack` | ✅ | `aclnnCat` | `aclnnInplaceZero` | - |
| 39 | `torch.fliplr` | ✅ | `aclnnFlip` | `aclnnFlip` | - |
| 40 | `torch.flipud` | ✅ | `aclnnFlip` | `aclnnFlip` | - |
| 41 | `torch.fmax` | ❌ | - | `aclnnGeTensor`, `aclnnInplaceLogicalOr`, `aclnnInplaceLogicalNot`, `aclnnInplaceMaskedFillScalar` | **`aten::fmax`（正向）** |
| 42 | `torch.fmod` | ✅ | `aclnnFmodScalar`, `aclnnFmodTensor` | `aclnnDivMod`, `aclnnMul`, `aclnnNeg` | - |
| 43 | `torch.gather` | ✅ | `aclnnGather` | `aclnnInplaceZero`, `aclnnScatterAdd` | - |
| 44 | `torch.gcd` | ✅ | `aclnnGcd` | - | - |
| 45 | `torch.ge` | ✅ | `aclnnGeTensor`, `aclnnGeScalar` | - | - |
| 46 | `torch.geqrf` | ❌ | - | - | **`aten::geqrf`（正向）**, **`aten::geqrf.a`（正向）** |
| 47 | `torch.ger` | ✅ | `aclnnMul` | `aclnnMul` | - |
| 48 | `torch.hamming_window` | ✅ | `aclnnArange`, `aclnnInplaceOne`, `aclnnInplaceMuls`, `aclnnInplaceCos`, `aclnnInplaceAdds` | - | - |
| 49 | `torch.hann_window` | ✅ | `aclnnArange`, `aclnnInplaceOne`, `aclnnInplaceMuls`, `aclnnInplaceCos`, `aclnnInplaceAdds` | - | - |
| 50 | `torch.heaviside` | ❌ | - | - | **`aten::heaviside`（正向）**, **`aten::heaviside.out`（正向）**, **`aten::heaviside_`（正向）** |
| 51 | `torch.hsplit` | ✅ | - | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 52 | `torch.hstack` | ✅ | `aclnnCat` | `aclnnInplaceZero` | - |
| 53 | `torch.hypot` | ❌ | - | `aclnnMul`, `aclnnDiv` | **`aten::hypot`（正向）** |
| 54 | `torch.i0` | ❌ | - | `aclnnMul` | **`aten::i0`（正向）**, **`aten::special_i1`（反向）** |
| 55 | `torch.igamma` | ❌ | - | `aclnnSubs`, `aclnnLog`, `aclnnMul`, `aclnnSub`, `aclnnExp` | **`aten::igamma`（正向）**, **`aten::lgamma`（反向）** |
| 56 | `torch.igammac` | ❌ | - | `aclnnNeg`, `aclnnMul`, `aclnnExp`, `aclnnSubs`, `aclnnLog`, `aclnnSub` | **`aten::igammac`（正向）**, **`aten::igammac.out`（正向）**, **`aten::lgamma`（反向）** |
| 57 | `torch.inner` | ✅ | `aclnnMul`, `aclnnMm`, `aclnnDot` | `aclnnMul`, `aclnnMm`, `aclnnDot` | - |
| 58 | `torch.is_complex` | ✅ | - | - | - |
| 59 | `torch.is_floating_point` | ✅ | - | - | - |
| 60 | `torch.is_nonzero` | ✅ | - | - | - |
| 61 | `torch.isnan` | ✅ | `aclnnNeTensor` | - | - |
| 62 | `torch.isposinf` | ✅ | `aclnnIsPosInf` | - | - |
| 63 | `torch.isreal` | ✅ | `aclnnInplaceOne`, `aclnnEqScalar` | - | - |
| 64 | `torch.kaiser_window` | ✅ | `aclnnInplaceOne`, `aclnnArange` | - | - |
| 65 | `torch.kron` | ✅ | `aclnnMul` | `aclnnMul` | - |
| 66 | `torch.lcm` | ❌ | - | 无（不可微） | 正向：`aten::lcm` |
| 67 | `torch.ldexp` | ✅ | `aclnnPowTensorTensor`、`aclnnMul` | `aclnnPowScalarTensor`、`aclnnMul`、`aclnnMuls` | - |
| 68 | `torch.linalg.eigvals` | ❌ | - | `aclnnDiv`、`aclnnInplaceCopy` | 正向：`aten::linalg_eigvals`、`aten::_linalg_eigvals`、`aten::linalg_eig`；反向：`aten::linalg_solve` |
| 69 | `torch.linalg.pinv` | ❌ | `aclnnSvd`、`aclnnSWhere`、`aclnnAmax` | `aclnnNeg`、`aclnnAdd`、`aclnnSub` | 正向：`aten::linalg_pinv`（全部重载）、`aten::linalg_eigh` |
| 70 | `torch.linalg.vecdot` | ✅ | `aclnnDot`、`aclnnMul`、`aclnnReduceSum` | `aclnnMul` | 复数 tensor 场景受限（`conj_bit` 可能触发 `_conj_physical`，无 NPU 实现） |
| 71 | `torch.logcumsumexp` | ❌ | - | `aclnnFlip`、`aclnnAbs`、`aclnnLog`、`aclnnSWhere`、`aclnnSub`、`aclnnAdd`、`aclnnExp`、`aclnnGtScalar`、`aclnnLtScalar` | 正向：`aten::logcumsumexp`、`aten::_logcumsumexp`；反向：`aten::logcumsumexp` |
| 72 | `torch.logdet` | ❌ | `aclnnSlogdet`、`aclnnSign`、`aclnnProd`、`aclnnMul`、`aclnnAbs`、`aclnnInplaceLog`、`aclnnReduceSum`、`aclnnSWhere`、`aclnnLog`、`aclnnAdd` | `aclnnMul`、`aclnnSub` | 反向：`aten::linalg_lu_solve`、`aten::linalg_solve`（正向 `logdet` CIA→`linalg_slogdet` aclnnSlogdet=是，正向不阻塞） |
| 73 | `torch.logit` | ✅ | `aclnnLogit` | `aclnnLogitGrad`、`aclnnLogicalAnd`、`aclnnGeScalar`、`aclnnLeScalar`、`aclnnSWhere`、`aclnnDiv`、`aclnnMul`、`aclnnSubs`、`aclnnInplaceZero`、`aclnnInplaceFillScalar` | - |
| 74 | `torch.logspace` | ✅ | `aclnnLogSpace` | 无（工厂函数） | - |
| 75 | `torch.lu_solve` | ❌ | - | `aclnnNeg`、`aclnnTriangularSolve`、`aclnnTril`、`aclnnTriu`、`aclnnAdd` | 正向：`aten::lu_solve`、`aten::linalg_lu_solve`（全部重载）；反向：`aten::linalg_lu_solve`、`aten::lu_unpack` |
| 76 | `torch.lu_unpack` | ❌ | `aclnnTriu`、`aclnnTril`、`aclnnInplaceFillScalar`、`aclnnArange`、`aclnnInplaceZero`、`aclnnScatterValue` | `aclnnTril`、`aclnnTriu`、`aclnnAdd`、`aclnnCat`、`aclnnInplaceZero` | 正向：`aten::lu_unpack` |
| 77 | `torch.moveaxis` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 78 | `torch.movedim` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 79 | `torch.msort` | ✅ | `aclnnSort` | `aclnnInplaceZero`、`aclnnInplaceScatter`、`aclnnScatter` | - |
| 80 | `torch.multiply` | ✅ | `aclnnMul`、`aclnnMuls` | `aclnnMul`、`aclnnMuls` | - |
| 81 | `torch.mvlgamma` | ❌ | `aclnnArange`、`aclnnAdd`、`aclnnReduceSum`、`aclnnInplaceAdds` | `aclnnArange`、`aclnnAdd`、`aclnnReduceSum`、`aclnnMul` | 正向：`aten::mvlgamma`、`aten::lgamma_`；反向：`aten::digamma_` |
| 82 | `torch.nanmean` | ✅ | `aclnnInplaceLogicalNot`、`aclnnReduceSum`、`aclnnReduceNansum`、`aclnnDiv` | `aclnnLogicalNot`、`aclnnMul`、`aclnnDiv`、`aclnnNeg` | - |
| 83 | `torch.nanmedian` | ✅ | `aclnnNanMedian`、`aclnnNanMedianDim` | `aclnnInplaceLogicalAnd`、`aclnnInplaceLogicalOr`、`aclnnEqTensor`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMul`、`aclnnInplaceZero`、`aclnnInplaceMaskedFillScalar`、`aclnnInplaceScatter` | - |
| 84 | `torch.nextafter` | ❌ | - | 无（不可微） | 正向：`aten::nextafter` |
| 85 | `torch.nn.AdaptiveMaxPool3d` | ✅ | `aclnnAdaptiveMaxPool3d` | `aclnnAdaptiveMaxPool3dBackward` | - |
| 86 | `torch.nn.AvgPool1d` | ✅ | `aclnnAvgPool2d` | `aclnnAvgPool2dBackward` | - |
| 87 | `torch.nn.CELU` | ✅ | `aclnnCelu`、`aclnnElu` | `aclnnEluBackward` | - |
| 88 | `torch.nn.CTCLoss` | ✅ | `aclnnCtcLoss` | `aclnnCtcLossBackward` | - |
| 89 | `torch.nn.ChannelShuffle` | ✅ | `aclnnChannelShuffle` | `aclnnChannelShuffle` | - |
| 90 | `torch.nn.ConvTranspose1d` | ✅ | `aclnnConvolution` | `aclnnConvolutionBackward` | - |
| 91 | `torch.nn.ConvTranspose3d` | ✅ | `aclnnConvolution` | `aclnnConvolutionBackward` | - |
| 92 | `torch.nn.Dropout1d` | ✅ | `aclnnInplaceBernoulli`, `aclnnInplaceDiv`, `aclnnMul` | `aclnnMul` | - |
| 93 | `torch.nn.Dropout2d` | ✅ | `aclnnInplaceBernoulli`, `aclnnInplaceDiv`, `aclnnMul` | `aclnnMul` | - |
| 94 | `torch.nn.Dropout3d` | ✅ | `aclnnInplaceBernoulli`, `aclnnInplaceDiv`, `aclnnMul` | `aclnnMul` | - |
| 95 | `torch.nn.GRU` | ❌ | `aclnnAddmm`, `aclnnInplaceSigmoid`, `aclnnInplaceTanh`, `aclnnInplaceMul`, `aclnnInplaceAdd` | `aclnnSigmoidBackward`, `aclnnTanhBackward`, `aclnnMul`, `aclnnSub`, `aclnnCat`, `aclnnReduceSum` | **`aten::gru.input`（正向）**, **`aten::gru.data`（正向）**, **`aten::_thnn_fused_gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell_backward`（反向）**, **`aten::_thnn_differentiable_gru_cell_backward`（反向）** |
| 96 | `torch.nn.GRUCell` | ❌ | `aclnnAddmm`, `aclnnInplaceAdd`, `aclnnInplaceSigmoid`, `aclnnInplaceMul`, `aclnnInplaceTanh`, `aclnnSub` | `aclnnSigmoidBackward`, `aclnnTanhBackward`, `aclnnMul`, `aclnnSub`, `aclnnCat`, `aclnnReduceSum` | **`aten::gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell_backward`（反向）**, **`aten::_thnn_differentiable_gru_cell_backward`（反向）** |
| 97 | `torch.nn.GaussianNLLLoss` | ✅ | `aclnnLtScalar`, `aclnnAny`, `aclnnInplaceCopy`, `aclnnClamp`, `aclnnSub`, `aclnnPowTensorScalar`, `aclnnDiv`, `aclnnLog`, `aclnnAdd`, `aclnnMuls`, `aclnnMean` | `aclnnNeg`, `aclnnMul`, `aclnnMuls`, `aclnnDiv`, `aclnnPowTensorScalar`, `aclnnDivs` | - |
| 98 | `torch.nn.HardTanh` | ✅ | `aclnnHardtanh`, `aclnnClamp` | `aclnnHardtanhBackward` | - |
| 99 | `torch.nn.HingeEmbeddingLoss` | ✅ | `aclnnInplaceZero`, `aclnnRsubs`, `aclnnClampMin`, `aclnnSWhere`, `aclnnAdd`, `aclnnMean` | `aclnnNeg`, `aclnnGeScalar`, `aclnnSWhere`, `aclnnDivs` | - |
| 100 | `torch.nn.HuberLoss` | ❌ | `aclnnMean` | - | **`aten::huber_loss`（正向）**, **`aten::huber_loss_backward`（反向）** |
| 101 | `torch.nn.InstanceNorm1d` | ✅ | `aclnnBatchNorm` | `aclnnBatchNormBackward` | - |
| 102 | `torch.nn.InstanceNorm2d` | ✅ | `aclnnBatchNorm` | `aclnnBatchNormBackward` | - |
| 103 | `torch.nn.InstanceNorm3d` | ✅ | `aclnnBatchNorm` | `aclnnBatchNormBackward` | - |
| 104 | `torch.nn.LPPool1d` | ✅ | `aclnnPowTensorScalar`, `aclnnAvgPool2d`, `aclnnSign`, `aclnnAbs`, `aclnnRelu`, `aclnnMul`, `aclnnMuls` | `aclnnPowTensorScalar`, `aclnnMul`, `aclnnMuls`, `aclnnInplaceZero`, `aclnnAvgPool2dBackward`, `aclnnThresholdBackward`, `aclnnSign` | - |
| 105 | `torch.nn.LPPool2d` | ✅ | `aclnnPowTensorScalar`, `aclnnAvgPool2d`, `aclnnSign`, `aclnnAbs`, `aclnnRelu`, `aclnnMul`, `aclnnMuls` | `aclnnPowTensorScalar`, `aclnnMul`, `aclnnMuls`, `aclnnInplaceZero`, `aclnnAvgPool2dBackward`, `aclnnThresholdBackward`, `aclnnSign` | - |
| 106 | `torch.nn.LSTM` | ❌ | - | - | **`aten::lstm.input`**、**`aten::_cudnn_rnn`**、**`aten::_cudnn_rnn_backward`** |
| 107 | `torch.nn.LSTMCell` | ❌ | - | - | **`aten::lstm_cell`**、**`aten::_thnn_fused_lstm_cell`**、**`aten::_thnn_differentiable_lstm_cell_backward`**、**`aten::_thnn_fused_lstm_cell_backward`** |
| 108 | `torch.nn.LeakyReLU` | ✅ | `aclnnLeakyRelu` | `aclnnLeakyReluBackward` | - |
| 109 | `torch.nn.MarginRankingLoss` | ✅ | `aclnnSub`、`aclnnNeg`、`aclnnMul`、`aclnnAdds`、`aclnnClampMin`、`aclnnMean` | `aclnnSub`、`aclnnNeg`、`aclnnMul`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | - |
| 110 | `torch.nn.MaxPool1d` | ✅ | `aclnnMaxPool2dWithIndices` | `aclnnMaxPool2dWithIndicesBackward` | - |
| 111 | `torch.nn.MaxPool2d` | ✅ | `aclnnMaxPool2dWithIndices` | `aclnnMaxPool2dWithIndicesBackward` | - |
| 112 | `torch.nn.MaxPool3d` | ✅ | `aclnnMaxPool3dWithArgmax` | `aclnnMaxPool3dWithArgmaxBackward` | - |
| 113 | `torch.nn.MaxUnpool1d` | ✅ | `aclnnMaxUnpool2d` | `aclnnGather` | - |
| 114 | `torch.nn.MaxUnpool3d` | ✅ | `aclnnMaxUnpool3d` | `aclnnGather` | - |
| 115 | `torch.nn.MultiLabelMarginLoss` | ❌ | `aclnnMultilabelMarginLoss` | - | **`aten::multilabel_margin_loss_backward`** |
| 116 | `torch.nn.MultiLabelSoftMarginLoss` | ✅ | `aclnnLogSigmoidForward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | `aclnnLogSigmoidBackward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | - |
| 117 | `torch.nn.MultiMarginLoss` | ❌ | - | - | **`aten::multi_margin_loss`**、**`aten::multi_margin_loss_backward`** |
| 118 | `torch.nn.MultiheadAttention` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnBatchMatMul`、`aclnnSoftmax`、`aclnnDropoutGenMaskV2`、`aclnnBaddbmm` | `aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnBatchMatMul`、`aclnnAddmm`、`aclnnBaddbmm` | - |
| 119 | `torch.nn.PixelUnshuffle` | ✅ | `aclnnInplaceCopy`（clone） | `aclnnInplaceCopy`（clone） | - |
| 120 | `torch.nn.PoissonNLLLoss` | ✅ | `aclnnExp`、`aclnnMul`、`aclnnSub`、`aclnnAdds`、`aclnnLog`、`aclnnLeTensor`、`aclnnInplaceMaskedFillScalar`、`aclnnMean` | `aclnnExp`、`aclnnMul`、`aclnnSub`、`aclnnAdds`、`aclnnLog`、`aclnnLeTensor`、`aclnnMean`、`aclnnReduceSum` | - |
| 121 | `torch.nn.RNN` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnInplaceAdd`、`aclnnTanh`、`aclnnRelu`、`aclnnDropoutGenMaskV2`、`aclnnCat` | `aclnnAddmm`、`aclnnMm`、`aclnnTanhBackward`、`aclnnThresholdBackward`、`aclnnDropoutDoMask`、`aclnnCat` | - |
| 122 | `torch.nn.RNNCell` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnInplaceAdd`、`aclnnTanh`、`aclnnRelu` | `aclnnAddmm`、`aclnnMm`、`aclnnAdd`、`aclnnTanhBackward`、`aclnnThresholdBackward` | - |
| 123 | `torch.nn.RReLU` | ✅ | `aclnnRReluWithNoise` | `aclnnMul`、`aclnnLeakyReluBackward` | - |
| 124 | `torch.nn.SoftMarginLoss` | ✅ | `aclnnSoftMarginLoss` | `aclnnSoftMarginLossBackward` | - |
| 125 | `torch.nn.Softmax2d` | ✅ | `aclnnSoftmax` | `aclnnSoftmaxBackward` | - |
| 126 | `torch.nn.Softmin` | ✅ | `aclnnNeg`、`aclnnSoftmax` | `aclnnSoftmaxBackward`、`aclnnNeg` | - |
| 127 | `torch.nn.Softsign` | ✅ | `aclnnAbs`、`aclnnAdds`、`aclnnDiv` | `aclnnDiv`、`aclnnAbs`、`aclnnSign`、`aclnnAdds`、`aclnnMul`、`aclnnNeg` | - |
| 128 | `torch.nn.Tanhshrink` | ✅ | `aclnnTanh`、`aclnnSub` | `aclnnTanhBackward`、`aclnnNeg` | - |
| 129 | `torch.nn.Transformer` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnSoftmax`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 130 | `torch.nn.TransformerDecoder` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnSoftmax`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 131 | `torch.nn.TransformerDecoderLayer` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 132 | `torch.nn.TransformerEncoder` | ✅ | `aclnnFlashAttentionScore`、`aclnnLayerNorm` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 133 | `torch.nn.TransformerEncoderLayer` | ✅ | `aclnnAddmm`、`aclnnMm`、`aclnnFlashAttentionScore`、`aclnnDropoutGenMaskV2`、`aclnnLayerNorm`、`aclnnRelu`、`aclnnGelu` | `aclnnLayerNormBackward`、`aclnnSoftmaxBackward`、`aclnnDropoutDoMask`、`aclnnAddmm`、`aclnnThresholdBackward`、`aclnnGeluBackward`、`aclnnAdd` | - |
| 134 | `torch.nn.TripletMarginLoss` | ✅ | `aclnnNorm`、`aclnnClampMin`、`aclnnMinimum`、`aclnnMean`、`aclnnReduceSum` | `aclnnNorm`、`aclnnMinimum`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | - |
| 135 | `torch.nn.functional.affine_grid` | ✅ | `aclnnAffineGrid` | `aclnnBatchMatMul` | - |
| 136 | `torch.nn.functional.celu` | ✅ | `aclnnCelu`、`aclnnElu` | `aclnnEluBackward` | - |
| 137 | `torch.nn.functional.cosine_similarity` | ✅ | `aclnnLinalgVectorNorm`、`aclnnDiv`、`aclnnMul`、`aclnnReduceSum`、`aclnnClampMin` | `aclnnDiv`、`aclnnMul`、`aclnnReduceSum` | - |
| 138 | `torch.nn.functional.ctc_loss` | ✅ | `aclnnCtcLoss` | `aclnnCtcLossBackward` | - |
| 139 | `torch.nn.functional.dropout1d` | ✅ | `aclnnInplaceBernoulli`、`aclnnInplaceDiv`、`aclnnMul` | `aclnnMul` | - |
| 140 | `torch.nn.functional.dropout2d` | ✅ | `aclnnInplaceBernoulli`、`aclnnInplaceDiv`、`aclnnMul` | `aclnnMul` | - |
| 141 | `torch.nn.functional.dropout3d` | ✅ | `aclnnInplaceBernoulli`、`aclnnInplaceDiv`、`aclnnMul` | `aclnnMul` | `aten::feature_dropout`（正向，CIA 分解后子算子均可支持） |
| 142 | `torch.nn.functional.gaussian_nll_loss` | ✅ | `aclnnInplaceCopy`、`aclnnClamp`、`aclnnLog`、`aclnnSub`、`aclnnPowTensorScalar`、`aclnnDiv`、`aclnnAdd`、`aclnnMuls`、`aclnnMean` | `aclnnLog`、`aclnnSub`、`aclnnPowTensorScalar`、`aclnnDiv`、`aclnnAdd`、`aclnnMuls`、`aclnnMean`、`aclnnReduceSum` | - |
| 143 | `torch.nn.functional.gumbel_softmax` | ✅ | `aclnnLog`、`aclnnNeg`、`aclnnAdd`、`aclnnDivs`、`aclnnSoftmax`、`aclnnMaxDim`、`aclnnInplaceZero`、`aclnnScatterValue`、`aclnnSub` | `aclnnSoftmaxBackward`、`aclnnDivs`、`aclnnAdd`、`aclnnNeg`、`aclnnSub` | `aten::exponential_`（正向，NPU 通过已有 aclnn 算子组合实现） |
| 144 | `torch.nn.functional.hardtanh` | ✅ | `aclnnHardtanh`、`aclnnClamp` | `aclnnHardtanhBackward` | - |
| 145 | `torch.nn.functional.hinge_embedding_loss` | ✅ | `aclnnInplaceZero`、`aclnnSubs`、`aclnnClampMin`、`aclnnSWhere`、`aclnnAdd`、`aclnnMean`、`aclnnReduceSum` | `aclnnSWhere`、`aclnnClampMin`、`aclnnAdd`、`aclnnMean`、`aclnnReduceSum` | `aten::hinge_embedding_loss`（正向，CIA 分解后子算子均可支持） |
| 146 | `torch.nn.functional.huber_loss` | ❌ | `aclnnMean`、`aclnnReduceSum`、`aclnnMul` | `aclnnMul`、`aclnnMean`、`aclnnReduceSum` | **`aten::huber_loss`（正向）、`aten::huber_loss_backward`（反向）** |
| 147 | `torch.nn.functional.lp_pool1d` | ✅ | `aclnnPowTensorScalar`、`aclnnAvgPool2d`、`aclnnSign`、`aclnnAbs`、`aclnnRelu`、`aclnnMuls` | `aclnnPowTensorScalar`、`aclnnAvgPool2dBackward`、`aclnnSign`、`aclnnThresholdBackward`、`aclnnMul`、`aclnnMuls`、`aclnnInplaceZero` | `aten::avg_pool1d`（正向，CIA 分解后子算子均可支持） |
| 148 | `torch.nn.functional.lp_pool2d` | ✅ | `aclnnPowTensorScalar`、`aclnnAvgPool2d`、`aclnnSign`、`aclnnAbs`、`aclnnRelu`、`aclnnMuls` | `aclnnPowTensorScalar`、`aclnnAvgPool2dBackward`、`aclnnSign`、`aclnnThresholdBackward`、`aclnnMul`、`aclnnMuls`、`aclnnInplaceZero` | - |
| 149 | `torch.nn.functional.margin_ranking_loss` | ✅ | `aclnnSub`、`aclnnMul`、`aclnnNeg`、`aclnnAdds`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | `aclnnSub`、`aclnnMul`、`aclnnNeg`、`aclnnAdds`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | `aten::margin_ranking_loss`（正向，CIA 分解后子算子均可支持） |
| 150 | `torch.nn.functional.max_pool3d` | ✅ | `aclnnMaxPool3dWithArgmax` | `aclnnMaxPool3dWithArgmaxBackward` | `aten::max_pool3d`（正向，CIA 包装层，委托给已支持的 with_indices 变体） |
| 151 | `torch.nn.functional.max_unpool1d` | ✅ | `aclnnMaxUnpool2d` | `aclnnGather` | - |
| 152 | `torch.nn.functional.max_unpool3d` | ✅ | `aclnnMaxUnpool3d` | `aclnnGather` | - |
| 153 | `torch.nn.functional.multi_margin_loss` | ❌ | - | - | **`aten::multi_margin_loss`（正向）、`aten::multi_margin_loss_backward`（反向）** |
| 154 | `torch.nn.functional.multilabel_margin_loss` | ❌ | `aclnnMultilabelMarginLoss` | - | **`aten::multilabel_margin_loss_backward`（反向）** |
| 155 | `torch.nn.functional.multilabel_soft_margin_loss` | ✅ | `aclnnLogSigmoidForward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | `aclnnLogSigmoidBackward`、`aclnnNeg`、`aclnnRsubs`、`aclnnMul`、`aclnnAdd`、`aclnnReduceSum`、`aclnnDivs`、`aclnnMean` | - |
| 156 | `torch.nn.functional.pixel_unshuffle` | ✅ | `aclnnInplaceCopy` | - | `aten::pixel_unshuffle`（正向）、`aten::pixel_shuffle`（反向），均可通过 CEA 分解为 view + clone 执行 |
| 157 | `torch.nn.functional.rms_norm` | ✅ | `aclnnRmsNorm`、`aclnnPowTensorScalar`、`aclnnMean`、`aclnnInplaceAdd`、`aclnnRsqrt`、`aclnnMul` | `aclnnPowTensorScalar`、`aclnnMean`、`aclnnInplaceAdd`、`aclnnRsqrt`、`aclnnMul` | `aten::_fused_rms_norm`、`aten::_fused_rms_norm_backward`（NPU 走自定义 npu_rms_norm 路径，不影响支持） |
| 158 | `torch.nn.functional.rrelu` | ✅ | `aclnnRReluWithNoise`、`aclnnLeakyRelu` | `aclnnMul`、`aclnnLeakyReluBackward` | `aten::rrelu`（正向，CIA 壳）、`aten::rrelu_with_noise_backward`（反向，helper 展开后子算子均可支持） |
| 159 | `torch.nn.functional.soft_margin_loss` | ✅ | `aclnnSoftMarginLoss`、`aclnnNeg`、`aclnnInplaceMul`、`aclnnInplaceExp`、`aclnnInplaceLog1p`、`aclnnMean` | `aclnnSoftMarginLossBackward` | - |
| 160 | `torch.nn.functional.softmin` | ✅ | `aclnnNeg`、`aclnnSoftmax` | `aclnnNeg`、`aclnnSoftmaxBackward` | - |
| 161 | `torch.nn.functional.softsign` | ✅ | `aclnnAbs`、`aclnnAdds`、`aclnnDiv` | `aclnnDiv`、`aclnnSign`、`aclnnMul`、`aclnnAdds` | - |
| 162 | `torch.nn.functional.triplet_margin_loss` | ✅ | `aclnnSub`、`aclnnAdds`、`aclnnNorm`、`aclnnMinimum`、`aclnnClampMin`、`aclnnMean` | `aclnnSub`、`aclnnAdds`、`aclnnNorm`、`aclnnMinimum`、`aclnnClampMin`、`aclnnMean`、`aclnnReduceSum` | `aten::triplet_margin_loss`、`aten::pairwise_distance`（均为 CIA 分解，子算子均可支持） |
| 163 | `torch.nn.functional.upsample` | ✅ | `aclnnUpsampleNearest2d`、`aclnnUpsampleNearest2dV2`、`aclnnUpsampleBilinear2d` | `aclnnUpsampleNearest2dBackward`、`aclnnUpsampleBilinear2dBackward` | - |
| 164 | `torch.nn.modules.ChannelShuffle` | ✅ | `aclnnChannelShuffle` | `aclnnChannelShuffle` | - |
| 165 | `torch.nn.modules.flatten.Flatten` | ✅ | 全为 view，无需 aclnn | 全为 view backward | - |
| 166 | `torch.nn.modules.flatten.Unflatten` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 167 | `torch.numel` | ✅ | 无 ATen 依赖 | 无（不可微） | - |
| 168 | `torch.optim.ASGD` | ✅ | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnSub`、`aclnnInplaceCopy`、`aclnnForeachAddScalar`、`aclnnForeachNeg`、`aclnnForeachAddList`、`aclnnForeachAddcmulScalar`、`aclnnForeachSubList`、`aclnnForeachCopy`、`aclnnForeachMaximumScalar`、`aclnnForeachPowScalar`、`aclnnForeachReciprocal` | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnSub`、`aclnnInplaceCopy` | - |
| 169 | `torch.optim.Adamax` | ✅ | `aclnnInplaceAdd`、`aclnnAdd`、`aclnnInplaceLerps`、`aclnnInplaceMul`、`aclnnAbs`、`aclnnMaximum`、`aclnnInplaceAddcdiv`、`aclnnForeachAddScalar`、`aclnnForeachAddList`、`aclnnForeachLerpScalar`、`aclnnForeachMulScalar`、`aclnnForeachAbs`、`aclnnForeachMaximumList`、`aclnnForeachPowList`、`aclnnForeachDivScalar`、`aclnnForeachMulList`、`aclnnCat`、`aclnnAmax` | `aclnnInplaceAdd`、`aclnnAdd`、`aclnnInplaceLerps`、`aclnnInplaceMul`、`aclnnAbs`、`aclnnMaximum`、`aclnnInplaceAddcdiv` | `aten::_foreach_addcdiv_.ScalarList`（正向，foreach 路径 fallback，不影响功能） |
| 170 | `torch.optim.RMSprop` | ✅ | `aclnnInplaceAdd`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnInplaceAddcmul`、`aclnnSqrt`、`aclnnInplaceAddcdiv`、`aclnnInplaceLerps`、`aclnnAddcmul`、`aclnnInplaceSqrt`、`aclnnForeachAddScalar`、`aclnnForeachAddList`、`aclnnForeachMulScalar`、`aclnnForeachAddcmulScalar`、`aclnnForeachLerpScalar`、`aclnnForeachSqrt`、`aclnnForeachAddcdivScalar` | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnInplaceAddcmul`、`aclnnInplaceLerps`、`aclnnSqrt`、`aclnnInplaceSqrt`、`aclnnInplaceAddcdiv` | - |
| 171 | `torch.optim.Rprop` | ✅ | `aclnnNeg`、`aclnnMul`、`aclnnSign`、`aclnnInplaceMul`、`aclnnClamp`、`aclnnInplaceAddcmul`、`aclnnInplaceCopy`、`aclnnForeachAddScalar`、`aclnnForeachMulList`、`aclnnForeachNeg`、`aclnnForeachCopy`、`aclnnForeachSign`、`aclnnForeachAddcmulScalar` | 无（优化器，不可微） | - |
| 172 | `torch.optim.adadelta.Adadelta` | ✅ | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceMul`、`aclnnInplaceAddcmul`、`aclnnInplaceSqrt`、`aclnnInplaceDiv`、`aclnnForeachAddScalar`、`aclnnForeachNeg`、`aclnnForeachAddScalar`、`aclnnForeachMulList`、`aclnnForeachAddcmulScalar`、`aclnnForeachSqrt`、`aclnnForeachDivScalar` | 无（优化器，不可微） | - |
| 173 | `torch.optim.adagrad.Adagrad` | ✅ | `aclnnInplaceAdd`、`aclnnNeg`、`aclnnAdd`、`aclnnInplaceAddcmul`、`aclnnSqrt`、`aclnnInplaceSqrt`、`aclnnInplaceAddcdiv`、`aclnnPowTensorTensor`、`aclnnForeachAddScalar`、`aclnnForeachNeg`、`aclnnForeachAddcmulScalar`、`aclnnForeachSqrt`、`aclnnForeachMulList`、`aclnnForeachAddcdivScalar` | 无（优化器，不可微） | `aten::sparse_mask`（仅 sparse grad 路径）、`aten::_fused_adagrad_`（仅 CPU fused 路径），常规 NPU 路径不依赖 |
| 174 | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` | ✅ | 无 ATen 依赖 | 无（不可微） | - |
| 175 | `torch.orgqr` | ❌ | `aclnnInplaceCopy` | `aclnnTril`、`aclnnInplaceFillScalar`、`aclnnReduceSum`、`aclnnInplaceZero`、`aclnnCat`、`aclnnInplaceCopy` | **`aten::orgqr`（正向）、`aten::linalg_householder_product`（正反向）、`aten::linalg_householder_product.out`（正向）** |
| 176 | `torch.ormqr` | ❌ | `aclnnInplaceCopy` | `aclnnTril`, `aclnnInplaceFillScalar`, `aclnnReduceSum`, `aclnnInplaceZero`, `aclnnCat`, `aclnnInplaceCopy` | 正向 **`aten::ormqr`（否）**，反向 **`aten::ormqr`（否）** |
| 177 | `torch.pdist` | ❌ | `aclnnPdist` | `aclnnReduceSum` | 反向 **`aten::_pdist_backward`（否）** |
| 178 | `torch.poisson` | ❌ | `aclnnInplaceZero` | `aclnnInplaceZero` | 正向 **`aten::poisson`（否）** |
| 179 | `torch.polygamma` | ❌ | - | `aclnnMul` | 正向 **`aten::polygamma`（否）**，反向 **`aten::polygamma`（否）** |
| 180 | `torch.positive` | ✅ | 全为 view，无需 aclnn | - | - |
| 181 | `torch.rad2deg` | ✅ | `aclnnMul` | `aclnnMuls` | - |
| 182 | `torch.range` | ✅ | `aclnnRange` | - | - |
| 183 | `torch.renorm` | ✅ | `aclnnRenorm`, `aclnnLinalgVectorNorm`, `aclnnMul` | `aclnnLinalgVectorNorm`, `aclnnMul`, `aclnnReduceSum`, `aclnnSign`, `aclnnDiv`, `aclnnInplaceMaskedFillScalar`, `aclnnEqScalar`, `aclnnAbs`, `aclnnLogicalOr`, `aclnnReciprocal`, `aclnnAdds`, `aclnnMuls`, `aclnnSub`, `aclnnSWhere`, `aclnnGtScalar` | - |
| 184 | `torch.rot90` | ✅ | `aclnnFlip`, `aclnnInplaceCopy` | `aclnnFlip`, `aclnnInplaceCopy` | - |
| 185 | `torch.row_stack` | ✅ | `aclnnCat` | 全为 view，无需 aclnn | - |
| 186 | `torch.select_scatter` | ✅ | `aclnnInplaceCopy` | `aclnnInplaceZero` | - |
| 187 | `torch.sgn` | ✅ | `aclnnSign` | `aclnnAbs`, `aclnnMul`, `aclnnDiv`, `aclnnInplaceMaskedFillScalar`, `aclnnEqScalar` | 复数 tensor 场景受限（`conj_bit` 可能触发 `_conj_physical`，无 NPU 实现） |
| 188 | `torch.signbit` | ✅ | `aclnnSignbit`, `aclnnInplaceFillScalar` | - | - |
| 189 | `torch.slogdet` | ❌ | `aclnnSlogdet`, `aclnnSvd`, `aclnnSign`, `aclnnProd`, `aclnnMul`, `aclnnAbs`, `aclnnInplaceLog`, `aclnnReduceSum` | - | 正向 **`aten::linalg_lu_factor_ex.out`（否）**，反向 **`aten::linalg_lu_solve`（否）**、**`aten::linalg_solve`（否）** |
| 190 | `torch.special.bessel_j0` | ❌ | - | - | 正向 **`aten::special_bessel_j0`（否）** |
| 191 | `torch.special.bessel_j1` | ❌ | - | - | 正向 **`aten::special_bessel_j1`（否）** |
| 192 | `torch.special.bessel_y0` | ❌ | - | - | 正向 **`aten::special_bessel_y0`（否）** |
| 193 | `torch.special.bessel_y1` | ❌ | - | - | 正向 **`aten::special_bessel_y1`（否）** |
| 194 | `torch.special.i0` | ❌ | - | `aclnnMul` | 正向 **`aten::special_i0`（否）**、**`aten::i0`（否）**，反向 **`aten::special_i1`（否）** |
| 195 | `torch.special.i1` | ❌ | - | `aclnnAbs`, `aclnnSWhere`, `aclnnReciprocal`, `aclnnSub`, `aclnnMul` | 正向 **`aten::special_i1`（否）**，反向 **`aten::i0`（否）** |
| 196 | `torch.special.i1e` | ❌ | - | `aclnnSign`, `aclnnReciprocal`, `aclnnSWhere`, `aclnnAbs`, `aclnnMul`, `aclnnSub` | 正向 **`aten::special_i1e`（否）**，反向 **`aten::special_i0e`（否）** |
| 197 | `torch.special.zeta` | ❌ | - | `aclnnAdds`, `aclnnMul`, `aclnnNeg` | 正向 **`aten::special_zeta`（否）**、**`aten::special_zeta.self_scalar`（否）**，反向 **`aten::special_zeta`（否）** |
| 198 | `torch.subtract` | ✅ | `aclnnSub`, `aclnnSubs` | `aclnnNeg`, `aclnnMuls` | - |
| 199 | `torch.svd` | ✅ | `aclnnSvd` | `aclnnDiv`, `aclnnAdd`, `aclnnSub`, `aclnnMul` | - |
| 200 | `torch.swapdims` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 201 | `torch.tanhshrink` | ✅ | `aclnnTanh`, `aclnnSub` | `aclnnTanhBackward`, `aclnnNeg` | - |
| 202 | `torch.tensor_split` | ✅ | 全为 view，无需 aclnn | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 203 | `torch.tensordot` | ✅ | `aclnnReduceSum`, `aclnnMm`, `aclnnDot`, `aclnnMul` | `aclnnMm`, `aclnnDot`, `aclnnReduceSum`, `aclnnMul` | - |
| 204 | `torch.trapezoid` | ✅ | `aclnnSub`, `aclnnAdd`, `aclnnMul`, `aclnnMuls`, `aclnnReduceSum`, `aclnnDiv` | `aclnnSub`, `aclnnAdd`, `aclnnMul`, `aclnnMuls`, `aclnnReduceSum`, `aclnnDivs` | - |
| 205 | `torch.tril_indices` | ❌ | - | - | 正向 **`aten::tril_indices`（否）** |
| 206 | `torch.triu_indices` | ❌ | - | - | 正向 **`aten::triu_indices`（否）** |
| 207 | `torch.true_divide` | ✅ | `aclnnDiv`, `aclnnDivs` | `aclnnDiv`, `aclnnDivs`, `aclnnMul`, `aclnnNeg` | - |
| 208 | `torch.vander` | ✅ | `aclnnInplaceFillScalar`, `aclnnInplaceCopy`, `aclnnCumprod`, `aclnnFlip` | `aclnnFlip`, `aclnnInplaceZero` | - |
| 209 | `torch.view_as_real` | ✅ | 全为 view，无需 aclnn | 全为 view，无需 aclnn | - |
| 210 | `torch.vsplit` | ✅ | 全为 view，无需 aclnn | `aclnnInplaceZero`, `aclnnInplaceCopy` | - |
| 211 | `torch.vstack` | ✅ | `aclnnCat` | 全为 view，无需 aclnn | - |

## 不可支持接口清单

| 序号 | API | 阻塞原因（未接入 aclnn 的 ATen 接口） |
|------|-----|--------------------------------------|
| 1 | `torch.cholesky_solve` | 正向 `aten::cholesky_solve`（否）、`aten::_cholesky_solve_helper`（否），反向 `aten::cholesky_solve`（否） |
| 2 | `torch.copysign` | 正向 `aten::copysign.Tensor`（否）、`aten::copysign.Scalar`（否） |
| 3 | `torch.digamma` | 正向 `aten::digamma`（否），反向 `aten::polygamma`（否） |
| 4 | `torch.distribution.gamma.Gamma` | 正向 `aten::_standard_gamma`（否）、`aten::lgamma`（否），反向 `aten::_standard_gamma_grad`（否） |
| 5 | `torch.fmax` | **`aten::fmax`（正向）** |
| 6 | `torch.geqrf` | **`aten::geqrf`（正向）**, **`aten::geqrf.a`（正向）** |
| 7 | `torch.heaviside` | **`aten::heaviside`（正向）**, **`aten::heaviside.out`（正向）**, **`aten::heaviside_`（正向）** |
| 8 | `torch.hypot` | **`aten::hypot`（正向）** |
| 9 | `torch.i0` | **`aten::i0`（正向）**, **`aten::special_i1`（反向）** |
| 10 | `torch.igamma` | **`aten::igamma`（正向）**, **`aten::lgamma`（反向）** |
| 11 | `torch.igammac` | **`aten::igammac`（正向）**, **`aten::igammac.out`（正向）**, **`aten::lgamma`（反向）** |
| 12 | `torch.lcm` | 正向：`aten::lcm` |
| 13 | `torch.linalg.eigvals` | 正向：`aten::linalg_eigvals`、`aten::_linalg_eigvals`、`aten::linalg_eig`；反向：`aten::linalg_solve` |
| 14 | `torch.linalg.pinv` | 正向：`aten::linalg_pinv`（全部重载）、`aten::linalg_eigh` |
| 15 | `torch.logcumsumexp` | 正向：`aten::logcumsumexp`、`aten::_logcumsumexp`；反向：`aten::logcumsumexp` |
| 16 | `torch.logdet` | 反向：`aten::linalg_lu_solve`、`aten::linalg_solve`（正向 `logdet` CIA→`linalg_slogdet` aclnnSlogdet=是，正向不阻塞） |
| 17 | `torch.lu_solve` | 正向：`aten::lu_solve`、`aten::linalg_lu_solve`（全部重载）；反向：`aten::linalg_lu_solve`、`aten::lu_unpack` |
| 18 | `torch.lu_unpack` | 正向：`aten::lu_unpack` |
| 19 | `torch.mvlgamma` | 正向：`aten::mvlgamma`、`aten::lgamma_`；反向：`aten::digamma_` |
| 20 | `torch.nextafter` | 正向：`aten::nextafter` |
| 21 | `torch.nn.GRU` | **`aten::gru.input`（正向）**, **`aten::gru.data`（正向）**, **`aten::_thnn_fused_gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell_backward`（反向）**, **`aten::_thnn_differentiable_gru_cell_backward`（反向）** |
| 22 | `torch.nn.GRUCell` | **`aten::gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell`（正向）**, **`aten::_thnn_fused_gru_cell_backward`（反向）**, **`aten::_thnn_differentiable_gru_cell_backward`（反向）** |
| 23 | `torch.nn.HuberLoss` | **`aten::huber_loss`（正向）**, **`aten::huber_loss_backward`（反向）** |
| 24 | `torch.nn.LSTM` | **`aten::lstm.input`**、**`aten::_cudnn_rnn`**、**`aten::_cudnn_rnn_backward`** |
| 25 | `torch.nn.LSTMCell` | **`aten::lstm_cell`**、**`aten::_thnn_fused_lstm_cell`**、**`aten::_thnn_differentiable_lstm_cell_backward`**、**`aten::_thnn_fused_lstm_cell_backward`** |
| 26 | `torch.nn.MultiLabelMarginLoss` | **`aten::multilabel_margin_loss_backward`** |
| 27 | `torch.nn.MultiMarginLoss` | **`aten::multi_margin_loss`**、**`aten::multi_margin_loss_backward`** |
| 28 | `torch.nn.functional.huber_loss` | **`aten::huber_loss`（正向）、`aten::huber_loss_backward`（反向）** |
| 29 | `torch.nn.functional.multi_margin_loss` | **`aten::multi_margin_loss`（正向）、`aten::multi_margin_loss_backward`（反向）** |
| 30 | `torch.nn.functional.multilabel_margin_loss` | **`aten::multilabel_margin_loss_backward`（反向）** |
| 31 | `torch.orgqr` | **`aten::orgqr`（正向）、`aten::linalg_householder_product`（正反向）、`aten::linalg_householder_product.out`（正向）** |
| 32 | `torch.ormqr` | 正向 **`aten::ormqr`（否）**，反向 **`aten::ormqr`（否）** |
| 33 | `torch.pdist` | 反向 **`aten::_pdist_backward`（否）** |
| 34 | `torch.poisson` | 正向 **`aten::poisson`（否）** |
| 35 | `torch.polygamma` | 正向 **`aten::polygamma`（否）**，反向 **`aten::polygamma`（否）** |
| 36 | `torch.slogdet` | 正向 **`aten::linalg_lu_factor_ex.out`（否）**，反向 **`aten::linalg_lu_solve`（否）**、**`aten::linalg_solve`（否）** |
| 37 | `torch.special.bessel_j0` | 正向 **`aten::special_bessel_j0`（否）** |
| 38 | `torch.special.bessel_j1` | 正向 **`aten::special_bessel_j1`（否）** |
| 39 | `torch.special.bessel_y0` | 正向 **`aten::special_bessel_y0`（否）** |
| 40 | `torch.special.bessel_y1` | 正向 **`aten::special_bessel_y1`（否）** |
| 41 | `torch.special.i0` | 正向 **`aten::special_i0`（否）**、**`aten::i0`（否）**，反向 **`aten::special_i1`（否）** |
| 42 | `torch.special.i1` | 正向 **`aten::special_i1`（否）**，反向 **`aten::i0`（否）** |
| 43 | `torch.special.i1e` | 正向 **`aten::special_i1e`（否）**，反向 **`aten::special_i0e`（否）** |
| 44 | `torch.special.zeta` | 正向 **`aten::special_zeta`（否）**、**`aten::special_zeta.self_scalar`（否）**，反向 **`aten::special_zeta`（否）** |
| 45 | `torch.tril_indices` | 正向 **`aten::tril_indices`（否）** |
| 46 | `torch.triu_indices` | 正向 **`aten::triu_indices`（否）** |

## 分类统计

### 基础数学运算

**支持率：27/43（62.8%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.absolute` | ✅ |
| `torch.addcdiv` | ✅ |
| `torch.addcmul` | ✅ |
| `torch.angle` | ✅ |
| `torch.clip` | ✅ |
| `torch.copysign` | ❌ |
| `torch.cov` | ✅ |
| `torch.deg2rad` | ✅ |
| `torch.digamma` | ❌ |
| `torch.dist` | ✅ |
| `torch.fmax` | ❌ |
| `torch.fmod` | ✅ |
| `torch.gcd` | ✅ |
| `torch.ge` | ✅ |
| `torch.heaviside` | ❌ |
| `torch.hypot` | ❌ |
| `torch.i0` | ❌ |
| `torch.igamma` | ❌ |
| `torch.igammac` | ❌ |
| `torch.lcm` | ❌ |
| `torch.ldexp` | ✅ |
| `torch.logcumsumexp` | ❌ |
| `torch.logit` | ✅ |
| `torch.logspace` | ✅ |
| `torch.multiply` | ✅ |
| `torch.mvlgamma` | ❌ |
| `torch.nanmean` | ✅ |
| `torch.nanmedian` | ✅ |
| `torch.nextafter` | ❌ |
| `torch.pdist` | ❌ |
| `torch.poisson` | ❌ |
| `torch.polygamma` | ❌ |
| `torch.positive` | ✅ |
| `torch.rad2deg` | ✅ |
| `torch.range` | ✅ |
| `torch.renorm` | ✅ |
| `torch.sgn` | ✅ |
| `torch.signbit` | ✅ |
| `torch.subtract` | ✅ |
| `torch.tanhshrink` | ✅ |
| `torch.trapezoid` | ✅ |
| `torch.true_divide` | ✅ |
| `torch.vander` | ✅ |

### 线性代数

**支持率：7/18（38.9%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.addr` | ✅ |
| `torch.cholesky` | ✅ |
| `torch.cholesky_solve` | ❌ |
| `torch.geqrf` | ❌ |
| `torch.ger` | ✅ |
| `torch.inner` | ✅ |
| `torch.kron` | ✅ |
| `torch.linalg.eigvals` | ❌ |
| `torch.linalg.pinv` | ❌ |
| `torch.linalg.vecdot` | ✅ |
| `torch.logdet` | ❌ |
| `torch.lu_solve` | ❌ |
| `torch.lu_unpack` | ❌ |
| `torch.orgqr` | ❌ |
| `torch.ormqr` | ❌ |
| `torch.slogdet` | ❌ |
| `torch.svd` | ✅ |
| `torch.tensordot` | ✅ |

### 损失函数 (nn.functional / nn.modules)

**支持率：16/22（72.7%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.nn.CTCLoss` | ✅ |
| `torch.nn.GaussianNLLLoss` | ✅ |
| `torch.nn.HingeEmbeddingLoss` | ✅ |
| `torch.nn.HuberLoss` | ❌ |
| `torch.nn.MarginRankingLoss` | ✅ |
| `torch.nn.MultiLabelMarginLoss` | ❌ |
| `torch.nn.MultiLabelSoftMarginLoss` | ✅ |
| `torch.nn.MultiMarginLoss` | ❌ |
| `torch.nn.PoissonNLLLoss` | ✅ |
| `torch.nn.SoftMarginLoss` | ✅ |
| `torch.nn.TripletMarginLoss` | ✅ |
| `torch.nn.functional.cosine_similarity` | ✅ |
| `torch.nn.functional.ctc_loss` | ✅ |
| `torch.nn.functional.gaussian_nll_loss` | ✅ |
| `torch.nn.functional.hinge_embedding_loss` | ✅ |
| `torch.nn.functional.huber_loss` | ❌ |
| `torch.nn.functional.margin_ranking_loss` | ✅ |
| `torch.nn.functional.multi_margin_loss` | ❌ |
| `torch.nn.functional.multilabel_margin_loss` | ❌ |
| `torch.nn.functional.multilabel_soft_margin_loss` | ✅ |
| `torch.nn.functional.soft_margin_loss` | ✅ |
| `torch.nn.functional.triplet_margin_loss` | ✅ |

### 神经网络层 (nn.modules)

**支持率：56/60（93.3%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.nn.AdaptiveMaxPool3d` | ✅ |
| `torch.nn.AvgPool1d` | ✅ |
| `torch.nn.CELU` | ✅ |
| `torch.nn.ChannelShuffle` | ✅ |
| `torch.nn.ConvTranspose1d` | ✅ |
| `torch.nn.ConvTranspose3d` | ✅ |
| `torch.nn.Dropout1d` | ✅ |
| `torch.nn.Dropout2d` | ✅ |
| `torch.nn.Dropout3d` | ✅ |
| `torch.nn.GRU` | ❌ |
| `torch.nn.GRUCell` | ❌ |
| `torch.nn.HardTanh` | ✅ |
| `torch.nn.InstanceNorm1d` | ✅ |
| `torch.nn.InstanceNorm2d` | ✅ |
| `torch.nn.InstanceNorm3d` | ✅ |
| `torch.nn.LPPool1d` | ✅ |
| `torch.nn.LPPool2d` | ✅ |
| `torch.nn.LSTM` | ❌ |
| `torch.nn.LSTMCell` | ❌ |
| `torch.nn.LeakyReLU` | ✅ |
| `torch.nn.MaxPool1d` | ✅ |
| `torch.nn.MaxPool2d` | ✅ |
| `torch.nn.MaxPool3d` | ✅ |
| `torch.nn.MaxUnpool1d` | ✅ |
| `torch.nn.MaxUnpool3d` | ✅ |
| `torch.nn.MultiheadAttention` | ✅ |
| `torch.nn.PixelUnshuffle` | ✅ |
| `torch.nn.RNN` | ✅ |
| `torch.nn.RNNCell` | ✅ |
| `torch.nn.RReLU` | ✅ |
| `torch.nn.Softmax2d` | ✅ |
| `torch.nn.Softmin` | ✅ |
| `torch.nn.Softsign` | ✅ |
| `torch.nn.Tanhshrink` | ✅ |
| `torch.nn.Transformer` | ✅ |
| `torch.nn.TransformerDecoder` | ✅ |
| `torch.nn.TransformerDecoderLayer` | ✅ |
| `torch.nn.TransformerEncoder` | ✅ |
| `torch.nn.TransformerEncoderLayer` | ✅ |
| `torch.nn.functional.affine_grid` | ✅ |
| `torch.nn.functional.celu` | ✅ |
| `torch.nn.functional.dropout1d` | ✅ |
| `torch.nn.functional.dropout2d` | ✅ |
| `torch.nn.functional.dropout3d` | ✅ |
| `torch.nn.functional.gumbel_softmax` | ✅ |
| `torch.nn.functional.hardtanh` | ✅ |
| `torch.nn.functional.lp_pool1d` | ✅ |
| `torch.nn.functional.lp_pool2d` | ✅ |
| `torch.nn.functional.max_pool3d` | ✅ |
| `torch.nn.functional.max_unpool1d` | ✅ |
| `torch.nn.functional.max_unpool3d` | ✅ |
| `torch.nn.functional.pixel_unshuffle` | ✅ |
| `torch.nn.functional.rms_norm` | ✅ |
| `torch.nn.functional.rrelu` | ✅ |
| `torch.nn.functional.softmin` | ✅ |
| `torch.nn.functional.softsign` | ✅ |
| `torch.nn.functional.upsample` | ✅ |
| `torch.nn.modules.ChannelShuffle` | ✅ |
| `torch.nn.modules.flatten.Flatten` | ✅ |
| `torch.nn.modules.flatten.Unflatten` | ✅ |

### 特殊函数 (torch.special.*)

**支持率：0/8（0.0%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.special.bessel_j0` | ❌ |
| `torch.special.bessel_j1` | ❌ |
| `torch.special.bessel_y0` | ❌ |
| `torch.special.bessel_y1` | ❌ |
| `torch.special.i0` | ❌ |
| `torch.special.i1` | ❌ |
| `torch.special.i1e` | ❌ |
| `torch.special.zeta` | ❌ |

### 优化器 (torch.optim.*)

**支持率：7/7（100.0%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.optim.ASGD` | ✅ |
| `torch.optim.Adamax` | ✅ |
| `torch.optim.RMSprop` | ✅ |
| `torch.optim.Rprop` | ✅ |
| `torch.optim.adadelta.Adadelta` | ✅ |
| `torch.optim.adagrad.Adagrad` | ✅ |
| `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` | ✅ |

### 分布 (torch.distribution.*)

**支持率：2/3（66.7%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.distribution.gamma.Gamma` | ❌ |
| `torch.distribution.laplace.Laplace` | ✅ |
| `torch.distribution.uniform.Uniform` | ✅ |

### 张量操作与变换

**支持率：48/50（96.0%）**

| API | A5 支持 |
|-----|:-------:|
| `torch.adjoint` | ✅ |
| `torch.aminmax` | ✅ |
| `torch.argwhere` | ✅ |
| `torch.atleast_1d` | ✅ |
| `torch.atleast_2d` | ✅ |
| `torch.atleast_3d` | ✅ |
| `torch.bartlett_window` | ✅ |
| `torch.bitwise_left_shift` | ✅ |
| `torch.bitwise_right_shift` | ✅ |
| `torch.blackman_window` | ✅ |
| `torch.block_diag` | ✅ |
| `torch.bucketize` | ✅ |
| `torch.column_stack` | ✅ |
| `torch.combinations` | ✅ |
| `torch.cond` | ✅ |
| `torch.conj` | ✅ |
| `torch.diag_embed` | ✅ |
| `torch.diagflat` | ✅ |
| `torch.diagonal` | ✅ |
| `torch.diagonal_scatter` | ✅ |
| `torch.dsplit` | ✅ |
| `torch.dstack` | ✅ |
| `torch.fliplr` | ✅ |
| `torch.flipud` | ✅ |
| `torch.gather` | ✅ |
| `torch.hamming_window` | ✅ |
| `torch.hann_window` | ✅ |
| `torch.hsplit` | ✅ |
| `torch.hstack` | ✅ |
| `torch.is_complex` | ✅ |
| `torch.is_floating_point` | ✅ |
| `torch.is_nonzero` | ✅ |
| `torch.isnan` | ✅ |
| `torch.isposinf` | ✅ |
| `torch.isreal` | ✅ |
| `torch.kaiser_window` | ✅ |
| `torch.moveaxis` | ✅ |
| `torch.movedim` | ✅ |
| `torch.msort` | ✅ |
| `torch.numel` | ✅ |
| `torch.rot90` | ✅ |
| `torch.row_stack` | ✅ |
| `torch.select_scatter` | ✅ |
| `torch.swapdims` | ✅ |
| `torch.tensor_split` | ✅ |
| `torch.tril_indices` | ❌ |
| `torch.triu_indices` | ❌ |
| `torch.view_as_real` | ✅ |
| `torch.vsplit` | ✅ |
| `torch.vstack` | ✅ |

## 分析总结

在全部 211 个 torch API 中，**163 个（77.3%）可在 A5（昇腾 NPU）上通过 aclnn 算子全链路支持**，**48 个（22.7%）因部分正向或反向 ATen 算子尚未接入 aclnn 而暂不可支持**。

各类别支持情况如下：

| 类别 | 可支持 | 总数 | 支持率 |
|------|:------:|:----:|:------:|
| 基础数学运算 | 27 | 43 | 62.8% |
| 线性代数 | 7 | 18 | 38.9% |
| 损失函数 (nn.functional / nn.modules) | 16 | 22 | 72.7% |
| 神经网络层 (nn.modules) | 56 | 60 | 93.3% |
| 特殊函数 (torch.special.*) | 0 | 8 | 0.0% |
| 优化器 (torch.optim.*) | 7 | 7 | 100.0% |
| 分布 (torch.distribution.*) | 2 | 3 | 66.7% |
| 张量操作与变换 | 48 | 50 | 96.0% |

**关键发现：**

1. **优化器（torch.optim.*）** 支持率最高，达到 100%（7/7），所有优化器均已完整接入 aclnn。
2. **张量操作与变换** 支持率较高（48/50），除 `tril_indices` / `triu_indices` 外其余接口均已覆盖。
3. **特殊函数（torch.special.*）** 支持率最低（0/8），全部 Bessel、修正 Bessel 及 zeta 函数均缺少对应 aclnn 算子。
4. **线性代数** 支持率仍偏低（7/18），主要受 `aten::linalg_solve`、`aten::linalg_eig` / `aten::linalg_eigh`、`aten::linalg_lu_factor_ex.out`、`aten::lu_unpack` 等核心算子未接入的制约。
5. **神经网络层** 中，LSTM/GRU 系列因依赖 cuDNN fused 算子（`aten::_cudnn_rnn`、`aten::_thnn_fused_*`）而不可支持，其余层均已覆盖。
6. 本轮复查已将 `aten::slice_backward`、`aten::rad2deg`、`aten::linalg_vecdot`、`aten::tensordot`、`aten::vander` 等 composite/PTA 场景从阻塞项中剔除；剩余阻塞主要集中在线代核心 kernel、RNN fused kernel 与 `torch.special.*` 专用算子。
