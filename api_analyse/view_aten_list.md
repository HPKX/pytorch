# PyTorch View 类 ATen 接口全集

> 来源: `tools/autograd/gen_inplace_or_view_type.py` 中的 `VIEW_FUNCTIONS`、`VIEW_FUNCTIONS_WITH_METADATA_CHANGE`、`RETURNS_VIEWS_OF_INPUT`、`ALL_VIEW_FUNCTIONS` 以及 `native_functions.yaml` 中 `tags: inplace_view` 和 `tags: view_copy` 标记的接口。

View 操作返回与输入共享底层存储的张量（不拷贝数据），由 `ADInplaceOrView` dispatch key 追踪。

---

## 一、VIEW_FUNCTIONS（根 view 操作 —— 始终返回 view）

这些操作在 `ADInplaceOrView` kernel 中被注册为 view，通过 `as_view()` 建立 autograd view 关系。

| # | 接口名 | schema | dispatch 策略 | 备注 |
|---|--------|--------|--------------|------|
| 1 | `aten::numpy_T` | `numpy_T(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | `.T` 属性 |
| 2 | `aten::alias` | `alias(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd; NestedTensor 特化 | core |
| 3 | `aten::as_strided` | `as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)` | CPU/CUDA/MTIA/MPS/Meta; Quantized 特化 | core, 最底层 view 原语 |
| 4 | `aten::diagonal` | `diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)` | CompositeExplicitAutograd | core |
| 5 | `aten::diagonal.Dimname` | `diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)` | CompositeImplicitAutograd | 命名维度重载 |
| 6 | `aten::expand` | `expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)` | CompositeExplicitAutograd | core |
| 7 | `aten::permute` | `permute(Tensor(a) self, int[] dims) -> Tensor(a)` | CompositeExplicitAutograd; MPS/Sparse 特化 | core |
| 8 | `aten::select.int` | `select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)` | CompositeExplicitAutograd; SparseCsr/NestedTensor 特化 | core |
| 9 | `aten::select.Dimname` | `select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)` | CompositeImplicitAutograd | 命名维度重载 |
| 10 | `aten::slice.Tensor` | `slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)` | CompositeExplicitAutograd | core |
| 11 | `aten::slice_inverse` | `slice_inverse(Tensor(a) self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)` | CompositeExplicitAutograd | |
| 12 | `aten::split.Tensor` | `split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]` | CompositeExplicitAutograd | 返回 Tensor[] |
| 13 | `aten::split.sizes` | `split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]` | CompositeImplicitAutograd | 返回 Tensor[] |
| 14 | `aten::split_with_sizes` | `split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]` | CompositeExplicitAutograd; NestedTensor 特化 | core, 返回 Tensor[] |
| 15 | `aten::squeeze` | `squeeze(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd; Quantized/NestedTensor 特化 | |
| 16 | `aten::squeeze.dim` | `squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)` | CompositeExplicitAutograd; Quantized/NestedTensor 特化 | core |
| 17 | `aten::squeeze.dimname` | `squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)` | CompositeImplicitAutograd | 命名维度重载 |
| 18 | `aten::squeeze.dims` | `squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)` | CompositeExplicitAutograd; Quantized/NestedTensor 特化 | core |
| 19 | `aten::t` | `t(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd | |
| 20 | `aten::transpose.int` | `transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)` | CompositeExplicitAutograd; NestedTensor 特化 | |
| 21 | `aten::transpose.Dimname` | `transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)` | CompositeImplicitAutograd | 命名维度重载 |
| 22 | `aten::unfold` | `unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)` | CPU/CUDA/Meta/MPS/MTIA; Quantized 特化 | method only |
| 23 | `aten::unsqueeze` | `unsqueeze(Tensor(a) self, int dim) -> Tensor(a)` | CompositeExplicitAutograd; Sparse/Quantized/NestedTensor 特化 | core |
| 24 | `aten::flatten.using_ints` | `flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)` | CompositeImplicitAutograd | |
| 25 | `aten::flatten.named_out_dim` | `flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)` | CompositeImplicitAutograd | |
| 26 | `aten::flatten.using_names` | `flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)` | CompositeImplicitAutograd | |
| 27 | `aten::flatten.DimnameList` | `flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)` | CompositeImplicitAutograd | |
| 28 | `aten::view` | `view(Tensor(a) self, SymInt[] size) -> Tensor(a)` | ZeroTensor/Meta/CPU/CUDA/Quantized/MPS/MTIA; Mkldnn/NestedTensor 特化 | core |
| 29 | `aten::view.dtype` | `view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)` | CompositeExplicitAutograd | dtype 重解释 |
| 30 | `aten::unbind.int` | `unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]` | CompositeExplicitAutograd; NestedTensor 特化 | 返回 Tensor[] |
| 31 | `aten::unbind.Dimname` | `unbind.Dimname(Tensor(a -> *) self, Dimname dim) -> Tensor(a)[]` | CompositeImplicitAutograd | 返回 Tensor[] |
| 32 | `aten::_indices` | `_indices(Tensor(a) self) -> Tensor(a)` | SparseCPU/SparseCUDA/SparseMPS/SparseMeta | sparse 内部 |
| 33 | `aten::_values` | `_values(Tensor(a) self) -> Tensor(a)` | SparseCPU/SparseCUDA/SparseMPS/SparseMeta | sparse 内部 |
| 34 | `aten::indices` | `indices(Tensor(a) self) -> Tensor(a)` | Sparse 特化; CompositeExplicitAutograd fallback | sparse 公开接口 |
| 35 | `aten::values` | `values(Tensor(a) self) -> Tensor(a)` | Sparse/SparseCsr/NestedTensor 特化; CompositeExplicitAutograd fallback | 多布局 |
| 36 | `aten::crow_indices` | `crow_indices(Tensor(a) self) -> Tensor(a)` | SparseCsr 特化; CompositeExplicitAutograd fallback | CSR 格式 |
| 37 | `aten::col_indices` | `col_indices(Tensor(a) self) -> Tensor(a)` | SparseCsr 特化; CompositeExplicitAutograd fallback | CSR 格式 |
| 38 | `aten::ccol_indices` | `ccol_indices(Tensor(a) self) -> Tensor(a)` | SparseCsr 特化; CompositeExplicitAutograd fallback | CSC 格式 |
| 39 | `aten::row_indices` | `row_indices(Tensor(a) self) -> Tensor(a)` | SparseCsr 特化; CompositeExplicitAutograd fallback | CSC 格式 |
| 40 | `aten::_sparse_coo_tensor_with_dims_and_tensors` | `_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? is_coalesced=None) -> Tensor` | SparseCPU/SparseCUDA/SparseMeta/SparseMPS/Meta | view of values |
| 41 | `aten::_reshape_alias` | `_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)` | CPU/CUDA/Meta/Quantized/ZeroTensor/MPS/MTIA | 内部 reshape view 路径 |
| 42 | `aten::_test_autograd_multiple_dispatch_view` | `_test_autograd_multiple_dispatch_view(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd | 测试用 |

---

## 二、VIEW_FUNCTIONS_WITH_METADATA_CHANGE（带元数据变更的 view）

这些 view 操作修改了张量的元数据（如 dtype、conjugate/negation 位、nested 结构），需要特殊的 `ViewFunc` replay 逻辑。

| # | 接口名 | schema | dispatch 策略 | 备注 |
|---|--------|--------|--------------|------|
| 1 | `aten::view_as_complex` | `view_as_complex(Tensor(a) self) -> Tensor(a)` | CPU/CUDA/MPS/Meta; Sparse 特化 | float->complex |
| 2 | `aten::view_as_real` | `view_as_real(Tensor(a) self) -> Tensor(a)` | CPU/CUDA/MPS/Meta; Sparse 特化 | complex->float |
| 3 | `aten::_conj` | `_conj(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd | 共轭位 view |
| 4 | `aten::_neg_view` | `_neg_view(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd | 取负位 view |
| 5 | `aten::_nested_get_values` | `_nested_get_values(Tensor(a) self) -> Tensor(a)` | dispatch: {} (fallback) | nested tensor 内部 |
| 6 | `aten::_nested_view_from_buffer` | `_nested_view_from_buffer(Tensor(a) self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor(a)` | CPU/CUDA | nested tensor 内部 |
| 7 | `aten::_nested_view_from_jagged` | `_nested_view_from_jagged(Tensor(a) self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None) -> Tensor(a)` | dispatch: {} (fallback) | jagged nested tensor |

---

## 三、RETURNS_VIEWS_OF_INPUT（复合 view —— 可能返回 view 也可能返回 copy）

这些操作是根 view 操作的组合，**可能**返回 view（取决于输入布局/连续性等），也可能返回拷贝。JIT 将它们视为潜在 view。

| # | 接口名 | schema | dispatch 策略 | 备注 |
|---|--------|--------|--------------|------|
| 1 | `aten::chunk` | `chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]` | CompositeImplicitAutograd; NestedTensor 特化 | 基于 split |
| 2 | `aten::detach` | `detach(Tensor(a) self) -> Tensor(a)` | CompositeExplicitAutograd; NestedTensor 特化 | 断开 autograd |
| 3 | `aten::contiguous` | `contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)` | CompositeImplicitAutograd | 已连续时返回 self |
| 4 | `aten::reshape` | `reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)` | CompositeImplicitAutograd; NestedTensor 特化 | 连续时为 view |
| 5 | `aten::reshape_as` | `reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)` | CompositeImplicitAutograd; NestedTensor 特化 | 基于 reshape |
| 6 | `aten::expand_as` | `expand_as(Tensor(a) self, Tensor other) -> Tensor(a)` | CompositeImplicitAutograd | 基于 expand |
| 7 | `aten::view_as` | `view_as(Tensor(a) self, Tensor other) -> Tensor(a)` | CompositeImplicitAutograd | 基于 view |
| 8 | `aten::real` | `real(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | 复数实部 view |
| 9 | `aten::imag` | `imag(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | 复数虚部 view |
| 10 | `aten::narrow` | `narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)` | CompositeImplicitAutograd; NestedTensor 特化 | 基于 slice |
| 11 | `aten::narrow.Tensor` | `narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)` | CompositeImplicitAutograd | start 为 Tensor |
| 12 | `aten::movedim.intlist` | `movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)` | CompositeImplicitAutograd | 基于 permute |
| 13 | `aten::movedim.int` | `movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)` | CompositeImplicitAutograd | 单维度版 |
| 14 | `aten::tensor_split.sections` | `tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]` | CompositeImplicitAutograd | 基于 slice |
| 15 | `aten::tensor_split.indices` | `tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]` | CompositeImplicitAutograd | 基于 slice |
| 16 | `aten::tensor_split.tensor_indices_or_sections` | `tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]` | CompositeImplicitAutograd | |
| 17 | `aten::swapdims` | `swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)` | CompositeImplicitAutograd | = transpose |
| 18 | `aten::swapaxes` | `swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)` | CompositeImplicitAutograd | = transpose |
| 19 | `aten::mT` | `mT(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | 最后两维转置 |
| 20 | `aten::mH` | `mH(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | 共轭转置 |
| 21 | `aten::adjoint` | `adjoint(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | = mH |
| 22 | `aten::matrix_H` | `matrix_H(Tensor(a) self) -> Tensor(a)` | CompositeImplicitAutograd | = mH |

---

## 四、_unsafe_view（特殊 view）

不参与 `ADInplaceOrView` 追踪，但在 `gen_variable_type` 中被视为 view（用于 StorageImpl/TensorImpl 验证）。

| # | 接口名 | schema | dispatch 策略 | 备注 |
|---|--------|--------|--------------|------|
| 1 | `aten::_unsafe_view` | `_unsafe_view(Tensor self, SymInt[] size) -> Tensor` | CompositeExplicitAutograd | 返回类型是 `Tensor` 而非 `Tensor(a)`，不传播别名标注 |

---

## 五、inplace_view 操作（tags: inplace_view）

原地修改张量的 shape/stride/storage 等元信息，不修改数据本身。由 `ADInplaceOrView` dispatch key 拦截以维护 autograd 版本计数。

| # | 接口名 | schema | dispatch 策略 | 备注 |
|---|--------|--------|--------------|------|
| 1 | `aten::as_strided_` | `as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)` | CompositeExplicitAutogradNonFunctional | |
| 2 | `aten::detach_` | `detach_(Tensor(a!) self) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 3 | `aten::rename_` | `rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)` | CompositeImplicitAutograd | 命名维度 |
| 4 | `aten::resize_` | `resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)` | Meta/CPU/CUDA/MPS; Quantized/SparseCsr 特化 | core |
| 5 | `aten::resize_as_` | `resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 6 | `aten::squeeze_` | `squeeze_(Tensor(a!) self) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 7 | `aten::squeeze_.dim` | `squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 8 | `aten::squeeze_.dims` | `squeeze_.dims(Tensor(a!) self, int[] dim) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 9 | `aten::squeeze_.dimname` | `squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)` | CompositeImplicitAutograd | |
| 10 | `aten::t_` | `t_(Tensor(a!) self) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 11 | `aten::transpose_` | `transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 12 | `aten::unsqueeze_` | `unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)` | CompositeExplicitAutograd | |
| 13 | `aten::set_` | `set_(Tensor(a!) self) -> Tensor(a!)` | CPU/CUDA/Meta/MPS | 清空 |
| 14 | `aten::set_.source_Storage` | `set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)` | CPU/CUDA/Meta/MPS | |
| 15 | `aten::set_.source_Storage_storage_offset` | `set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)` | CPU/Meta/CUDA/MPS; Quantized 特化 | |
| 16 | `aten::set_.source_Tensor` | `set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)` | CPU/CUDA/Meta/MPS | |
| 17 | `aten::set_.source_Tensor_storage_offset` | `set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)` | CompositeImplicitAutograd | |
| 18 | `aten::swapaxes_` | `swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)` | CompositeImplicitAutograd | = transpose_ |
| 19 | `aten::swapdims_` | `swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)` | CompositeImplicitAutograd | = transpose_ |

---

## 六、view_copy 操作（tags: view_copy）

对应 view 操作的拷贝版本，用于 functionalization（将 view/inplace 转换为 pure functional 语义）。返回新 Tensor 而非共享存储。所有 view_copy 操作均 dispatch 到 `CompositeExplicitAutogradNonFunctional`。

| # | 接口名 | schema | 对应的 view 操作 |
|---|--------|--------|-----------------|
| 1 | `aten::alias_copy` | `alias_copy(Tensor self) -> Tensor` | `aten::alias` |
| 2 | `aten::as_strided_copy` | `as_strided_copy(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor` | `aten::as_strided` |
| 3 | `aten::diagonal_copy` | `diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor` | `aten::diagonal` |
| 4 | `aten::expand_copy` | `expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor` | `aten::expand` |
| 5 | `aten::permute_copy` | `permute_copy(Tensor self, int[] dims) -> Tensor` | `aten::permute` |
| 6 | `aten::select_copy.int` | `select_copy.int(Tensor self, int dim, SymInt index) -> Tensor` | `aten::select.int` |
| 7 | `aten::detach_copy` | `detach_copy(Tensor self) -> Tensor` | `aten::detach` |
| 8 | `aten::slice_copy.Tensor` | `slice_copy.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor` | `aten::slice.Tensor` |
| 9 | `aten::split_copy.Tensor` | `split_copy.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]` | `aten::split.Tensor` |
| 10 | `aten::split_with_sizes_copy` | `split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]` | `aten::split_with_sizes` |
| 11 | `aten::squeeze_copy` | `squeeze_copy(Tensor self) -> Tensor` | `aten::squeeze` |
| 12 | `aten::squeeze_copy.dim` | `squeeze_copy.dim(Tensor self, int dim) -> Tensor` | `aten::squeeze.dim` |
| 13 | `aten::squeeze_copy.dims` | `squeeze_copy.dims(Tensor self, int[] dim) -> Tensor` | `aten::squeeze.dims` |
| 14 | `aten::t_copy` | `t_copy(Tensor self) -> Tensor` | `aten::t` |
| 15 | `aten::transpose_copy.int` | `transpose_copy.int(Tensor self, int dim0, int dim1) -> Tensor` | `aten::transpose.int` |
| 16 | `aten::unsqueeze_copy` | `unsqueeze_copy(Tensor self, int dim) -> Tensor` | `aten::unsqueeze` |
| 17 | `aten::_indices_copy` | `_indices_copy(Tensor self) -> Tensor` | `aten::_indices` |
| 18 | `aten::_values_copy` | `_values_copy(Tensor self) -> Tensor` | `aten::_values` |
| 19 | `aten::indices_copy` | `indices_copy(Tensor self) -> Tensor` | `aten::indices` |
| 20 | `aten::values_copy` | `values_copy(Tensor self) -> Tensor` | `aten::values` |
| 21 | `aten::crow_indices_copy` | `crow_indices_copy(Tensor self) -> Tensor` | `aten::crow_indices` |
| 22 | `aten::col_indices_copy` | `col_indices_copy(Tensor self) -> Tensor` | `aten::col_indices` |
| 23 | `aten::ccol_indices_copy` | `ccol_indices_copy(Tensor self) -> Tensor` | `aten::ccol_indices` |
| 24 | `aten::row_indices_copy` | `row_indices_copy(Tensor self) -> Tensor` | `aten::row_indices` |
| 25 | `aten::unbind_copy.int` | `unbind_copy.int(Tensor self, int dim=0) -> Tensor[]` | `aten::unbind.int` |
| 26 | `aten::view_copy` | `view_copy(Tensor self, SymInt[] size) -> Tensor` | `aten::view` |
| 27 | `aten::view_copy.dtype` | `view_copy.dtype(Tensor self, ScalarType dtype) -> Tensor` | `aten::view.dtype` |
| 28 | `aten::unfold_copy` | `unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor` | `aten::unfold` |
| 29 | `aten::view_as_real_copy` | `view_as_real_copy(Tensor self) -> Tensor` | `aten::view_as_real` |
| 30 | `aten::view_as_complex_copy` | `view_as_complex_copy(Tensor self) -> Tensor` | `aten::view_as_complex` |
| 31 | `aten::_conj_copy` | `_conj_copy(Tensor self) -> Tensor` | `aten::_conj` |
| 32 | `aten::_neg_view_copy` | `_neg_view_copy(Tensor self) -> Tensor` | `aten::_neg_view` |
| 33 | `aten::_reshape_alias_copy` | `_reshape_alias_copy(Tensor self, SymInt[] size, SymInt[] stride) -> Tensor` | `aten::_reshape_alias` |
| 34 | `aten::_nested_view_from_buffer_copy` | `_nested_view_from_buffer_copy(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor` | `aten::_nested_view_from_buffer` |
| 35 | `aten::_nested_view_from_jagged_copy` | `_nested_view_from_jagged_copy(Tensor self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None) -> Tensor` | `aten::_nested_view_from_jagged` |
| 36 | `aten::_nested_get_values_copy` | `_nested_get_values_copy(Tensor self) -> Tensor` | `aten::_nested_get_values` |
| 37 | `aten::_test_autograd_multiple_dispatch_view_copy` | `_test_autograd_multiple_dispatch_view_copy(Tensor self) -> Tensor` | `aten::_test_autograd_multiple_dispatch_view` |
| 38 | `aten::_fw_primal_copy` | `_fw_primal_copy(Tensor self, int level) -> Tensor` | `aten::_fw_primal` (forward-mode AD) |
| 39 | `aten::_make_dual_copy` | `_make_dual_copy(Tensor primal, Tensor tangent, int level) -> Tensor` | `aten::_make_dual` (forward-mode AD) |
| 40 | `aten::_sparse_broadcast_to_copy` | `_sparse_broadcast_to_copy(Tensor self, int[] size) -> Tensor` | sparse broadcast |
| 41 | `aten::lift_fresh_copy` | `lift_fresh_copy(Tensor self) -> Tensor` | `aten::lift_fresh` (functionalization) |
| 42 | `aten::narrow_copy` | `narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor` | `aten::narrow` |
| 43 | `aten::slice_scatter` | `slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor` | core, scatter 版 slice |

---

## 七、统计汇总

| 类别 | 基础操作数 | 含重载的接口数 |
|------|-----------|--------------|
| VIEW_FUNCTIONS（根 view） | 31 | 42 |
| VIEW_FUNCTIONS_WITH_METADATA_CHANGE | 7 | 7 |
| RETURNS_VIEWS_OF_INPUT（复合 view） | 18 | 22 |
| _unsafe_view | 1 | 1 |
| inplace_view | 14 | 19 |
| view_copy | 38 | 43 |
| **合计（去重）** | **~85 基础操作** | **~134 接口** |

---

## 八、View 机制关键文件

| 文件 | 作用 |
|------|------|
| `tools/autograd/gen_inplace_or_view_type.py` | 定义 `VIEW_FUNCTIONS`、`RETURNS_VIEWS_OF_INPUT` 等列表，生成 `ADInplaceOrViewType` |
| `tools/autograd/gen_view_funcs.py` | 生成 `ViewFunc` 类，用于 view replay |
| `torch/csrc/autograd/variable.h` | `CreationMeta` 枚举，`AutogradMeta` 中的 view 关系追踪 |
| `torch/csrc/autograd/VariableTypeUtils.h` | `as_view()` 模板函数，建立 view 关系 |
| `torch/csrc/autograd/autograd_not_implemented_fallback.cpp` | view 操作的 fallback 逻辑 |
| `aten/src/ATen/native/native_functions.yaml` | 所有操作的声明式注册 |

---

## 九、View 判定规则

一个 ATen 操作被视为 view 操作，需满足以下任一条件：

1. **在 `VIEW_FUNCTIONS` 字典中**：最严格的 view 定义，始终共享存储，在 `ADInplaceOrView` 中通过 `as_view()` 追踪。
2. **在 `RETURNS_VIEWS_OF_INPUT` 中**：可能返回 view 的复合操作（如 `reshape` 在连续时返回 view，否则返回 copy）。
3. **在 `ALL_VIEW_FUNCTIONS` 中**：包含 `_unsafe_view`，用于 StorageImpl/TensorImpl 验证。
4. **带 `tags: inplace_view`**：原地修改元信息的操作。
5. **带 `tags: view_copy`**：view 操作的 functional 拷贝版本。

Schema 层面的识别特征：
- **返回类型 `Tensor(a)` 或 `Tensor(a)[]`**：`(a)` 标注表示输出与输入 `self` 共享别名。
- **返回类型 `Tensor(a!)`**：表示原地操作（inplace_view 均为此形式）。
- **返回类型 `Tensor`（无别名标注）**：`_unsafe_view` 和 `view_copy` 操作为此形式。
