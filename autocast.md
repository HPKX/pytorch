  PyTorch Autocast 机制完整流程图

  1. 整体架构概览

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              用户代码层 (Python)                                  │
  │  ┌──────────────────────────────────────────────────────────────────────────┐   │
  │  │  with torch.autocast(device_type="cuda", dtype=torch.float16):           │   │
  │  │      output = model(input)                                               │   │
  │  │      loss = loss_fn(output, target)                                      │   │
  │  └──────────────────────────────────────────────────────────────────────────┘   │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                         Python autocast 类 (Context Manager)                    │
  │                         torch/amp/autocast_mode.py:52                           │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │  __enter__():                                                              │ │
  │  │    1. 保存之前的状态 (prev, prev_fastdtype, prev_cache_enabled)              │ │
  │  │    2. torch.set_autocast_enabled(device, True)                             │ │
  │  │    3. torch.set_autocast_dtype(device, fast_dtype)                         │ │
  │  │    4. torch.autocast_increment_nesting()                                   │ │
  │  │    5. torch.set_autocast_cache_enabled(cache_enabled)                      │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                         C++ 绑定层 (torch/csrc/autograd/init.cpp)               │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │  set_autocast_enabled(device_type, enabled)                                │ │
  │  │       ↓                                                                     │ │
  │  │  at::autocast::set_autocast_enabled(device_type, enabled)                  │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                    C++ Autocast 核心实现 (aten/src/ATen/autocast_mode.cpp)       │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │  void set_autocast_enabled(DeviceType device_type, bool enabled) {         │ │
  │  │      DispatchKey key = get_autocast_dispatch_key_from_device_type(...);    │ │
  │  │      c10::impl::tls_set_dispatch_key_excluded(key, !enabled);              │ │
  │  │  }                                                                          │ │
  │  │                                                                             │ │
  │  │  核心机制: 通过修改 TLS (Thread-Local Storage) 中的 DispatchKey 排除集合    │ │
  │  │           来启用/禁用 Autocast 的 Dispatcher 后端                           │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────────────┘

  ---
  2. 算子调用时的 Dispatch 流程

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                               算子调用: torch.mm(a, b)                           │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                            PyTorch Dispatcher                                   │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │   检查 DispatchKey 集合:                                                    │ │
  │  │   ┌──────────────────────────────────────────────────────────────────────┐ │ │
  │  │   │  DispatchKey::Python                                                 │ │ │
  │  │   │  DispatchKey::Autocast  ← 如果未被排除，则拦截调用                       │ │ │
  │  │   │  DispatchKey::AutogradCUDA                                           │ │ │
  │  │   │  DispatchKey::CUDA                                                   │ │ │
  │  │   │  ...                                                                 │ │ │
  │  │   └──────────────────────────────────────────────────────────────────────┘ │ │
  │  │                                                                            │ │
  │  │   Autocast 启用时: Autocast Key 不在排除集合中 → 路由到 Autocast kernel        │ │
  │  │   Autocast 禁用时: Autocast Key 在排除集合中 → 跳过，继续到下一个 Key           │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                           Autocast Kernel (WrapFunction_)                       │
  │                           aten/src/ATen/autocast_mode.h:470-485                 │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │  template<CastPolicy::lower_precision_fp, ...>                             │ │
  │  │  struct WrapFunction_ {                                                    │ │
  │  │    static Ret call(Args... args) {                                         │ │
  │  │      // 1. 禁用 Autocast Key，防止无限递归                                    │ │
  │  │      ExcludeDispatchKeyGuard no_autocast(AutocastKey);                     │ │
  │  │                                                                            │ │
  │  │      // 2. 对输入进行类型转换 (cached_cast)                                   │ │
  │  │      // 3. 调用真实的算子实现                                                 │ │
  │  │      return (*F)(cached_cast(lower_precision_fp, args, device)...);        │ │
  │  │    }                                                                       │ │
  │  │  };                                                                        │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                          cached_cast 类型转换逻辑                                │
  │                          aten/src/ATen/autocast_mode.cpp:122-157                │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │  Tensor cached_cast(ScalarType to_type, const Tensor& arg, DeviceType) {   │ │
  │  │      if (!is_eligible(arg, device) || arg.scalar_type() == to_type)        │ │
  │  │          return arg;  // 不需要转换                                          │ │
  │  │                                                                             │ │
  │  │      // 缓存启发式: 缓存 fp32 模型权重 (leaves) 的低精度转换                     │ │
  │  │      bool can_try_cache =                                                   │ │
  │  │          (to_type == lower_precision_fp) &&                                 │ │
  │  │          (arg.scalar_type() == kFloat) &&                                   │ │
  │  │          arg.requires_grad() &&                                             │ │
  │  │          arg.is_leaf() &&                                                   │ │
  │  │          !arg.is_view() &&                                                  │ │
  │  │          cache_enabled &&                                                   │ │
  │  │          !InferenceMode::is_enabled();                                      │ │
  │  │                                                                             │ │
  │  │      if (can_try_cache) {                                                   │ │
  │  │          // 查找缓存或创建新的缓存条目                                          │ │
  │  │          auto it = cached_casts.find(arg.unsafeGetTensorImpl());            │ │
  │  │          if (it != cached_casts.end())                                      │ │
  │  │              return cached_tensor;                                          │ │
  │  │          else {                                                             │ │
  │  │              auto casted = arg.to(to_type);                                 │ │
  │  │              cached_casts.emplace(...);                                     │ │
  │  │              return casted;                                                 │ │
  │  │          }                                                                  │ │
  │  │      }                                                                      │ │
  │  │      return arg.to(to_type);  // 不缓存                                      │ │
  │  │  }                                                                          │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              真实算子执行                                         │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │    由于 ExcludeDispatchKeyGuard，Dispatcher 跳过 Autocast Key                │ │
  │  │    直接路由到 CUDA/CPU 等真实的后端实现                                        │ │
  │  │                                                                            │ │
  │  │    例如: at::mm (CUDA kernel, cuBLAS) 在 float16 精度下执行                   │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────────────┘

  ---
  3. Cast Policy (类型转换策略) 详解

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              CastPolicy 枚举                                     │
  │                              aten/src/ATen/autocast_mode.h:416-438              │
  └─────────────────────────────────────────────────────────────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
  │  lower_precision_fp │   │        fp32         │   │       promote       │
  │                     │   │                     │   │                     │
  │  所有输入转为低精度    │   │  所有输入转为 fp32   │   │  取输入中最宽的类型  │
  │  (fp16/bf16)        │   │  确保数值稳定性       │   │                     │
  ├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
  │  适用算子:           │   │  适用算子:           │   │  适用算子:          │
  │  • conv2d           │   │  • log, exp, pow    │   │  • cat, stack       │
  │  • mm, matmul       │   │  • softmax          │   │  • addcdiv          │
  │  • linear           │   │  • layer_norm       │   │  • index_put        │
  │  • bmm, baddbmm     │   │  • batch_norm       │   │  • scatter_add      │
  │  • einsum           │   │  • loss functions   │   │  • dot, cross       │
  │  • scaled_dot_prod  │   │  • upsample_*       │   │  • tensordot        │
  │  • lstm_cell        │   │  • cosine_sim       │   │  • grid_sampler     │
  └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
  │ WrapFunction_       │   │ WrapFunction_       │   │ WrapFunction_       │
  │ <lower_precision_fp>│   │ <fp32>              │   │ <promote>           │
  │                     │   │                     │   │                     │
  │ cached_cast(        │   │ cached_cast(        │   │ to_type = promote_  │
  │   get_lower_prec_fp,│   │   kFloat,           │   │   type(args...);    │
  │   args,             │   │   args,             │   │ cached_cast(        │
  │   device);          │   │   device);          │   │   to_type, args);   │
  └─────────────────────┘   └─────────────────────┘   └─────────────────────┘

  额外策略:
  ┌─────────────────────┐   ┌─────────────────────┐
  │ fp32_set_opt_dtype  │   │  fp32_append_dtype  │
  │                     │   │                     │
  │ 对于有可选 dtype     │   │ 对于无 dtype 参数     │
  │ 参数的函数:          │   │ 的重载:               │
  │ • softmax(int)      │   │ • norm(Scalar)      │
  │ • sum(dim_IntList)  │   │ • norm(ScalarOpt)   │
  │                     │   │                     │
  │ 如果用户未指定dtype   │   │ 追加 kFloat dtype    │
  │ 则设为 kFloat        │   │ 并重定向到有 dtype    │
  │                     │   │ 参数的重载            │
  └─────────────────────┘   └─────────────────────┘

  ---
  4. 多设备支持架构

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                             DispatchKey 映射                                     │
  │                             aten/src/ATen/autocast_mode.h:174-202               │
  └─────────────────────────────────────────────────────────────────────────────────┘

  Device Type          DispatchKey                  默认 dtype
  ─────────────────────────────────────────────────────────────────
  CUDA          →      DispatchKey::Autocast        float16
  CPU           →      DispatchKey::AutocastCPU     bfloat16
  XPU           →      DispatchKey::AutocastXPU     float16
  MPS           →      DispatchKey::AutocastMPS     float16
  MTIA          →      DispatchKey::AutocastMTIA    float16
  MAIA          →      DispatchKey::AutocastMAIA    float16
  HPU           →      DispatchKey::AutocastHPU     bfloat16
  XLA           →      DispatchKey::AutocastXLA     bfloat16
  IPU           →      DispatchKey::AutocastIPU     float16
  PrivateUse1   →      DispatchKey::AutocastPU1     float16

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                        每个设备独立注册 Autocast Kernels                         │
  │                        aten/src/ATen/autocast_mode.cpp:177-593                  │
  │  ┌────────────────────────────────────────────────────────────────────────────┐ │
  │  │  TORCH_LIBRARY_IMPL(_, Autocast, m) {         // CUDA fallthrough          │ │
  │  │      m.fallback(torch::CppFunction::makeFallthrough());                    │ │
  │  │  }                                                                         │ │
  │  │  TORCH_LIBRARY_IMPL(aten, Autocast, m) {      // CUDA op 注册               │ │
  │  │      AT_FORALL_LOWER_PRECISION_FP(_KERNEL_CUDA_LOW_PRECISION_FP)            │ │
  │  │      AT_FORALL_FP32(_KERNEL_CUDA_FP32)                                      │ │
  │  │      AT_FORALL_PROMOTE(_KERNEL_CUDA_PROMOTE)                                │ │
  │  │  }                                                                          │ │
  │  │                                                                             │ │
  │  │  TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) { ... }  // CPU                   │ │
  │  │  TORCH_LIBRARY_IMPL(aten, AutocastMPS, m) { ... }  // MPS                   │ │
  │  │  TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) { ... }  // XPU                   │ │
  │  │  ...                                                                        │ │
  │  └────────────────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────────────┘

  ---
  5. 缓存机制详解

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              权重缓存机制                                        │
  │                              aten/src/ATen/autocast_mode.cpp:21-51              │
  └─────────────────────────────────────────────────────────────────────────────────┘

  目的: 避免每次 forward 都对相同的模型权重进行 fp32→fp16 转换

  缓存结构:
  ┌────────────────────────────────────────────────────────────────┐
  │  ska::flat_hash_map<TensorImpl*, val_type> cached_casts        │
  │                                                                │
  │  Key:   TensorImpl*  (原始 fp32 tensor 的内存地址)               │
  │  Value: tuple<weakref, Tensor>                                 │
  │         ├─ weakref:    指向原始 tensor 的弱引用                  │
  │         │              (防止地址被回收后误命中)                   │
  │         └─ Tensor:     转换后的 fp16/bf16 tensor                │
  └────────────────────────────────────────────────────────────────┘

  缓存条件 (必须全部满足):
  ┌────────────────────────────────────────────────────────────────┐
  │  1. to_type == lower_precision_fp (fp16/bf16)                  │
  │  2. arg.scalar_type() == kFloat                                │
  │  3. arg.requires_grad() == true                                │
  │  4. arg.is_leaf() == true                                      │
  │  5. arg.is_view() == false                                     │
  │  6. cache_enabled == true                                      │
  │  7. !InferenceMode::is_enabled()                               │
  └────────────────────────────────────────────────────────────────┘

  缓存生命周期:
  ┌────────────────────────────────────────────────────────────────┐
  │  1. autocast __enter__: 进入 autocast 区域                      │
  │  2. autocast_increment_nesting(): nesting++                   │
  │  3. forward 执行中: cached_cast 进行缓存查找/创建                 │
  │  4. autocast __exit__: 退出 autocast 区域                       │
  │  5. autocast_decrement_nesting(): nesting--                    │
  │  6. if (nesting == 0): clear_cache()  // 清空缓存               │
  └────────────────────────────────────────────────────────────────┘

  ---
  6. 嵌套 Autocast 处理

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                           嵌套 Autocast 场景                                     │
  └─────────────────────────────────────────────────────────────────────────────────┘

  with torch.autocast("cuda", dtype=torch.float16):  # 外层: fp16
      # nesting = 1, dtype = fp16
      output1 = model_part1(x)                        # 运行在 fp16

      with torch.autocast("cuda", enabled=False):    # 内层: 禁用
          # nesting = 2, enabled = False
          output2 = model_part2(output1.float())      # 运行在 fp32

      with torch.autocast("cuda", dtype=torch.bfloat16):  # 内层: bf16
          # nesting = 2, dtype = bf16
          output3 = model_part3(output2)              # 运行在 bf16

      # 恢复到外层状态: nesting = 1, dtype = fp16
      output4 = model_part4(output3)                  # 运行在 fp16

  # nesting = 0, 清空缓存

  状态保存与恢复:
  ┌────────────────────────────────────────────────────────────────┐
  │  __enter__:                                                    │
  │    self.prev = torch.is_autocast_enabled(device)              │
  │    self.prev_fastdtype = torch.get_autocast_dtype(device)     │
  │    self.prev_cache_enabled = torch.is_autocast_cache_enabled()│
  │    → 设置新状态                                                │
  │                                                                │
  │  __exit__:                                                     │
  │    → 恢复之前保存的状态                                          │
  │    if decrement_nesting() == 0: clear_cache()                 │
  └────────────────────────────────────────────────────────────────┘

  ---
  7. JIT/TorchScript 中的 Autocast

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                        JIT Autocast Pass (编译期处理)                            │
  │                        torch/csrc/jit/passes/autocast.cpp                       │
  └─────────────────────────────────────────────────────────────────────────────────┘

  JIT 使用完全不同的机制: 在图编译阶段插入显式的类型转换节点

  handleBlock(node) {
      switch (node->kind()) {
          case aten::conv2d:
          case aten::matmul:
          case aten::linear:
              // 插入: aten::_autocast_to_reduced_precision
              castTensorInputs(node, aten::_autocast_to_reduced_precision, ...);
              break;

          case aten::log:
          case aten::exp:
          case aten::softmax:
              // 插入: aten::_autocast_to_full_precision
              castTensorInputs(node, aten::_autocast_to_full_precision, ...);
              break;
      }
  }

  原始图:
  ┌────────────────────────┐
  │  %x = aten::conv2d(...)│
  │  %y = aten::log(...)   │
  └────────────────────────┘
             │
             ▼
  转换后的图:
  ┌─────────────────────────────────────────────────────────────────┐
  │  %input_fp16 = aten::_autocast_to_reduced_precision(%input)    │
  │  %x = aten::conv2d(%input_fp16, ...)                            │
  │  %log_input_fp32 = aten::_autocast_to_full_precision(%y)       │
  │  %z = aten::log(%log_input_fp32, ...)                           │
  └─────────────────────────────────────────────────────────────────┘

  ---
  8. 完整的数据流总结

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                           Autocast 完整数据流                                    │
  └─────────────────────────────────────────────────────────────────────────────────┘

  用户代码                Python 层               C++ 绑定            C++ 核心
  ────────               ────────               ────────            ────────

  with autocast() ──────► __enter__() ──────────► set_autocast_   ──► tls_set_dispatch_
                          │                       enabled()           key_excluded()
                          │                       set_autocast_
                          │                       dtype()
                          │                       increment_nesting()
                          ▼
  model(input) ─────────────────────────────────────────────────────► Dispatcher
                                                                          │
                                                              ┌───────────┴───────────┐
                                                              │ Autocast Key 未排除?  │
                                                              └───────────┬───────────┘
                                                                         YES
                                                                          ▼
                                                              ┌─────────────────────┐
                                                              │  Autocast Kernel    │
                                                              │  (WrapFunction_)    │
                                                              └──────────┬──────────┘
                                                                         │
                                                              ┌──────────▼──────────┐
                                                              │ 1. ExcludeKeyGuard  │
                                                              │ 2. cached_cast()    │
                                                              │ 3. 调用真实算子     │
                                                              └──────────┬──────────┘
                                                                         │
                                                              ┌──────────▼──────────┐
                                                              │   真实后端执行       │
                                                              │   (CUDA/CPU/etc)    │
                                                              └─────────────────────┘

  exit autocast ────────► __exit__() ───────────► decrement_nesting()
                                                  if (nesting==0):
                                                      clear_cache()
                                                  恢复之前状态

  ---
  这个流程图完整展示了 PyTorch Autocast 机制从用户调用到底层执行的全过程，包括:

  1. Python Context Manager 管理状态进入/退出
  2. TLS (Thread-Local Storage) 存储每线程的 autocast 状态
  3. Dispatcher 根据 DispatchKey 路由算子调用
  4. WrapFunction_ 模板 实现不同 CastPolicy 的类型转换逻辑
  5. cached_cast 实现权重缓存优化
  6. 嵌套支持 通过状态保存/恢复和 nesting 计数实现
  7. 多设备支持 通过不同的 DispatchKey 和 TORCH_LIBRARY_IMPL 注册实现


这两个 CastPolicy 是为了处理有 dtype 参数的函数的特殊情况。让我详细解释：

  fp32_set_opt_dtype

  场景：函数有一个 std::optional<ScalarType> dtype 参数

  策略：如果用户没有指定 dtype，就设置为 kFloat；如果用户已指定，则尊重用户的选择

  例子：softmax

  # 函数签名
  # torch.softmax(input, dim, dtype=None)

  with torch.autocast("cuda"):
      x = torch.randn(10, 10, device="cuda")  # fp32 输入

      # 情况1: 用户没有指定 dtype
      y = torch.softmax(x, dim=0)
      # Autocast 会设置 dtype=torch.float32
      # 相当于: torch.softmax(x, dim=0, dtype=torch.float32)
      # 输出是 fp32，确保数值稳定性

      # 情况2: 用户显式指定了 dtype
      y = torch.softmax(x, dim=0, dtype=torch.float16)
      # Autocast 不会修改，尊重用户选择
      # 输出是 fp16

  C++ 实现逻辑 (autocast_mode.h:522-533):
  static Ret call(Args... args) {
      ExcludeDispatchKeyGuard no_autocast(...);
      if (firstarg_is_eligible(device_type, args...)) {
          // 对每个参数调用 set_opt_dtype
          // 如果是 optional<ScalarType> 且为空，设为 kFloat
          return (*F)(set_opt_dtype(at::kFloat, args)...);
      } else {
          return (*F)(args...);  // 不符合条件，原样调用
      }
  }

  ---
  fp32_append_dtype

  场景：函数有多个重载，有些接受 dtype 参数，有些不接受

  策略：包装「不接受 dtype」的重载，追加一个 kFloat 参数，然后重定向到「接受 dtype」的重载

  例子：norm

  # torch.norm 有多个重载:
  # norm(input, p)                              -> 没有 dtype 参数
  # norm(input, p, dtype)                       -> 有 dtype 参数
  # norm(input, p, dim, keepdim)                -> 没有 dtype 参数  
  # norm(input, p, dim, keepdim, dtype)         -> 有 dtype 参数

  with torch.autocast("cuda"):
      x = torch.randn(10, 10, device="cuda", dtype=torch.float16)

      # 用户调用没有 dtype 的版本
      y = torch.norm(x, p=2)

      # Autocast 内部会转换为:
      # torch.norm(x, p=2, dtype=torch.float32)
      # 
      # 即: 追加 dtype=float32，重定向到有 dtype 参数的重载

  注册方式 (autocast_mode.cpp 和 autocast_mode.h:933-958):

  // 这个宏定义了如何重定向
  #define AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(_)                         \
    _(ADD_NS(norm),                                                           \
      "norm.Scalar",                          /* 注册名 */                     \
      Tensor(const Tensor&, const Scalar&),   /* 原始签名 (无dtype) */         \
      Tensor(const Tensor&, const std::optional<Scalar>&, ScalarType), /* 目标签名 */ \
      fp32_append_dtype)

  C++ 实现逻辑 (autocast_mode.h:550-557):
  static Ret call(Args... args) {
      ExcludeDispatchKeyGuard no_autocast(...);
      // 从第一个参数推断输出类型
      at::ScalarType out_type = type_from_firstarg(device_type, at::kFloat, args...);
      // 追加 dtype 参数，调用另一个重载
      return (*F)(args..., out_type);  // 注意这里多了 out_type
  }

  ---
  对比总结
  ┌────────────────────┬─────────────────────────────┬─────────────────────────────────────┐
  │       Policy       │            场景             │                行为                 │
  ├────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ fp32_set_opt_dtype │ 函数有 optional<dtype> 参数 │ 如果 dtype 是空的，设为 fp32        │
  ├────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
  │ fp32_append_dtype  │ 函数有/无 dtype 的多个重载  │ 追加 fp32 参数，调用有 dtype 的重载 │
  └────────────────────┴─────────────────────────────┴─────────────────────────────────────┘
  为什么需要这些策略？

  像 softmax、norm、sum 这类归约/数值敏感操作，在低精度下容易溢出或精度丢失。Autocast 强制它们在 fp32 下执行，但又要兼容用户显式指定 dtype 的情况，所以需要这两种特殊处理策略。


反向节点如何记录正向 autocast 策略

  这是一个非常深刻的问题。答案分为两个层次：

  核心洞察：反向传播通常不需要显式记录 autocast 策略

  原因是 autocast 并不改变计算图本身，它只改变前向计算时输入张量的 dtype。反向传播的 dtype 由前向输出的实际 dtype 自然决定。

  正向: fp32 权重 → autocast(cached_cast) → fp16 输入 → mm kernel → fp16 输出
  反向: fp16 grad_output → mm_backward kernel → fp16 grad_input

  反向 kernel 直接根据 grad_output 和保存的张量的 dtype 来工作，无需知道 autocast 策略。

  具体机制分为三层：

  ---
  1️⃣ 张量 dtype 自然传递（大部分情况）

  正向时：
  ┌─────────────────────────────────────────────────────────┐
  │  mm(a_fp32, b_fp32)                                     │
  │       ↓ autocast enabled                                │
  │  WrapFunction_<lower_precision_fp>                      │
  │       ↓ cached_cast                                     │
  │  mm(a_fp16, b_fp16) → output_fp16                       │
  │       ↓                                                 │
  │  SavedVariable 保存 a_fp16, b_fp16 (已经是转换后的 dtype) │
  └─────────────────────────────────────────────────────────┘

  反向时：
  ┌─────────────────────────────────────────────────────────┐
  │  grad_output: fp16 (与 output 同 dtype)                  │
  │  saved_a: fp16, saved_b: fp16                           │
  │       ↓                                                 │
  │  mm_backward(grad_output_fp16, saved_a_fp16)            │
  │       ↓                                                 │
  │  grad_input: fp16                                       │
  └─────────────────────────────────────────────────────────┘

  关键点：SavedVariable 保存的是 autocast 转换后 的张量，所以反向 kernel 自然在正确的 dtype 下执行。

  ---
  2️⃣ ThreadLocalState 保存 DispatchKey 排除集（autograd engine）

  当 backward() 调用时，GraphTask 会捕获并恢复 TLS 状态：

  // aten/src/ATen/ThreadLocalState.h:86-88
  std::array<at::ScalarType, at::COMPILE_TIME_MAX_DEVICE_TYPES>
      autocast_dtypes_{};  // 保存每个设备的 autocast dtype

  // ThreadLocalState.cpp:25-27 (构造时捕获)
  for(size_t i=0; i<autocast_dtypes_.size(); i++) {
     autocast_dtypes_[i] = at::autocast::get_autocast_dtype(static_cast<at::DeviceType>(i));
  }

  // engine.cpp:557 (执行反向节点前恢复)
  at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);

  但这里有个微妙点：ThreadLocalState 保存的是 autocast_dtypes_，但 autocast 是否启用 是通过 LocalDispatchKeySet.excluded_ 来控制的（dispatch_key_ 成员）。

  ---
  3️⃣ 自定义 autograd 函数需要显式处理

  对于 torch.autograd.Function，需要手动使用 @custom_fwd 和 @custom_bwd：

  # torch/amp/autocast_mode.py:476-494
  def decorate_fwd(*args, **kwargs):
      args[0]._dtype = torch.get_autocast_dtype(device_type)      # 保存 dtype
      args[0]._fwd_used_autocast = torch.is_autocast_enabled(device_type)  # 保存是否启用
      return fwd(*args, **kwargs)

  # :520-527
  def decorate_bwd(*args, **kwargs):
      with autocast(
          device_type=device_type,
          enabled=args[0]._fwd_used_autocast,  # 读取保存的状态
          dtype=args[0]._dtype,
      ):
          return bwd(*args, **kwargs)

  这是通过在 ctx 上保存 _dtype 和 _fwd_used_autocast 两个属性实现的。

  ---
  4️⃣ Checkpointing 的特殊处理

  torch.utils.checkpoint 需要在反向时重算前向，所以必须显式保存 autocast 状态：

  # torch/utils/checkpoint.py
  class CheckpointFunction(torch.autograd.Function):
      @staticmethod
      def forward(ctx, ...):
          ctx.device_autocast_kwargs = _get_autocast_kwargs(device)  # 保存
          ...

      @staticmethod
      def backward(ctx, *args):
          with torch.amp.autocast(**ctx.device_autocast_kwargs):  # 恢复
              outputs = ctx.run_function(*ctx.inputs)  # 重算前向

  ---
  总结：为什么"不需要显式记录"却又"有机制记录"？
  ┌─────────────────┬────────────────────────────────────┬───────────────────────────────────────────────────────────────┐
  │      场景       │                机制                │                            为什么                             │
  ├─────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ 原生算子        │ dtype 自然传递                     │ SavedVariable 保存转换后的张量，反向 kernel 按实际 dtype 执行 │
  ├─────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ Engine 级别     │ ThreadLocalState                   │ 保存 dispatch key 排除集和 autocast_dtypes，用于恢复 TLS 环境 │
  ├─────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ 自定义 Function │ ctx._dtype, ctx._fwd_used_autocast │ 需要手动用 @custom_fwd/@custom_bwd 装饰器                     │
  ├─────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ Checkpointing   │ ctx.device_autocast_kwargs         │ 重算前向需要完整恢复 autocast 上下文                          │
  └─────────────────┴────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
  核心理解：Autocast 的本质是在调用 kernel 前对输入做 cast，一旦 cast 完成，后续计算（包括反向）就在该 dtype 下进行。反向传播不是在"相同的 autocast 策略下"运行，而是在"相同 dtype 的张量上"运行——这是一个微妙但重要的区别。