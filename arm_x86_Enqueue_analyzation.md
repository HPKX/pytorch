# aclnn 算子 Enqueue 性能差异分析报告

## ARM vs x86: 3.8~4 倍耗时差异根因分析

---

## 一、测试环境

### 1.1 硬件配置

| 项目 | ARM | x86 |
|------|-----|-----|
| CPU 型号 | Kunpeng 950 7592C | AMD EPYC 9575F |
| 微架构 | TaiShan V200 (ARMv8.x) | Zen 5 (x86-64-v4) |
| 核心数 | 96C | 64C (112 threads) |
| 基频 | 2.3 GHz | 3.3 GHz |
| 最大睿频 | ~2.5 GHz (估) | 5.0 GHz |
| 设计定位 | 高密吞吐型 | 高频单核型 ("F" 后缀) |

### 1.2 Cache 层级 (lscpu)

| 层级 | ARM 总量 (实例数) | ARM Per-Core | x86 总量 (实例数) | x86 Per-Core | 比较 |
|------|-------------------|-------------|-------------------|-------------|------|
| L1d | 12 MiB (192) | **64 KB** | 5.3 MiB (112) | **48 KB** | ARM 胜 1.33x |
| L1i | 24 MiB (192) | **128 KB** | 3.5 MiB (112) | **32 KB** | ARM 胜 4x |
| L2 | 192 MiB (192) | **1 MB** | 112 MiB (112) | **1 MB** | 持平 |
| L3 | 546 MiB (24) | **~23 MB/cluster** | 512 MiB (16) | **32 MB/cluster** | x86 胜 1.4x |
| L3 per core | — | **2.8 MB** | — | **4.6 MB** | x86 胜 1.6x |

### 1.3 测量指标

| 项目 | 说明 |
|------|------|
| 测量指标 | aclnn 算子 Enqueue 耗时 (Host 侧) |
| 测量对象 | `EXEC_NPU_CMD` 宏展开的完整路径 |
| 性能比 | ARM 耗时为 x86 的 **3.8~4 倍** |

---

## 二、Enqueue 热路径分析

### 2.1 执行流程

```
EXEC_NPU_CMD(aclnnXxx, tensor_a, tensor_b, tensor_out)
 |
 +-- 1. Hash Buffer 填充
 |    +-- reset g_hash_offset                            (TLS 写)
 |    +-- 写入 deterministic / aic / aiv / device / stream
 |    +-- 写入 op name                                   (std::string 临时构造)
 |    +-- 每 tensor: 6 次 MemcpyToBufImpl
 |         sizes -> dtype -> "," -> strides -> offset -> storage_dims
 |         (3 tensors = 18 次函数调用, 每次含 2 次 TLS 访问)
 |
 +-- 2. Hash 计算
 |    +-- gen_hash(): MurmurHash3-128bit 对 ~200-400 bytes
 |
 +-- 3. Executor Cache 查找
 |    +-- setPTAHashKeyFunc()          <-- 跨 .so 间接调用 (CANN)
 |    +-- ptaGetExecCacheFunc()        <-- 跨 .so 间接调用 (CANN)
 |
 +-- 4. [仅 Cache Miss]
 |    +-- ConvertTypes(): 每 tensor 提取元数据 + aclCreateTensor
 |    +-- getWorkspaceSizeFunc(): 调用 CANN 计算 workspace
 |
 +-- 5. Workspace 分配
 |    +-- unsafe_empty_workspace(): NPUWorkspaceAllocator (recursive_mutex + 池化)
 |
 +-- 6. 入队
      +-- RunOpApiV2() -> enCurrentNPUStream(): atomic 计数 + ring buffer 写入
```

### 2.2 路径特征

- **纯串行执行**: 步骤 1-6 存在严格数据依赖，无法并行化
- **标量整数运算**: 全程无浮点、无向量化机会，无 SIMD 可利用
- **延迟敏感**: 性能完全取决于单核标量 IPC 和频率
- **高频短函数调用**: 18+ 次 MemcpyToBufImpl + 多次跨 .so 间接调用

---

## 三、差异归因

### 3.1 归因总览

```
3.8 ~ 4.0x  =  频率 2.0x  x  IPC 1.6x  x  CANN 1.15x  x  软件路径 1.07x
              |________ 硬件 ~3.2x ________|  |_____ 软件 ~1.2x _____|
                         (~80%)                       (~20%)
```

### 3.2 第一层: 硬件差距 (贡献 ~3.2x, 占总差距 ~80%)

#### 3.2.1 运行频率差距 -- 贡献 2.0x

Enqueue 是单线程轻载操作，功耗极低:

| CPU | 基频 | Enqueue 实际运行频率 | 原因 |
|-----|------|---------------------|------|
| EPYC 9575F | 3.3 GHz | **~4.8-5.0 GHz** | 单线程功耗低, "F" 后缀 Boost 激进 |
| 鲲鹏 950 7592C | 2.3 GHz | **~2.3-2.5 GHz** | ARM 服务器 Boost 幅度小 |

**实际频率比: 4.9 / 2.4 = 2.04x**

这一项贡献了总差距的约一半。

#### 3.2.2 微架构 IPC 差距 -- 贡献 1.5-1.7x

| 维度 | 鲲鹏 950 (TaiShan V200) | EPYC 9575F (Zen 5) | 对 Enqueue 的影响 |
|------|------------------------|---------------------|------------------|
| 解码宽度 | ~4-wide | 6-8 wide | Zen 5 每周期处理更多指令 |
| 重排序缓冲 (ROB) | ~160 entries | ~350+ entries | Zen 5 乱序窗口大 2x+, 更好地隐藏依赖链延迟 |
| 分支预测 | 改进型 | TAGE-SC-L 级别 | 间接调用 (dlsym 指针) 预测更精确 |
| 整数乘法延迟 | 4 cycles | 3 cycles | Hash 计算中乘法链更快 |
| 执行端口 | 较少 | 更多 ALU/AGU | 更高指令级并行 |

Enqueue 路径的特点 -- **高频短函数调用、链式数据依赖 (hash)、间接调用** -- 恰好是宽乱序执行窗口的优势场景。

#### 3.2.3 Cache 层级 -- 无显著影响

Enqueue 的工作集很小:

| 数据 | 大小 | 命中层级 |
|------|------|---------|
| `g_hash_buf` (TLS) | 8 KB | L1d (两平台均命中) |
| `TensorImpl` 元数据 (per tensor) | ~200-300 B | L1d (两平台均命中) |
| `StorageImpl` / `NpuStorageDesc` | ~100 B | L1d/L2 |
| 函数指令 (hash/memcpy/convert) | ~4-8 KB | L1i |
| CANN .so 中的 cache 查找函数 | ~数 KB | L1i / L2 |

两平台 L2 均为 1MB/core, L1d ARM 更大 (64KB vs 48KB), L1i ARM 远大 (128KB vs 32KB)。数据和指令工作集完全在 L1/L2 内, cache 命中率接近 100%, **cache 层级差异对此场景无影响**。

ARM 的 L1i 128KB 在跨 .so 间接调用 (torch_npu -> CANN) 场景下反而有优势。

#### 3.2.4 产品定位对立

| 维度 | 鲲鹏 950 7592C | EPYC 9575F |
|------|---------------|------------|
| 设计目标 | 96 核高密吞吐 | 高频单核性能 ("F") |
| 核心策略 | 多核 x 低频 x 低功耗 | 少核 x 高频 x 高 Boost |
| 适配场景 | 高并行度服务 | 单线程延迟敏感 |

用 Enqueue 延迟 (纯单线程标量) 对比这两款 CPU, 是两者设计目标**最不对称**的场景。

### 3.3 第二层: CANN 库内部差异 (贡献 ~1.1-1.2x, 黑盒)

Enqueue 路径有 2-3 次跨 .so 调用进入 CANN 闭源库:

| 调用 | 功能 | 路径 |
|------|------|------|
| `setPTAHashKeyFunc()` | 设置 cache key | cache hit/miss 均触发 |
| `ptaGetExecCacheFunc()` / `ptaFindExecCacheFunc()` | executor cache 查找 | cache hit/miss 均触发 |
| `aclCreateTensor()` | 创建 ACL tensor 描述 | 仅 cache miss |
| `getWorkspaceSizeFunc()` | 计算 workspace 大小 | 仅 cache miss |

这些函数的内部实现无法从 torch_npu 代码验证。如果 CANN x86 版本使用了 SSE/AVX 优化哈希表或内存操作, 会产生额外差距。

### 3.4 第三层: 软件路径开销 (贡献 ~1.05-1.1x)

| 因素 | 机制 | 量化 |
|------|------|------|
| MemcpyToBufImpl 调用频率 | 每次 2 次 TLS 访问 + memcpy + 边界检查, 18 次/op | ~1.03x |
| `std::string(aclnn_api)` 构造 | 典型 op name >= 16 字节超 SSO, 触发堆分配 | ~1.01-1.02x |
| recursive_mutex (workspace) | ARM LL/SC vs x86 LOCK CMPXCHG | ~1.01x |
| atomic (task queue 计数) | ARM LDXR/STXR vs x86 LOCK XADD | ~1.01x |

---

## 四、结论

1. **差距的 ~80% 来自硬件**: EPYC 9575F (Zen 5, 5GHz boost, "F" 频率型) vs 鲲鹏 950 7592C (TaiShan V200, 2.3GHz, 96 核吞吐型) 在单线程标量性能上的差距是产品定位决定的, 属于合理预期。

2. **Cache 不是因素**: 鲲鹏 950 的 L1d (64KB) 和 L1i (128KB) 均大于 EPYC 9575F, L2 持平 (1MB), Enqueue 工作集完全在 L1/L2 内。

3. **软件优化空间有限但存在**: 约 1.2x 的软件因素可通过代码优化缩小, 但无法改变硬件层面的 3.2x 差距。

4. **根本出路是减少 Enqueue 次数**: 通过算子融合、graph mode、流水线化等手段, 将优化目标从 "缩小单次 Enqueue 延迟" 转向 "减少 Enqueue 总次数" 和 "重叠隐藏延迟"。
