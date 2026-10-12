# ans48

实时全带降噪库：48 kHz / 44.1 kHz 单声道，C 接口，Android（arm64-v8a、armeabi-v7a）预编译库，模型编在库里。

两档：

| 档位 | 用途 | 窗长 | 算法延迟（48 kHz / 44.1 kHz） | 强度范围 |
|---|---|---|---|---|
| `ANS48_MODE_MONITOR` | 耳返（自己听自己的声音，延迟敏感） | 256 样本 | 5.3 ms / 5.8 ms | 0 ~ 2 |
| `ANS48_MODE_STREAM` | 录音 / 推流（给别人听的成品） | 1024 样本 | 21.3 ms / 23.2 ms | 0 ~ 1 |

强度 = 掩码指数 γ：输出谱 = 输入谱 × 掩码^γ。0 = 原样直通（只有延迟），1 = 模型全档，>1 比全档压得更狠。耳返档允许到 2，因为演唱者对自己声音里的轻微损伤基本听不出来；推流档上限 1，成品要给听众听。

---

## 目录

| 路径 | 内容 |
|---|---|
| `include/ans48.h` | 对外接口（C） |
| `src/` | 实现：`ans48.cpp`（接口）、`ulunas_kernel.*`（网络的手写流式推理）、`denoise_stage.h`（分帧 / FFT / 掩码 / 重叠相加）、`models.cpp`（内嵌权重，自动生成） |
| `models/` | 权重原件（`kernel_weights.bin` + `kernel_manifest.json`），`tools/embed_models.py` 把它们编进 `src/models.cpp` |
| `prebuilt/arm64-v8a/libans48.so`、`prebuilt/armeabi-v7a/libans48.so` | 预编译库（NDK r28，minSdk 21，c++_static） |
| `test/ans48_check.cpp`、`test/ans48_robust.cpp`、`test/ans48_wrap.cpp`、`test/ans48_bench.cpp`、`test/vec/` | 自检、接口健壮性测试、计数器回绕测试、端上测速、测试向量 |
| `tools/build_android.sh` | 编两个 ABI 的库与手机上跑的测试程序 |

---

## 快速开始

### 用预编译库

把 `include/ans48.h` 与 `prebuilt/<ABI>/libans48.so` 加进工程（例如放到 `app/src/main/jniLibs/<ABI>/`），在自己的 native 代码里 `#include "ans48.h"` 并链接 `libans48.so`。库只依赖系统 libc / libm，C++ 运行时静态链接在库内。

### 从源码编译

```sh
ANDROID_NDK=/path/to/ndk tools/build_android.sh        # 出 prebuilt/<ABI>/libans48.so 与 build-android-<ABI>/ 下的测试程序
cmake -S . -B build-host && cmake --build build-host  # 主机版（自检用）
```

### 最小调用

```c
#include "ans48.h"

Ans48* dn = ans48_create(48000, ANS48_MODE_MONITOR);   // 失败返回 NULL
ans48_set_strength(dn, 1.0f);                           // 可在任意线程调，下一帧生效

// 音频回调里：任意长度，in 与 out 可以同址
ans48_process(dn, in, out, frames);                     // float，±1 满幅
// 或 ans48_process_s16(dn, in16, out16, frames);

ans48_destroy(dn);
```

---

## 接口

| 函数 | 说明 |
|---|---|
| `Ans48* ans48_create(int sample_rate, int mode)` | 建实例。`sample_rate` 只支持 48000、44100；`mode` 为 `ANS48_MODE_MONITOR` 或 `ANS48_MODE_STREAM`。失败（参数不支持、内存不足）返回 NULL，不会结束宿主进程。初始强度 1。 |
| `int ans48_process(Ans48*, const float* in, float* out, int n)` | 处理 n 个样本，任意长度，`out` 可以与 `in` 同址。返回 0；参数无效返回 −1。 |
| `int ans48_process_s16(Ans48*, const short* in, short* out, int n)` | int16 版，与 float 版输出一致（量化到 int16）。 |
| `float ans48_set_strength(Ans48*, float gamma)` | 设强度，截到 `[0, ans48_max_strength]`，NaN 忽略；返回实际生效的值。可在任意线程调用。 |
| `float ans48_get_strength(const Ans48*)` / `float ans48_max_strength(const Ans48*)` | 当前强度 / 该档上限（耳返 2、推流 1）。 |
| `int ans48_latency_samples(const Ans48*)` | 输出相对输入的固定延迟（样本数）= 窗长。 |
| `void ans48_reset(Ans48*)` | 清空内部状态（换歌、断流后重新开始）；强度不变。不要与 `process` 并发调用。 |
| `void ans48_stats(const Ans48*, double out3[3])` | 最近一秒每帧平均处理耗时 ms、最大 ms、已处理样本数。一帧 = 一个 hop（窗长的一半）。 |
| `void ans48_destroy(Ans48*)` / `const char* ans48_version(void)` | 销毁 / 版本与模型标识。 |

### 延迟

输出比输入晚 `ans48_latency_samples()` 个样本（= 窗长：耳返 256、推流 1024）。这是算法延迟，不含设备与应用的音频缓冲。开头这一段是启动段，输出接近 0（重叠相加起步时的舍入余量，不超过 −60 dBFS）。强度 0 时输出就是输入延迟这么多样本（自检里有这一项）。

### 线程

- 一个实例同一时刻只能在一个线程里 `process`。不同实例互不影响，可以在不同线程同时用。
- `ans48_set_strength` 可以在任意线程调用（原子变量，`process` 每次开始时读一次）。
- `process` / `process_s16` 不分配内存、不加锁、不做文件读写，可以直接在音频回调里调用。

---

## 接入要点

- 采样率要与实际流一致；只支持 48000 与 44100，其他采样率请在外部重采样后再送进来（或换用支持的采样率采集）。
- 每次调用的块长任意，内部按 hop（耳返 128、推流 512 样本）凑帧；块长等于 hop 的整数倍时每次调用的耗时最均匀。
- 耳返档每 128 样本（48 kHz 下 2.7 ms）算一帧，**只能放在大 / 中核上跑**（小核算不过来，见性能表）；回调块长小于 128 时，有的回调不算帧、有的回调算一帧，单次回调的耗时会不均匀。推流档每 512 样本算一帧，小核也跑得下。
- 降噪放在回声消除之后（对回声消除的输出降噪）。两者都会引入延迟，总延迟相加。

---

## 自检

```sh
cd test/vec
../../build-host/ans48_check 00d2fda6_in48k.pcm 48000 monitor 1 00d2fda6_monitor_48k_g1.pcm
../../build-host/ans48_check 00d2fda6_in48k.pcm 48000 monitor 2 00d2fda6_monitor_48k_g2.pcm
../../build-host/ans48_check 00d2fda6_in441.pcm 44100 monitor 1 00d2fda6_monitor_441_g1.pcm
../../build-host/ans48_check 00d2fda6_in441.pcm 44100 monitor 2 00d2fda6_monitor_441_g2.pcm
../../build-host/ans48_check 00d2fda6_in48k.pcm 48000 stream 1 00d2fda6_stream_48k_g1.pcm
../../build-host/ans48_check 00d2fda6_in441.pcm 44100 stream 1 00d2fda6_stream_441_g1.pcm
```

期望输出由训练端的流式参照实现处理同一段录音得到。通过标准：库的输出按延迟挪齐后，与期望输出的差值能量低 ≥ 50 dB（实测 −97 ~ −103 dB，最大差 1 LSB，即 int16 取整的量级）。

`test/ans48_robust` 检查接口边界（主机与手机都能跑，全部通过返回 0）：不支持的参数返回 NULL / −1；强度 0 时输出 = 输入延迟窗长；块长 1 / 7 / 96 / 383 / 480 / 1000 / 4096 / 整段与原地处理逐样本相同；int16 与 float 一致；强度越大输出能量不增、超范围截断、NaN 忽略；reset 后与新实例相同；处理期不分配内存；反复创建 / 销毁不漏内存；多线程并发创建与处理；处理中从另一线程改强度。

```sh
./build-host/ans48_robust test/vec/00d2fda6_in48k.pcm test/vec/00d2fda6_in441.pcm
./build-host/ans48_wrap test/vec/00d2fda6_in48k.pcm     # 输出缓冲计数器跨过 2^32 / 2^64 回绕点时输出不中断
```

网络推理另有逐层核对（权重导出时同时录下参照网络每一块在两帧输入下的输出，手写实现逐块比较；主机与手机、arm64 与 armv7 均通过，最大差 1e-5 量级）。该核对依赖训练端代码，不在本仓库。

---

## 性能

### 测法

`test/ans48_bench`：按真实节奏送数据（每次一个 hop：耳返 128 样本 = 2.67 ms，推流 512 样本 = 10.7 ms，48 kHz），用 `taskset` 把进程钉在指定核心上跑 30 s，统计处理线程的 CPU 时间 / 墙钟时间（= 该核心被占用的比例）、每块耗时与超时（单块超过一个块周期）次数；另以全速（不睡眠）跑 10 s 量实时率。按真实节奏时核心会降频，所以占用率比全速实时率高，前者才反映实际运行时的占用。

```sh
adb shell 'cd /data/local/tmp/ans48 && LD_LIBRARY_PATH=. taskset <核心掩码> ./ans48_bench 00d2fda6_in48k.pcm 48000 <monitor|stream> 30 1'
```

### 结果（0.1，48 kHz，强度 1）

| 设备（SoC） | 核心 | 档位 | ABI | 核心占用 | 每块耗时 平均 / p99 / 最大（ms） | 超时块 | 全速实时率 |
|---|---|---|---|---|---|---|---|
| Galaxy S10 SM-G9730（骁龙 855） | 中核 Kryo 485 Gold 2.42 GHz（A76 级） | 耳返 | arm64-v8a | 65.9% | 1.76 / 1.94 / 3.7 | 48 / 11250 | 0.198 |
| | | | armeabi-v7a | 69.1% | 1.85 / 2.14 / 6.0 | 31 / 11250 | 0.267 |
| | | 推流 | arm64-v8a | 20.3% | 2.15 / 2.39 / 4.3 | 0 | 0.055 |
| | | | armeabi-v7a | 26.6% | 2.83 / 3.07 / 5.7 | 0 | 0.074 |
| | 小核 Kryo 485 Silver 1.79 GHz（A55 级） | 耳返 | arm64-v8a | 84.9% | 2.29 / 2.72 / 8.5 | 144 / 11250 | 0.855 |
| | | | armeabi-v7a | 98.4% | 2.87 / 3.33 / 6.7 | 全部 | 1.072 |
| | | 推流 | arm64-v8a | 58.8% | 6.47 / 9.45 / 15.9 | 13 / 2812 | 0.238 |
| | | | armeabi-v7a | 68.0% | 7.43 / 9.53 / 15.0 | 17 / 2812 | 0.302 |
| vivo V1934A（Helio P65） | 大核 Cortex-A75 2.0 GHz | 耳返 | arm64-v8a | 73.6% | 1.95 / 2.14 / 5.2 | 5 / 11250 | 0.391 |
| | | | armeabi-v7a | 73.8% | 1.95 / 2.37 / 12.3 | 5 / 11250 | 0.465 |
| | | 推流 | arm64-v8a | 23.8% | 2.52 / 2.66 / 3.3 | 0 | 0.129 |
| | | | armeabi-v7a | 29.3% | 3.10 / 3.38 / 14.1 | 1 / 2812 | 0.156 |
| | 小核 Cortex-A55 1.7 GHz | 耳返 | arm64-v8a | 99.7% | 2.92 / 5.43 / 11.2 | 5428 / 11250 | 1.340 |
| | | | armeabi-v7a | 99.8% | 3.51 / 6.73 / 8.9 | 全部 | 1.559 |
| | | 推流 | arm64-v8a | 57.0% | 6.07 / 8.18 / 10.1 | 0 | 0.615 |
| | | | armeabi-v7a | 67.2% | 7.14 / 9.10 / 11.3 | 2 / 2812 | 0.690 |

- **推流档**任何核心都跑得下：大 / 中核约 20–30%，小核约 57–68%。
- **耳返档必须放在大 / 中核上**：每 2.67 ms 就要算一帧，小核算不过来（全速实时率接近或超过 1）。在大 / 中核上全速实时率 0.20–0.47，但按真实节奏时核心被调频压到低频，占用升到 66–74%、偶有超时。实际接入时建议：处理线程固定在大 / 中核（Android 的 `APerformanceHint` / 提高线程优先级），或在两次处理之间让线程保持忙碌，让调频器维持高频；块与块之间留至少一块的缓冲。
- 64 位比 32 位省 10–25%（全速实时率），优先用 arm64-v8a。
- 内存：每个实例常驻约 2.0–2.5 MB（含一份权重拷贝）；库文件 arm64 2.8 MB、armv7 2.6 MB（权重编在库内）。
- Galaxy S10 息屏进入打盹状态时，系统不允许把进程钉到中 / 大核（`taskset` 报 Invalid argument），测中核时须点亮屏幕。

---

## 版本与模型

- 当前：`ans48 0.1.1 (monitor ulunas-w256 e50, stream ulunas-w1024 e73)`。
- 0.1.1 修改：输出缓冲的读写计数器改为 64 位、可读判断改为"读写位置不相等"。0.1 在 32 位（armeabi-v7a）上计数器约 24.9 小时（48 kHz）回绕，之后持续输出 0 而接口照常返回成功。新增 `test/ans48_wrap`：把计数器设到 2^32 / 2^64 回绕点前处理，输出与从 0 开始逐样本相同。
- 网络为 UL-UNAS 结构的降噪掩码模型（约 15.4 万 / 20.3 万参数，耳返 / 推流），以 44.1 kHz 训练；48 kHz 下按 48 kHz 的频率刻度重建 ERB 投影。耳返档两个采样率的权重相同（高频段 64 个频点对 64 个 ERB 频带，投影与采样率无关）。
- 推理为手写流式实现（逐帧、显式状态、BatchNorm 折进卷积、每帧零分配），不依赖推理框架。

---

## 许可

本库源码、模型权重与预编译库不开源，见 [`LICENSE`](LICENSE)。第三方部分（UL-UNAS 网络结构、测试向量）按各自条款，见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

**集成方注意：** 分发带 `libans48.so` 的 App 时，要在 App 的开源许可 / 第三方声明页面里附上 `THIRD_PARTY_NOTICES.md` 中 UL-UNAS 的版权声明与 MIT 许可文本。

---

## 已知限制

- 只处理单声道；只支持 48 kHz 与 44.1 kHz。
- 模型以 44.1 kHz 训练，48 kHz 下按频率刻度重建 ERB 投影后使用；两个采样率的效果是否完全等价未单独评估。
- 降的是稳态与准稳态噪声；与人声同时出现、频谱相近的非稳态干扰（例如旁人说话、伴奏）不在设计目标内。
- 只提供 C 接口；Java / Kotlin 直接调用需要自行写 JNI 包装。
