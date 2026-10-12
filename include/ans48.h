/*
 * ans48 0.1.1 —— 实时全带降噪（48 kHz / 44.1 kHz，单声道）
 *
 * 两档：
 *   ANS48_MODE_MONITOR  耳返：窗 256 样本，算法延迟 256 样本（48 kHz 下 5.3 ms），强度 0 ~ 2
 *   ANS48_MODE_STREAM   录音 / 推流：窗 1024 样本，算法延迟 1024 样本（48 kHz 下 21.3 ms），强度 0 ~ 1
 * 强度 = 掩码指数 γ：输出谱 = 输入谱 × 掩码^γ。0 = 原样直通（只有延迟），1 = 模型全档，>1 比全档压得更狠。
 *
 * 线程：一个实例同一时刻只能在一个线程里 process；ans48_set_strength 可以在任意线程调用（下一帧生效）。
 * 处理期不分配内存、不加锁、不做文件读写，可以直接在音频回调里调用。
 */
#ifndef ANS48_H
#define ANS48_H

#ifdef __cplusplus
extern "C" {
#endif

#define ANS48_EXPORT __attribute__((visibility("default")))

typedef struct Ans48 Ans48;

enum { ANS48_MODE_MONITOR = 0, ANS48_MODE_STREAM = 1 };

/* 建实例。sample_rate 只支持 48000、44100；mode 见上。失败（参数不支持、内存不足）返回 NULL，不会结束宿主进程。
 * 初始强度为 1（模型全档）。 */
ANS48_EXPORT Ans48* ans48_create(int sample_rate, int mode);

/* 处理 n 个样本（float，±1 满幅），任意长度；out 可以与 in 同址。返回 0；参数无效返回 -1。
 * 输出比输入晚 ans48_latency_samples 个样本；开头这一段是启动段，输出接近 0（重叠相加起步时的舍入余量，不超过 −60 dBFS）。 */
ANS48_EXPORT int ans48_process(Ans48* a, const float* in, float* out, int n);

/* int16 版，同上（与 float 版输出一致，量化到 int16）。 */
ANS48_EXPORT int ans48_process_s16(Ans48* a, const short* in, short* out, int n);

/* 设置强度 γ，截到 [0, ans48_max_strength]；NaN 忽略。返回实际生效的值。 */
ANS48_EXPORT float ans48_set_strength(Ans48* a, float gamma);
ANS48_EXPORT float ans48_get_strength(const Ans48* a);
/* 该档允许的最大强度：耳返 2，推流 1。 */
ANS48_EXPORT float ans48_max_strength(const Ans48* a);

/* out 相对 in 的固定延迟（样本数）= 窗长。 */
ANS48_EXPORT int ans48_latency_samples(const Ans48* a);

/* 清空内部状态（例如换歌、断流后重新开始）；强度不变。不要与 process 并发调用。 */
ANS48_EXPORT void ans48_reset(Ans48* a);

/* 统计：[0] 最近一秒每帧平均处理耗时 ms  [1] 最近一秒每帧最大 ms  [2] 已处理样本数。一帧 = 一个 hop（窗长的一半）。 */
ANS48_EXPORT void ans48_stats(const Ans48* a, double out3[3]);

ANS48_EXPORT void ans48_destroy(Ans48* a);

/* 版本与模型标识。 */
ANS48_EXPORT const char* ans48_version(void);

#ifdef __cplusplus
}
#endif
#endif
