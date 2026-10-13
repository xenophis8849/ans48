/*
 * ans48 0.3.0 —— 实时全带降噪（48 kHz / 44.1 kHz，单声道）
 *
 * 三档：
 *   ANS48_MODE_MONITOR  耳返：窗 256 样本，算法延迟 256 样本（48 kHz 下 5.3 ms），强度 0 ~ 2
 *   ANS48_MODE_STREAM   录音 / 推流：窗 1024 样本，算法延迟 1024 样本（48 kHz 下 21.3 ms），强度 0 ~ 2
 *   频谱档（ans48_create_spec）：挂在 aec48 的频谱回调上，直接在回声消除后的频谱上降噪，不另加延迟，强度 0 ~ 2。
 *                                见下面「联动接口」一节与 README 同名一节。
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
/* 该档允许的最大强度：三档都是 2。 */
ANS48_EXPORT float ans48_max_strength(const Ans48* a);

/* out 相对 in 的固定延迟（样本数）= 窗长；频谱档为 0（不另加延迟）。 */
ANS48_EXPORT int ans48_latency_samples(const Ans48* a);

/* 清空内部状态（例如换歌、断流后重新开始）；强度不变。不要与 process 并发调用。 */
ANS48_EXPORT void ans48_reset(Ans48* a);

/* 统计：[0] 最近一秒每帧平均处理耗时 ms  [1] 最近一秒每帧最大 ms  [2] 已处理样本数。
 * 一帧 = 一个 hop（耳返 / 推流为窗长的一半；频谱档为一次 ans48_spec_frame，即 384 样本）。 */
ANS48_EXPORT void ans48_stats(const Ans48* a, double out3[3]);

ANS48_EXPORT void ans48_destroy(Ans48* a);

/* ---- ⚠ 联动接口：频谱档（与 aec48 串接）----------------------------------------------------------
 * 跨项目契约，准据是 aec48 的 README「⚠ 联动接口：频谱回调（与降噪库串接）」一节；本库按其中的契约版本实现。
 * aec48 每处理一块，在把输出频谱合成回声音之前调用挂上的回调；频谱档在这份频谱上原地乘掩码^γ，不另做分帧 / FFT，不加延迟。
 * 两个库互不链接，只靠帧格式约定对接。接入方可以在编译时核对两边的契约版本：
 *     #if AEC48_SPEC_CONTRACT_VERSION != ANS48_SPEC_CONTRACT_VERSION
 *     #error "aec48 与 ans48 的联动接口契约版本不一致"
 *     #endif
 * 接法（两行，处理开始前做好）：
 *     int sr, nfft, hop, win; aec48_spec_format(aec, &sr, &nfft, &hop, &win);
 *     Ans48* n = ans48_create_spec(sr, nfft, hop, win);          // 格式对不上返回 NULL，就不要挂
 *     aec48_set_spec_hook(aec, ans48_spec_frame, n);
 * 运行中临时关降噪：优先 ans48_set_strength(n, 0)——网络照常逐帧跑、输出原样，跨帧记忆连续，重开时没有过渡问题；
 * 用 aec48 的挂钩开关关掉时，ans48 收不到帧，重开后会带着关之前的记忆处理开头一小段。 */
#define ANS48_SPEC_CONTRACT_VERSION 1
#define ANS48_WINDOW_SQRT_HANN_PERIODIC 1   /* 与 aec48 的 window_id 编号一致 */

/* 建频谱档实例。参数取自 aec48_spec_format：sample_rate 48000 / 44100，nfft 1536，hop 384，window_id 1（周期 sqrt-Hann）；
 * 与本库实现的契约不符、或内存不足，返回 NULL。初始强度 1。ans48_process / ans48_process_s16 对频谱档返回 -1。 */
ANS48_EXPORT Ans48* ans48_create_spec(int sample_rate, int nfft, int hop, int window_id);

/* 频谱档逐帧处理，签名与 aec48_spec_hook 一致：user 传 ans48_create_spec 返回的 Ans48*，原地修改 re / im（bins = nfft/2 + 1 个）。
 * 在调用它的线程里同步执行，不分配内存、不加锁、不阻塞；同一实例同一时刻只能在一个线程里调用。
 * user 为空、不是频谱档实例或 bins 不符时什么都不做（频谱原样返回）。 */
ANS48_EXPORT void ans48_spec_frame(void* user, float* re, float* im, int bins);

/* 版本与模型标识。 */
ANS48_EXPORT const char* ans48_version(void);

#ifdef __cplusplus
}
#endif
#endif
