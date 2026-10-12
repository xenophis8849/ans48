// 自检：用库处理测试向量，与参照实现（训练端的流式 torch 版）的期望输出逐样本比较。
// 用法：ans48_check <in.pcm> <采样率> <monitor|stream> <强度> <期望输出.pcm> [每次块长=480] [写出.pcm]
//   输入 / 期望输出都是 int16 单声道；期望输出已在输入的时间轴上（第 i 个样本 = 输入第 i 个样本降噪后的结果），
//   库的输出按 ans48_latency_samples 挪齐后再比。通过标准：差值能量比期望输出低 ≥ 50 dB。
#include "ans48.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
static std::vector<short> rd(const char* p) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(2); } fseek(f, 0, SEEK_END); long n = ftell(f) / 2; fseek(f, 0, SEEK_SET); std::vector<short> v(n); if (fread(v.data(), 2, n, f) != (size_t)n) exit(2); fclose(f); return v; }
static double db(double e, double r) { return 10 * log10((e + 1e-9) / (r + 1e-9)); }
int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "用法：%s in.pcm 采样率 monitor|stream 强度 期望输出.pcm [块长] [写出.pcm]\n", argv[0]); return 2; }
    auto in = rd(argv[1]); int sr = atoi(argv[2]); const bool mon = !strcmp(argv[3], "monitor"); float g = (float)atof(argv[4]); auto exp_ = rd(argv[5]);
    const int blk = argc > 6 ? atoi(argv[6]) : 480;
    Ans48* a = ans48_create(sr, mon ? ANS48_MODE_MONITOR : ANS48_MODE_STREAM); if (!a) { fprintf(stderr, "ans48_create 失败\n"); return 1; }
    const float ga = ans48_set_strength(a, g); const long L = ans48_latency_samples(a), n = (long)in.size();
    printf("%s · 采样率 %d · %s · 强度 %.2f（上限 %.0f）· 延迟 %ld 样本（%.1f ms）· 块长 %d\n", ans48_version(), sr, mon ? "耳返" : "推流", ga, ans48_max_strength(a), L, 1000.0 * L / sr, blk);
    // 输入后面补 L 个零，把尾巴推出来
    std::vector<short> x(in); x.resize(n + L, 0); std::vector<short> y(n + L, 0); int bad = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < n + L; i += blk) { int c = (int)std::min<long>(blk, n + L - i); if (ans48_process_s16(a, &x[i], &y[i], c) != 0) ++bad; }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); double st[3]; ans48_stats(a, st);
    long m = std::min<long>(n, (long)exp_.size()); double d = 0, p = 0, ein = 0, eout = 0; long worst = 0;
    for (long i = 0; i < m; ++i) { double o = y[i + L], e = exp_[i]; d += (o - e) * (o - e); p += e * e; ein += (double)in[i] * in[i]; eout += o * o; worst = std::max(worst, (long)std::fabs(o - e)); }
    printf("处理 %.1f s 音频用时 %.2f s（实时率 %.3f）；最近一秒每帧均 %.3f ms、最大 %.3f ms\n", (double)n / sr, sec, sec / ((double)n / sr), st[0], st[1]);
    printf("输出 / 输入能量 %.2f dB\n", db(eout, ein));
    const double rel = db(d, p); const bool ok = rel <= -50 && bad == 0;
    printf("对期望输出：差值能量 %.1f dB，最大差 %ld LSB%s → %s\n", rel, worst, bad ? "，process 返回错误" : "", ok ? "通过" : "不通过");
    if (argc > 7) { FILE* f = fopen(argv[7], "wb"); if (f) { fwrite(&y[L], 2, m, f); fclose(f); } }
    ans48_destroy(a); return ok ? 0 : 1;
}
