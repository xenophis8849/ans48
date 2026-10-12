// 端上测速：按真实节奏喂数据，量处理线程的 CPU 占用与每块耗时。核心用 taskset 指定。
// 用法：ans48_bench <in.pcm> <采样率> <monitor|stream> <秒数> [paced=1] [每次块长=一个 hop（耳返 128、推流 512）] [强度]
//   输入不够长就循环；paced=0 时不睡、全速跑（量实时率）。输出一行：CPU 占用 = 处理线程 CPU 时间 / 墙钟时间；每块耗时均值 / p99 / 最大；
//   超时 = 单块超过一个块周期的次数；内存 = 建实例前后与跑完后的常驻内存差。块长 = 一个 hop 时每块恰好算一帧。
#include "ans48.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
static std::vector<short> rd(const char* p) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(2); } fseek(f, 0, SEEK_END); long n = ftell(f) / 2; fseek(f, 0, SEEK_SET); std::vector<short> v(n); if (fread(v.data(), 2, n, f) != (size_t)n) exit(2); fclose(f); return v; }
static long rss_kb() { FILE* f = fopen("/proc/self/status", "r"); char l[256]; long v = -1; while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6); if (f) fclose(f); return v; }
static double tcpu() { timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "用法：%s in.pcm 采样率 monitor|stream 秒数 [paced=1] [块长] [强度]\n", argv[0]); return 2; }
    auto in = rd(argv[1]); int sr = atoi(argv[2]); const bool mon = !strcmp(argv[3], "monitor"); double secs = atof(argv[4]); bool paced = argc < 6 || atoi(argv[5]) != 0;
    const int B = argc > 6 && atoi(argv[6]) > 0 ? atoi(argv[6]) : (mon ? 128 : 512);
    long nIn = (long)in.size() / B * B, nb = (long)(secs * sr / B);
    std::vector<float> x(nIn), o(B); for (long i = 0; i < nIn; ++i) x[i] = in[i] / 32768.f;
    long r0 = rss_kb(); Ans48* a = ans48_create(sr, mon ? ANS48_MODE_MONITOR : ANS48_MODE_STREAM); if (!a) { fprintf(stderr, "ans48_create 失败\n"); return 1; } long r1 = rss_kb();
    if (argc > 7) ans48_set_strength(a, (float)atof(argv[7]));
    std::vector<double> t(nb); int over = 0; const double per = 1000.0 * B / sr;
    auto w0 = std::chrono::steady_clock::now(); double c0 = tcpu(); auto next = w0;
    for (long k = 0; k < nb; ++k) {
        if (paced) { std::this_thread::sleep_until(next); next += std::chrono::nanoseconds((long long)(1e9 * B / sr)); }
        long off = (k * B) % nIn; auto s = std::chrono::steady_clock::now();
        ans48_process(a, &x[off], o.data(), B);
        t[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count(); if (t[k] > per) ++over;
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(), cpu = tcpu() - c0; long r2 = rss_kb();
    std::vector<double> s = t; std::sort(s.begin(), s.end()); double mean = 0; for (double v : t) mean += v; mean /= nb;
    printf("{\"mode\":\"%s\",\"sr\":%d,\"block\":%d,\"gamma\":%.2f,\"paced\":%d,\"sec\":%.1f,\"blocks\":%ld,\"cpu_pct\":%.2f,\"rtf\":%.4f,\"mean_ms\":%.3f,\"p99_ms\":%.3f,\"max_ms\":%.3f,\"over\":%d,\"rss_create_kb\":%ld,\"rss_run_kb\":%ld}\n",
           mon ? "monitor" : "stream", sr, B, ans48_get_strength(a), paced, wall, nb, 100.0 * cpu / wall, mean / per, mean, s[(size_t)(0.99 * (nb - 1))], s.back(), over, r1 - r0, r2 - r0);
    ans48_destroy(a); return 0;
}
