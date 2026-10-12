// ans48 C 接口实现。网络 = UL-UNAS 结构的降噪掩码模型（手写流式推理，ulunas_kernel）；外圈 = 分帧 / FFT / 掩码^γ / 逆变换 / 重叠相加（denoise_stage.h）。
// 权重编进库里（models.cpp，tools/embed_models.py 生成），从内存加载。
#include "ans48.h"
#include "denoise_stage.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

extern "C" {
extern const unsigned char ans48_w256_bin[]; extern const unsigned long ans48_w256_bin_size;
extern const unsigned char ans48_w256_48k_json[]; extern const unsigned char ans48_w256_44k_json[];
extern const unsigned char ans48_w1024_48k_bin[]; extern const unsigned long ans48_w1024_48k_bin_size; extern const unsigned char ans48_w1024_48k_json[];
extern const unsigned char ans48_w1024_44k_bin[]; extern const unsigned long ans48_w1024_44k_bin_size; extern const unsigned char ans48_w1024_44k_json[];
extern const char ans48_model_id[];
}

namespace {
// 按位判 NaN：库用 -ffast-math 编，编译器假定没有 NaN，std::isnan 会被优化成恒假
inline bool isNaN(float x) { uint32_t u; memcpy(&u, &x, 4); return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0; }
inline short toS16(float x) { float v = x * 32767.f; return (short)(v > 32767.f ? 32767 : (v < -32768.f ? -32768 : lrintf(v))); }
const int kChunk = 256;   // int16 接口的转换块
}  // namespace

struct Ans48 {
    ulk::Net net; dns::StreamDenoise sd; int mode = 0, sr = 48000; float maxG = 1.f;
    std::atomic<float> gamma{1.f};
    float fin[kChunk], fout[kChunk];
    // 统计（只在 process 线程写；对外按秒发布）
    std::atomic<double> avgMs{0}, maxMs{0}; std::atomic<long long> samples{0};
    double accMs = 0, accMax = 0; long accN = 0; std::chrono::steady_clock::time_point lastStat;
};

static void run(Ans48* a, const float* in, float* out, int n) {
    a->sd.gamma = a->gamma.load(std::memory_order_relaxed);
    int done = 0;
    while (done < n) {   // 按帧边界切块：每块末尾若凑满一个 hop 就恰好算一帧，用来逐帧计时
        const int c = std::min(n - done, a->sd.hop - a->sd.fill);
        const auto t0 = std::chrono::steady_clock::now();
        a->sd.process(in + done, out + done, c);   // 逐样本先读后写，同址安全
        if (a->sd.fill == 0) {
            const auto t1 = std::chrono::steady_clock::now(); const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            a->accMs += ms; a->accN++; if (ms > a->accMax) a->accMax = ms;
            if (t1 - a->lastStat > std::chrono::seconds(1)) { a->avgMs = a->accMs / a->accN; a->maxMs = a->accMax; a->accMs = 0; a->accN = 0; a->accMax = 0; a->lastStat = t1; }
        }
        done += c;
    }
    a->samples.fetch_add(n, std::memory_order_relaxed);
}

extern "C" {
const char* ans48_version(void) { static const std::string v = std::string("ans48 0.2 (") + ans48_model_id + ")"; return v.c_str(); }

Ans48* ans48_create(int sample_rate, int mode) {
    if ((sample_rate != 48000 && sample_rate != 44100) || (mode != ANS48_MODE_MONITOR && mode != ANS48_MODE_STREAM)) return nullptr;
    Ans48* a = nullptr;
    try {
        a = new Ans48(); a->mode = mode; a->sr = sample_rate; a->maxG = mode == ANS48_MODE_MONITOR ? 2.f : 1.f;
        const bool r48 = sample_rate == 48000; bool ok;
        if (mode == ANS48_MODE_MONITOR) ok = a->net.w.loadFromMemory(ans48_w256_bin, ans48_w256_bin_size, (const char*)(r48 ? ans48_w256_48k_json : ans48_w256_44k_json));
        else if (r48) ok = a->net.w.loadFromMemory(ans48_w1024_48k_bin, ans48_w1024_48k_bin_size, (const char*)ans48_w1024_48k_json);
        else ok = a->net.w.loadFromMemory(ans48_w1024_44k_bin, ans48_w1024_44k_bin_size, (const char*)ans48_w1024_44k_json);
        if (!ok || a->net.w.fs != sample_rate || !a->net.prepare()) { delete a; return nullptr; }
        a->net.reset(); a->sd.prepare(&a->net); a->lastStat = std::chrono::steady_clock::now();
        return a;
    } catch (...) { delete a; return nullptr; }   // 内存不足 / 权重不符：返回 NULL，不结束宿主进程
}

int ans48_process(Ans48* a, const float* in, float* out, int n) {
    if (!a || !in || !out || n <= 0) return -1;
    run(a, in, out, n); return 0;
}

int ans48_process_s16(Ans48* a, const short* in, short* out, int n) {
    if (!a || !in || !out || n <= 0) return -1;
    for (int off = 0; off < n; off += kChunk) {   // 分段转换（段长任意，处理本身逐样本，不影响时间对齐）；同址安全（每段先读后写）
        const int c = std::min(kChunk, n - off);
        for (int i = 0; i < c; ++i) a->fin[i] = in[off + i] / 32768.f;
        run(a, a->fin, a->fout, c);
        for (int i = 0; i < c; ++i) out[off + i] = toS16(a->fout[i]);
    }
    return 0;
}

float ans48_set_strength(Ans48* a, float g) {
    if (!a) return 0.f;
    if (!isNaN(g)) a->gamma.store(g < 0.f ? 0.f : (g > a->maxG ? a->maxG : g), std::memory_order_relaxed);
    return a->gamma.load(std::memory_order_relaxed);
}
float ans48_get_strength(const Ans48* a) { return a ? a->gamma.load(std::memory_order_relaxed) : 0.f; }
float ans48_max_strength(const Ans48* a) { return a ? a->maxG : 0.f; }
int ans48_latency_samples(const Ans48* a) { return a ? a->sd.nFft : 0; }
void ans48_reset(Ans48* a) { if (a) a->sd.reset(); }
void ans48_stats(const Ans48* a, double o[3]) { if (!a || !o) return; o[0] = a->avgMs; o[1] = a->maxMs; o[2] = (double)a->samples.load(); }
void ans48_destroy(Ans48* a) { delete a; }
}
