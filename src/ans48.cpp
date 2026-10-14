// ans48 C 接口实现。网络 = UL-UNAS 结构的降噪掩码模型（手写流式推理，ulunas_kernel）；外圈 = 分帧 / FFT / 掩码^γ / 逆变换 / 重叠相加（denoise_stage.h）。
// 权重编进库里（models.cpp，tools/embed_models.py 生成），从内存加载。
#include "ans48.h"
#include "denoise_stage.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

extern "C" {
extern const unsigned char ans48_w256_bin[]; extern const unsigned long ans48_w256_bin_size;
extern const unsigned char ans48_w256_48k_json[]; extern const unsigned char ans48_w256_44k_json[];
// 推流档 / 频谱档：一个 bin 两份清单（网络张量共用，ERB 投影两个采样率各一份；见 tools/embed_models.py）
extern const unsigned char ans48_w1024_bin[]; extern const unsigned long ans48_w1024_bin_size; extern const unsigned char ans48_w1024_48k_json[]; extern const unsigned char ans48_w1024_44k_json[];
extern const unsigned char ans48_w1536_bin[]; extern const unsigned long ans48_w1536_bin_size; extern const unsigned char ans48_w1536_48k_json[]; extern const unsigned char ans48_w1536_44k_json[];
extern const char ans48_model_id[];
}

namespace {
// 按位判 NaN：库用 -ffast-math 编，编译器假定没有 NaN，std::isnan 会被优化成恒假
inline bool isNaN(float x) { uint32_t u; memcpy(&u, &x, 4); return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0; }
inline short toS16(float x) { float v = x * 32767.f; return (short)(v > 32767.f ? 32767 : (v < -32768.f ? -32768 : lrintf(v))); }
const int kChunk = 256;   // int16 接口的转换块
const int kModeSpec = 2;  // 频谱档（只能经 ans48_create_spec 建）
// 频谱档实现的联动契约（aec48 README「⚠ 联动接口」版本 1）：窗长 1536、帧移 384、周期 sqrt-Hann
const int kSpecNfft = 1536, kSpecHop = 384, kSpecBins = kSpecNfft / 2 + 1;
}  // namespace

struct Ans48 {
    std::unique_ptr<ulk_if::INet> net; dns::StreamDenoiseT<ulk_if::INet> sd; int mode = 0, sr = 48000; float maxG = 2.f;   // net：按 CPU 能力选的那份内核（net_select.cpp）
    std::vector<float> smag, smask;   // 频谱档：每帧的 log 幅度与掩码（prepare 时分配，处理期不分配）
    std::atomic<float> gamma{1.f};
    float fin[kChunk], fout[kChunk];
    // 统计（只在 process 线程写；对外按秒发布）
    std::atomic<double> avgMs{0}, maxMs{0}; std::atomic<long long> samples{0};
    double accMs = 0, accMax = 0; long accN = 0; std::chrono::steady_clock::time_point lastStat;
};

static void statFrame(Ans48* a, std::chrono::steady_clock::time_point t0) {
    const auto t1 = std::chrono::steady_clock::now(); const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    a->accMs += ms; a->accN++; if (ms > a->accMax) a->accMax = ms;
    if (t1 - a->lastStat > std::chrono::seconds(1)) { a->avgMs = a->accMs / a->accN; a->maxMs = a->accMax; a->accMs = 0; a->accN = 0; a->accMax = 0; a->lastStat = t1; }
}

static void run(Ans48* a, const float* in, float* out, int n) {
    a->sd.gamma = a->gamma.load(std::memory_order_relaxed);
    int done = 0;
    while (done < n) {   // 按帧边界切块：每块末尾若凑满一个 hop 就恰好算一帧，用来逐帧计时
        const int c = std::min(n - done, a->sd.hop - a->sd.fill);
        const auto t0 = std::chrono::steady_clock::now();
        a->sd.process(in + done, out + done, c);   // 逐样本先读后写，同址安全
        if (a->sd.fill == 0) statFrame(a, t0);
        done += c;
    }
    a->samples.fetch_add(n, std::memory_order_relaxed);
}

extern "C" {
const char* ans48_version(void) { static const std::string v = std::string("ans48 0.4.1 (") + ans48_model_id + ")"; return v.c_str(); }

Ans48* ans48_create(int sample_rate, int mode) {
    if ((sample_rate != 48000 && sample_rate != 44100) || (mode != ANS48_MODE_MONITOR && mode != ANS48_MODE_STREAM)) return nullptr;
    Ans48* a = nullptr;
    try {
        a = new Ans48(); a->mode = mode; a->sr = sample_rate; a->maxG = 2.f;   // 10-13 起三档上限都是 2（推流档原为 1）
        const bool r48 = sample_rate == 48000; bool ok;
        a->net.reset(ulk_if::newNet());
        if (mode == ANS48_MODE_MONITOR) ok = a->net->load(ans48_w256_bin, ans48_w256_bin_size, (const char*)(r48 ? ans48_w256_48k_json : ans48_w256_44k_json));
        else ok = a->net->load(ans48_w1024_bin, ans48_w1024_bin_size, (const char*)(r48 ? ans48_w1024_48k_json : ans48_w1024_44k_json));
        if (!ok || !a->net->prepare() || a->net->fs != sample_rate) { delete a; return nullptr; }
        a->net->reset(); a->sd.prepare(a->net.get(), a->net->nFft, a->net->hop, a->net->bins); a->lastStat = std::chrono::steady_clock::now();
        return a;
    } catch (...) { delete a; return nullptr; }   // 内存不足 / 权重不符：返回 NULL，不结束宿主进程
}

Ans48* ans48_create_spec(int sample_rate, int nfft, int hop, int window_id) {
    if ((sample_rate != 48000 && sample_rate != 44100) || nfft != kSpecNfft || hop != kSpecHop || window_id != ANS48_WINDOW_SQRT_HANN_PERIODIC) return nullptr;
    Ans48* a = nullptr;
    try {
        a = new Ans48(); a->mode = kModeSpec; a->sr = sample_rate; a->maxG = 2.f;
        const bool r48 = sample_rate == 48000;
        a->net.reset(ulk_if::newNet());
        const bool ok = a->net->load(ans48_w1536_bin, ans48_w1536_bin_size, (const char*)(r48 ? ans48_w1536_48k_json : ans48_w1536_44k_json));
        // 嵌入的模型必须正是按契约分帧训的那一份：窗长、帧移、频点数、采样率逐项核
        if (!ok || !a->net->prepare() || a->net->fs != sample_rate || a->net->nFft != kSpecNfft || a->net->hop != kSpecHop || a->net->bins != kSpecBins) { delete a; return nullptr; }
        a->smag.assign(kSpecBins, 0.f); a->smask.assign(kSpecBins, 0.f);
        a->net->reset(); a->lastStat = std::chrono::steady_clock::now();
        return a;
    } catch (...) { delete a; return nullptr; }
}

// 与推流档外圈 StreamDenoise::frame 中「幅度 → log10 → 网络 → 掩码限到 [0,1] → ^γ → 乘回」同一组函数（dns::logMag / applyMask），只是频谱由 aec48 给、不做 FFT
void ans48_spec_frame(void* user, float* re, float* im, int bins) {
    Ans48* a = (Ans48*)user;
    if (!a || a->mode != kModeSpec || !re || !im || bins != kSpecBins) return;
    const auto t0 = std::chrono::steady_clock::now();
    const float g = a->gamma.load(std::memory_order_relaxed);
    float* mag = a->smag.data(); float* mask = a->smask.data();
    dns::logMag(re, im, mag, bins);
    a->net->run(mag, mask);
    dns::applyMask(re, im, mask, bins, g);
    statFrame(a, t0);
    a->samples.fetch_add(kSpecHop, std::memory_order_relaxed);
}

int ans48_process(Ans48* a, const float* in, float* out, int n) {
    if (!a || !in || !out || n <= 0 || a->mode == kModeSpec) return -1;
    run(a, in, out, n); return 0;
}

int ans48_process_s16(Ans48* a, const short* in, short* out, int n) {
    if (!a || !in || !out || n <= 0 || a->mode == kModeSpec) return -1;
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
int ans48_latency_samples(const Ans48* a) { return a && a->mode != kModeSpec ? a->sd.nFft : 0; }
void ans48_reset(Ans48* a) { if (!a) return; if (a->mode == kModeSpec) a->net->reset(); else a->sd.reset(); }
void ans48_stats(const Ans48* a, double o[3]) { if (!a || !o) return; o[0] = a->avgMs; o[1] = a->maxMs; o[2] = (double)a->samples.load(); }
void ans48_destroy(Ans48* a) { delete a; }
}
