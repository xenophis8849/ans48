// 降噪的流式外圈：分帧 → FFT → 网络（手写内核）→ 掩码^γ → 逆变换 → 重叠相加。逐样本推进，处理期零分配，可在音频回调里调用。
// 固有延迟 = 窗长：开头预置一个 hop 的滞后，加上一帧要凑满窗（窗 256 在 48 kHz 下 5.3 ms，窗 1024 为 21.3 ms）。
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "ulunas_kernel.h"
#include "net_iface.h"
#include "pffft.h"
#include <new>

namespace dns {

/** 实数 FFT（pffft，有序打包：out[0] = DC，out[1] = 奈奎斯特，之后 (实, 虚) 成对）。缓冲在 init 时对齐分配，处理期不分配。 */
struct RFft { int N = 0; PFFFT_Setup* s = nullptr; float *in = nullptr, *out = nullptr, *wk = nullptr;
    RFft() = default; RFft(const RFft&) = delete; RFft& operator=(const RFft&) = delete;
    ~RFft() { release(); }
    void release() { if (s) pffft_destroy_setup(s); pffft_aligned_free(in); pffft_aligned_free(out); pffft_aligned_free(wk); s = nullptr; in = out = wk = nullptr; }
    void init(int n) { release(); N = n; s = pffft_new_setup(n, PFFFT_REAL);
        in = (float*)pffft_aligned_malloc(n * 4); out = (float*)pffft_aligned_malloc(n * 4); wk = (float*)pffft_aligned_malloc(n * 4);
        if (!s || !in || !out || !wk) { release(); throw std::bad_alloc(); } }
};

// 对数幅度（10-14）：out[k] = log10(max(|X_k|, 1e-12)) = 0.5·log10(max(re² + im², 1e-24))，省掉开方；对数用 Cephes logf 的多项式
// （与 NEON_mathfun 的 log_ps 同一组系数，单精度误差约 1 ulp），4 个一组。原写法逐频点 sqrtf + 库函数 log10f：采样剖析 vivo 小核 32 位，
// 外圈（分帧 / FFT / 对数 / 掩码 / 逆变换 / 重叠相加）占整帧 9–11%，其中 log10f 一项 2.4%。推流档外圈与频谱档（ans48.cpp）共用。
typedef float lv4 __attribute__((vector_size(16)));
typedef int lv4i __attribute__((vector_size(16)));
static inline lv4 vlnf(lv4 x) {   // x 为正的规格化数
    lv4i bi; memcpy(&bi, &x, 16);
    const lv4i ei = ((bi >> 23) & 0xff) - 126;                 // x = m · 2^e，m ∈ [0.5, 1)
    bi = (bi & 0x007fffff) | 0x3f000000; lv4 m; memcpy(&m, &bi, 16);
    const lv4i lt = m < 0.707106781186547524f;                 // m < √½：取 2m、指数减 1，使 m − 1 落在 [−0.29, 0.41]
    lv4 e = __builtin_convertvector(ei + lt, lv4);
    lv4i mb; memcpy(&mb, &m, 16); mb &= lt; lv4 t; memcpy(&t, &mb, 16);
    m = m - 1.0f + t;
    const lv4 z = m * m;
    lv4 y = 7.0376836292e-2f * m - 1.1514610310e-1f; y = y * m + 1.1676998740e-1f; y = y * m - 1.2420140846e-1f; y = y * m + 1.4249322787e-1f;
    y = y * m - 1.6668057665e-1f; y = y * m + 2.0000714765e-1f; y = y * m - 2.4999993993e-1f; y = y * m + 3.3333331174e-1f;
    y = y * m * z + e * -2.12194440e-4f - 0.5f * z;
    return m + y + e * 0.693359375f;
}
inline void logMag(const float* re, const float* im, float* out, int n) {
    const lv4 lo = {1e-24f, 1e-24f, 1e-24f, 1e-24f};
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        lv4 r, i; memcpy(&r, re + k, 16); memcpy(&i, im + k, 16);
        lv4 p = r * r + i * i; p = p < lo ? lo : p;
        const lv4 y = vlnf(p) * 0.217147240951625947f;   // 0.5 / ln 10
        memcpy(out + k, &y, 16);
    }
    if (k < n) {
        float pr[4] = {1, 1, 1, 1}, pi[4] = {0, 0, 0, 0}, po[4];
        for (int j = 0; k + j < n; ++j) { pr[j] = re[k + j]; pi[j] = im[k + j]; }
        logMag(pr, pi, po, 4);
        for (int j = 0; k + j < n; ++j) out[k + j] = po[j];
    }
}

static inline lv4 vexpf(lv4 x) {   // Cephes expf（NEON_mathfun exp_ps 同一组系数），单精度误差约 1 ulp
    x = x > 88.3762626647949f ? (lv4){88.3762626647949f, 88.3762626647949f, 88.3762626647949f, 88.3762626647949f} : x;
    x = x < -88.3762626647949f ? (lv4){-88.3762626647949f, -88.3762626647949f, -88.3762626647949f, -88.3762626647949f} : x;
    lv4 fx = x * 1.44269504088896341f + 0.5f;
    lv4i fi = __builtin_convertvector(fx, lv4i); lv4 t = __builtin_convertvector(fi, lv4);
    const lv4i gt = t > fx; fi += gt; t = __builtin_convertvector(fi, lv4);   // 向下取整（截断后若大于原值再减 1；gt 为 −1）
    x = x - t * 0.693359375f - t * -2.12194440e-4f;
    const lv4 z = x * x;
    lv4 y = 1.9875691500e-4f * x + 1.3981999507e-3f; y = y * x + 8.3334519073e-3f; y = y * x + 4.1665795894e-2f; y = y * x + 1.6666665459e-1f; y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    lv4i pw = (fi + 127) << 23; lv4 sc; memcpy(&sc, &pw, 16);   // fi ≥ −127：x ≥ −88.38 时成立
    return y * sc;
}
// 掩码限到 [0, 1]、^γ、乘回频谱（10-14 向量化：原写法逐频点 powf，γ ≠ 1 时 129 / 513 / 769 次库函数调用）。
// v^γ = exp(γ·ln v)；v = 0 时同 powf：γ > 0 得 0、γ = 0 得 1。
inline void applyMask(float* re, float* im, const float* mask, int n, float gamma) {
    int k = 0;
    const lv4 zero = {0, 0, 0, 0}, one = {1, 1, 1, 1}, tiny = {1e-37f, 1e-37f, 1e-37f, 1e-37f};
    const lv4 g0 = gamma == 0.0f ? one : zero;
    for (; k + 4 <= n; k += 4) {
        lv4 v; memcpy(&v, mask + k, 16);
        v = v < zero ? zero : v; v = v > one ? one : v;
        if (gamma != 1.0f) { const lv4 p = vexpf(gamma * vlnf(v < tiny ? tiny : v)); v = v < tiny ? g0 : p; }
        lv4 r, i; memcpy(&r, re + k, 16); memcpy(&i, im + k, 16); r *= v; i *= v; memcpy(re + k, &r, 16); memcpy(im + k, &i, 16);
    }
    if (k < n) {
        float pr[4] = {0, 0, 0, 0}, pi[4] = {0, 0, 0, 0}, pm[4] = {0, 0, 0, 0};
        for (int j = 0; k + j < n; ++j) { pr[j] = re[k + j]; pi[j] = im[k + j]; pm[j] = mask[k + j]; }
        applyMask(pr, pi, pm, 4, gamma);
        for (int j = 0; k + j < n; ++j) { re[k + j] = pr[j]; im[k + j] = pi[j]; }
    }
}

// N = 网络类型：库里是 ulk_if::INet（按 CPU 能力选的那份内核），自检 / 开发工具直接用 ulk::Net。要求有 run(logmag, mask) 与 reset()。
template <class N>
struct StreamDenoiseT {
    N* net = nullptr;
    int nFft = 256, hop = 128, bins = 129;
    float gamma = 1.0f;
    RFft fft;
    std::vector<float> win, inbuf, num, den, re, im, mag, mask, fifo;
    std::vector<float> invDen;   // 稳态时重叠相加分母的倒数（只由窗定，prepare 里算好；见 frame）
    int fill = 0;          // 当前 hop 已攒了几个样本
    int nFr = 0;           // reset 以来处理过的帧数，饱和在 nFft / hop（凑满后分母进入稳态）
    uint64_t fifoR = 0, fifoW = 0;   // 按样本累计的读 / 写位置。64 位：32 位平台上 size_t 约 24.9 h（48 kHz）就回绕，旧的 fifoR < fifoW 判断会从此恒假、持续输出 0

    void prepare(N* n_, int nFft_, int hop_, int bins_) {
        net = n_;
        nFft = nFft_; hop = hop_; bins = bins_;
        fft.init(nFft);
        win.resize(nFft);
        for (int i = 0; i < nFft; ++i)
            win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / nFft);
        inbuf.assign(nFft, 0.0f);
        num.assign(nFft, 0.0f); den.assign(nFft, 0.0f);
        re.assign(nFft, 0.0f); im.assign(nFft, 0.0f);
        mag.assign(bins, 0.0f); mask.assign(bins, 0.0f);
        fifo.assign(1 << 13, 0.0f);
        // 稳态分母：输出位置 i（< hop）上叠着 nFft / hop 帧的窗平方，按原式的累加顺序（最早的一帧先加）
        invDen.assign(hop, 0.0f);
        for (int i = 0; i < hop; ++i) { float d = 0.0f; for (int j = nFft / hop - 1; j >= 0; --j) d += win[i + j * hop] * win[i + j * hop]; invDen[i] = 1.0f / (d < 1e-11f ? 1e-11f : d); }
        reset();
    }
    // 直接拿内核类型（ulk::Net）时的简写：分帧参数从权重清单取
    template <class M = N> void prepare(M* n_) { prepare(n_, n_->w.nFft, n_->w.hop, n_->bins); }

    void reset() {
        if (net) net->reset();
        std::fill(inbuf.begin(), inbuf.end(), 0.0f);
        std::fill(num.begin(), num.end(), 0.0f);
        std::fill(den.begin(), den.end(), 0.0f);
        std::fill(fifo.begin(), fifo.end(), 0.0f);
        fill = 0; nFr = 0; fifoR = 0; fifoW = (size_t)hop;   // 预置一个 hop 的固有滞后
    }

    inline void frame() {
        const int M = nFft / 2;
        for (int i = 0; i < nFft; ++i) fft.in[i] = inbuf[i] * win[i];
        pffft_transform_ordered(fft.s, fft.in, fft.out, fft.wk, PFFFT_FORWARD);
        re[0] = fft.out[0]; im[0] = 0.0f; re[M] = fft.out[1]; im[M] = 0.0f;
        for (int k = 1; k < M; ++k) { re[k] = fft.out[2 * k]; im[k] = fft.out[2 * k + 1]; }
        logMag(re.data(), im.data(), mag.data(), bins);
        net->run(mag.data(), mask.data());
        applyMask(re.data(), im.data(), mask.data(), bins, gamma);
        // 逆实数 FFT（DC 与奈奎斯特频点的虚部按定义为 0，同 torch.fft.irfft）
        fft.in[0] = re[0]; fft.in[1] = re[M];
        for (int k = 1; k < M; ++k) { fft.in[2 * k] = re[k]; fft.in[2 * k + 1] = im[k]; }
        pffft_transform_ordered(fft.s, fft.in, fft.out, fft.wk, PFFFT_BACKWARD);
        const float* tf = fft.out;
        const float inv = 1.0f / nFft;
        // 重叠相加：分子 w·ifft、分母 w²、相除。与离线逐字同式。
        memmove(num.data(), num.data() + hop, sizeof(float) * (nFft - hop));
        std::fill(num.end() - hop, num.end(), 0.0f);
        for (int i = 0; i < nFft; ++i) num[i] += win[i] * tf[i] * inv;
        const size_t mask_ = fifo.size() - 1;
        if (nFr < nFft / hop) ++nFr;   // 饱和在阈值：只用来区分开头几帧与稳态，长时间运行不能溢出（0.4.1，审查指出 int 每帧递增约 66 天溢出、退回开头分支）
        if (nFr >= nFft / hop) {
            // 10-14：凑满 nFft / hop 帧后分母只由窗定（各帧窗平方的重叠和），乘准备期算好的倒数，省掉每帧 hop 次除法与分母的整块搬移（差 ≤1 ulp）
            for (int i = 0; i < hop; ++i) fifo[(size_t)((fifoW + i) & mask_)] = num[i] * invDen[i];
        } else {   // 开头几帧：前面还没有帧叠上来，分母照原式逐帧累加
            memmove(den.data(), den.data() + hop, sizeof(float) * (nFft - hop));
            std::fill(den.end() - hop, den.end(), 0.0f);
            for (int i = 0; i < nFft; ++i) den[i] += win[i] * win[i];
            for (int i = 0; i < hop; ++i) {
                float d = den[i] < 1e-11f ? 1e-11f : den[i];
                fifo[(size_t)((fifoW + i) & mask_)] = num[i] / d;
            }
        }
        fifoW += hop;
    }

    /** 处理 n 个样本：吐出同样多的、已降噪的样本。**回调里调用，绝不分配。** */
    inline void process(const float* x, float* y, int n) {
        const size_t m = fifo.size() - 1;
        for (int i = 0; i < n; ++i) {
            inbuf[nFft - hop + fill] = x[i];
            if (++fill == hop) {
                frame();
                memmove(inbuf.data(), inbuf.data() + hop, sizeof(float) * (nFft - hop));
                fill = 0;
            }
            y[i] = (fifoR != fifoW) ? fifo[(size_t)(fifoR++ & m)] : 0.0f;   // 写位置不会落后读位置，不相等即有数据（即使回绕也成立）
        }
    }
};

using StreamDenoise = StreamDenoiseT<ulk::Net>;

}  // namespace dns
