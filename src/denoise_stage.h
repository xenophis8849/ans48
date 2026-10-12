// 降噪的流式外圈：分帧 → FFT → 网络（手写内核）→ 掩码^γ → 逆变换 → 重叠相加。逐样本推进，处理期零分配，可在音频回调里调用。
// 固有延迟 = 窗长：开头预置一个 hop 的滞后，加上一帧要凑满窗（窗 256 在 48 kHz 下 5.3 ms，窗 1024 为 21.3 ms）。
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "ulunas_kernel.h"
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

struct StreamDenoise {
    ulk::Net* net = nullptr;
    int nFft = 256, hop = 128, bins = 129;
    float gamma = 1.0f;
    RFft fft;
    std::vector<float> win, inbuf, num, den, re, im, mag, mask, fifo;
    int fill = 0;          // 当前 hop 已攒了几个样本
    uint64_t fifoR = 0, fifoW = 0;   // 按样本累计的读 / 写位置。64 位：32 位平台上 size_t 约 24.9 h（48 kHz）就回绕，旧的 fifoR < fifoW 判断会从此恒假、持续输出 0

    void prepare(ulk::Net* n_) {
        net = n_;
        nFft = net->w.nFft; hop = net->w.hop; bins = net->bins;
        fft.init(nFft);
        win.resize(nFft);
        for (int i = 0; i < nFft; ++i)
            win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / nFft);
        inbuf.assign(nFft, 0.0f);
        num.assign(nFft, 0.0f); den.assign(nFft, 0.0f);
        re.assign(nFft, 0.0f); im.assign(nFft, 0.0f);
        mag.assign(bins, 0.0f); mask.assign(bins, 0.0f);
        fifo.assign(1 << 13, 0.0f);
        reset();
    }

    void reset() {
        if (net) net->reset();
        std::fill(inbuf.begin(), inbuf.end(), 0.0f);
        std::fill(num.begin(), num.end(), 0.0f);
        std::fill(den.begin(), den.end(), 0.0f);
        std::fill(fifo.begin(), fifo.end(), 0.0f);
        fill = 0; fifoR = 0; fifoW = (size_t)hop;   // 预置一个 hop 的固有滞后
    }

    inline void frame() {
        const int M = nFft / 2;
        for (int i = 0; i < nFft; ++i) fft.in[i] = inbuf[i] * win[i];
        pffft_transform_ordered(fft.s, fft.in, fft.out, fft.wk, PFFFT_FORWARD);
        re[0] = fft.out[0]; im[0] = 0.0f; re[M] = fft.out[1]; im[M] = 0.0f;
        for (int k = 1; k < M; ++k) { re[k] = fft.out[2 * k]; im[k] = fft.out[2 * k + 1]; }
        for (int k = 0; k < bins; ++k) {
            float m = sqrtf(re[k] * re[k] + im[k] * im[k]);
            mag[k] = log10f(m < 1e-12f ? 1e-12f : m);
        }
        net->run(mag.data(), mask.data());
        for (int k = 0; k < bins; ++k) {
            float v = mask[k];
            if (v < 0.0f) v = 0.0f; else if (v > 1.0f) v = 1.0f;
            if (gamma != 1.0f) v = powf(v, gamma);
            re[k] *= v; im[k] *= v;
        }
        // 逆实数 FFT（DC 与奈奎斯特频点的虚部按定义为 0，同 torch.fft.irfft）
        fft.in[0] = re[0]; fft.in[1] = re[M];
        for (int k = 1; k < M; ++k) { fft.in[2 * k] = re[k]; fft.in[2 * k + 1] = im[k]; }
        pffft_transform_ordered(fft.s, fft.in, fft.out, fft.wk, PFFFT_BACKWARD);
        const float* tf = fft.out;
        const float inv = 1.0f / nFft;
        // 重叠相加：分子 w·ifft、分母 w²、相除。与离线逐字同式。
        memmove(num.data(), num.data() + hop, sizeof(float) * (nFft - hop));
        memmove(den.data(), den.data() + hop, sizeof(float) * (nFft - hop));
        std::fill(num.end() - hop, num.end(), 0.0f);
        std::fill(den.end() - hop, den.end(), 0.0f);
        for (int i = 0; i < nFft; ++i) {
            num[i] += win[i] * tf[i] * inv;
            den[i] += win[i] * win[i];
        }
        const size_t mask_ = fifo.size() - 1;
        for (int i = 0; i < hop; ++i) {
            float d = den[i] < 1e-11f ? 1e-11f : den[i];
            fifo[(size_t)((fifoW + i) & mask_)] = num[i] / d;
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

}  // namespace dns
