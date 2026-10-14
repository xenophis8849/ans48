// 回归测试：流式输出 FIFO 的读写计数器跨过回绕点时输出不中断。用法：ans48_wrap <in48k.pcm>
// 直接用库内部的流式处理类：把计数器设到 2^32 与 2^64 回绕点前，跨过回绕处理，与从 0 开始的实例逐样本比较。
#include "denoise_stage.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" { extern const unsigned char ans48_w1024_bin[]; extern const unsigned long ans48_w1024_bin_size; extern const unsigned char ans48_w1024_48k_json[];
             extern const unsigned char ans48_w256_bin[]; extern const unsigned long ans48_w256_bin_size; extern const unsigned char ans48_w256_48k_json[]; }
static int run(const unsigned char* bin, unsigned long n, const unsigned char* js, const std::vector<float>& x, uint64_t start, std::vector<float>& y, uint64_t* endR = nullptr, uint64_t* endW = nullptr) {
    ulk::Net net; if (!net.w.loadFromMemory(bin, n, (const char*)js) || !net.prepare()) return 1; net.reset();
    dns::StreamDenoise sd; sd.prepare(&net);
    sd.fifoR = start; sd.fifoW = start + (uint64_t)sd.hop;   // 与 reset() 同样预置一个 hop 的滞后，只是起点挪到回绕点前
    y.assign(x.size(), 0.f); for (size_t i = 0; i < x.size(); i += 480) sd.process(&x[i], &y[i], (int)std::min<size_t>(480, x.size() - i));
    if (endR) *endR = sd.fifoR; if (endW) *endW = sd.fifoW;
    return 0;
}
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s in48k.pcm\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 2; } std::vector<short> s(48000 * 3); size_t got = fread(s.data(), 2, s.size(), f); fclose(f);
    if (got < s.size()) { fprintf(stderr, "输入不足 3 s（%zu 样本）：覆盖不到回绕点，判失败\n", got); return 1; }
    std::vector<float> x(s.size()); for (size_t i = 0; i < s.size(); ++i) x[i] = s[i] / 32768.f;
    int fails = 0;
    struct M { const char* name; const unsigned char* bin; unsigned long n; const unsigned char* js; } ms[] = {{"推流", ans48_w1024_bin, ans48_w1024_bin_size, ans48_w1024_48k_json}, {"耳返", ans48_w256_bin, ans48_w256_bin_size, ans48_w256_48k_json}};
    for (const M& m : ms) {
        std::vector<float> ref, y; if (run(m.bin, m.n, m.js, x, 0, ref)) return 1;
        for (uint64_t start : {(uint64_t)0xFFFFFFFFull - 20000, (uint64_t)0xFFFFFFFFFFFFFFFFull - 20000}) {
            uint64_t eR = 0, eW = 0; if (run(m.bin, m.n, m.js, x, start, y, &eR, &eW)) return 1;
            double d = 0; for (size_t i = 0; i < x.size(); ++i) d = std::max(d, (double)std::fabs(y[i] - ref[i]));
            // 确实跨过了回绕点：2^32 那项读写位置都越过 2^32；2^64 那项读写位置都绕回了起点之前
            const bool crossed = start > 0xFFFFFFFFull ? (eR < start && eW < start) : (eR > 0xFFFFFFFFull && eW > 0xFFFFFFFFull);
            // 回绕点之后的输出不是全 0（回绕点在输出的第 20000 个样本附近）
            double e = 0; for (size_t i = 20000; i < y.size(); ++i) e += (double)y[i] * y[i];
            const bool ok = d == 0 && crossed && e > 1e-3; if (!ok) ++fails;
            if (!crossed) printf("  计数器没有跨过回绕点（读 %llu 写 %llu）\n", (unsigned long long)eR, (unsigned long long)eW);
            if (e <= 1e-3) printf("  回绕点之后输出近乎为 0（能量 %.3g）\n", e);
            printf("%s  %s：计数器从 %s 回绕点前 20000 样本起，%.1f s 输出与从 0 起逐样本%s\n", ok ? "通过  " : "不通过", m.name, start > 0xFFFFFFFFull ? "2^64" : "2^32", x.size() / 48000.0, ok ? "相同" : "不同");
        }
    }
    // 0.4.1：开头几帧计数器饱和在 nFft / hop（原来每帧递增，48 kHz 耳返约 66 天 int 溢出、退回开头分支）；reset 后归零
    for (const M& m : ms) {
        ulk::Net net; if (!net.w.loadFromMemory(m.bin, m.n, (const char*)m.js) || !net.prepare()) return 1; net.reset();
        dns::StreamDenoise sd; sd.prepare(&net); std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); i += 480) sd.process(&x[i], &y[i], (int)std::min<size_t>(480, x.size() - i));
        const int top = sd.nFft / sd.hop; const bool sat = sd.nFr == top; sd.reset(); const bool rz = sd.nFr == 0;
        if (!(sat && rz)) ++fails;
        printf("%s  %s：%.1f s 后开头帧计数 = %d（应饱和在 %d），reset 后 = 0：%s\n", sat && rz ? "通过  " : "不通过", m.name, x.size() / 48000.0, sat ? top : -1, top, rz ? "是" : "否");
    }
    printf("%s：不通过 %d 项\n", fails ? "有问题" : "全部通过", fails); return fails ? 1 : 0;
}
