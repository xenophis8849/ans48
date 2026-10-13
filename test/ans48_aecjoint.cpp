// aec48 × ans48 联调（10-13）：真实回声素材上把两个库串起来跑几种接法，写出各路输出并报切换处的突变。
// 只在给了 ANS48_AEC48_DIR 时编（同 ans48_aecchain）。用法：
//   ans48_aecjoint <mic.pcm> <ref.pcm> <采样率> <输出目录>      （int16 单声道原始 PCM，mic 与 ref 等长对齐）
// 写出（int16 PCM，都已挪回 mic 的时间轴）：
//   A_aec       aec48 单独（不挂回调）
//   B_spec_g1   aec48 + 频谱档 γ1（挂在回调上）          C_spec_g2   同，γ2
//   D_serial_g1 aec48 输出合成成声音后再串推流档 γ1（旧接法，多 1024 样本延迟，已挪齐）  E_serial_g2  同，γ2
//   F_tog_gain  频谱档 γ1，每 2 s 在强度 1 ↔ 0 之间切（推荐的临时关法）
//   G_tog_hook  频谱档 γ1，每 2 s 用 aec48 的挂钩开关关 / 开（关的那段 ans48 收不到帧）
// 标准输出一行 JSON：各路电平；F / G 对「分段参照」（开的段取 B、关的段取 A）在每个切换点之后的偏差，
// 以及 G 重新打开后多久才持续回到与 B 相差 < −40 dB（旧记忆的影响时长；持续 0.5 s 以上才算，观察窗内没做到的计为未收敛）。
#include "aec48.h"
#include "ans48.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> rd(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    std::vector<short> s; short b[4096]; size_t n; while ((n = fread(b, 2, 4096, f)) > 0) s.insert(s.end(), b, b + n); fclose(f);
    std::vector<float> x(s.size()); for (size_t i = 0; i < s.size(); ++i) x[i] = s[i] / 32768.f; return x;
}
static void wr(const std::string& p, const std::vector<float>& y) {
    FILE* f = fopen(p.c_str(), "wb"); if (!f) { perror(p.c_str()); exit(2); }
    for (float v : y) { float t = v * 32767.f; short s = (short)(t > 32767.f ? 32767 : (t < -32768.f ? -32768 : lrintf(t))); fwrite(&s, 2, 1, f); }
    fclose(f);
}
static double rmsDb(const std::vector<float>& v, size_t a, size_t b) { double e = 0; for (size_t i = a; i < b && i < v.size(); ++i) e += (double)v[i] * v[i]; return 10 * log10(e / std::max<size_t>(1, b - a) + 1e-20); }

enum Tog { NONE, GAIN, HOOK };
// 跑 aec48（可挂频谱档），返回挪回 mic 时间轴的输出
static std::vector<float> runChain(const std::vector<float>& mic, const std::vector<float>& ref, int sr, Ans48* n, float g, Tog tog, int togBlocks) {
    const int B = 384; Aec48* a = aec48_create(sr, B); if (!a) { fprintf(stderr, "aec48_create 失败\n"); exit(2); }
    const int lat = aec48_latency_samples(a);
    if (n) { ans48_reset(n); ans48_set_strength(n, g); aec48_set_spec_hook(a, ans48_spec_frame, n); }
    std::vector<float> y; y.reserve(mic.size() + B); std::vector<float> out(B);
    for (size_t i = 0, k = 0; i + B <= mic.size(); i += B, ++k) {
        if (tog != NONE && k > 0 && k % togBlocks == 0) {
            const bool on = (k / togBlocks) % 2 == 0;
            if (tog == GAIN) ans48_set_strength(n, on ? g : 0.f); else aec48_set_spec_hook_enabled(a, on ? 1 : 0);
        }
        aec48_process(a, &mic[i], &ref[i], out.data(), B); y.insert(y.end(), out.begin(), out.end());
    }
    aec48_destroy(a);
    std::vector<float> z(y.begin() + std::min<size_t>(lat, y.size()), y.end()); return z;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "用法：%s mic.pcm ref.pcm 采样率 输出目录\n", argv[0]); return 2; }
    std::vector<float> mic = rd(argv[1]), ref = rd(argv[2]); const int sr = atoi(argv[3]); const std::string od = argv[4];
    const size_t n0 = std::min(mic.size(), ref.size()); mic.resize(n0); ref.resize(n0);   // 两路截到等长（盲测集个别条长度差几个样本）
    int fsr, nfft, hop, win; { Aec48* t = aec48_create(sr, 384); aec48_spec_format(t, &fsr, &nfft, &hop, &win); aec48_destroy(t); }
    Ans48* n = ans48_create_spec(fsr, nfft, hop, win); if (!n) { fprintf(stderr, "ans48_create_spec 失败\n"); return 2; }
    const int togBlocks = (int)(2.0 * sr / 384);   // 每 2 s 切一次

    const auto A = runChain(mic, ref, sr, nullptr, 1.f, NONE, 0);
    const auto B1 = runChain(mic, ref, sr, n, 1.f, NONE, 0);
    const auto C2 = runChain(mic, ref, sr, n, 2.f, NONE, 0);
    const auto F = runChain(mic, ref, sr, n, 1.f, GAIN, togBlocks);
    const auto G = runChain(mic, ref, sr, n, 1.f, HOOK, togBlocks);
    std::vector<float> D1, E2;
    for (float g : {1.f, 2.f}) {   // 旧接法：aec48 输出（波形）→ 推流档
        Ans48* s = ans48_create(sr, ANS48_MODE_STREAM); ans48_set_strength(s, g); const int L = ans48_latency_samples(s);
        std::vector<float> o(A.size()); ans48_process(s, A.data(), o.data(), (int)A.size()); ans48_destroy(s);
        std::vector<float> z(o.begin() + std::min<size_t>(L, o.size()), o.end()); (g == 1.f ? D1 : E2) = z;
    }
    wr(od + "/A_aec.pcm", A); wr(od + "/B_spec_g1.pcm", B1); wr(od + "/C_spec_g2.pcm", C2); wr(od + "/D_serial_g1.pcm", D1);
    wr(od + "/E_serial_g2.pcm", E2); wr(od + "/F_tog_gain.pcm", F); wr(od + "/G_tog_hook.pcm", G);

    // 切换分析：第 j 段（每段 togBlocks 块）开 = j 偶数。切换点 = 段边界（mic 时间轴，aec 延迟已挪掉；
    // 帧窗跨 1536 样本，故边界前后各 1536 样本算过渡带）
    const size_t seg = (size_t)togBlocks * 384, L = std::min({A.size(), B1.size(), F.size(), G.size()});
    auto devDb = [&](const std::vector<float>& x, const std::vector<float>& r, size_t a, size_t b) {
        double e = 0, s = 0; for (size_t i = a; i < b && i < L; ++i) { e += (double)(x[i] - r[i]) * (x[i] - r[i]); s += (double)r[i] * r[i]; } return 10 * log10(e / (s + 1e-20) + 1e-20); };
    double fSteady = -300, gSteady = -300, fJump = 0, gJump = 0, settleMax = 0; int nsw = 0, nReopen = 0, nUnsettled = 0;
    // 遍历每个实际发生的切换（最后一段不满 2 s 也算，只要切换点之后还有一个过渡带的样本）——10-13 Codex 审查指出原写法漏掉末次切换
    // 「后面还有没有切换」按实际处理过的输入长度判断，不按挪掉 aec 延迟后的输出长度 L：切换落在输入末尾几块里时已经生效，
    // 但它的输出过渡带被延迟截在 L 之外（10-13 Codex 复审指出）
    const size_t P = mic.size() / 384 * 384;
    for (size_t j = 1; j * seg < P && j * seg + 1536 < L; ++j) {
        const bool on = j % 2 == 0; const std::vector<float>& ref_ = on ? B1 : A;
        const size_t s0 = j * seg; const bool nextSw = (j + 1) * seg < P; const size_t s1 = std::min((j + 1) * seg, L);
        // 稳态：过渡带之后到段末。下一次切换后算出的第一帧，窗口覆盖切换点之前约 1000 个样本，所以有下一次切换时段尾要让开 1536 个样本
        // （10-13 实测：差异全在段尾这一截）；后面确实没有切换才不用让
        const size_t e1 = nextSw ? std::min((j + 1) * seg - 1536, L) : L;
        if (e1 > s0 + 1536 + 384) {
            fSteady = std::max(fSteady, devDb(F, ref_, s0 + 1536, e1));
            if (!on) gSteady = std::max(gSteady, devDb(G, A, s0 + 1536, e1));
        }
        // 突变：过渡带内相邻样本差的最大值 / 该处信号峰值（对比同位置参照自身的相邻差）
        auto jump = [&](const std::vector<float>& x) { double mx = 0, mr = 1e-9; for (size_t i = (s0 > 1536 ? s0 - 1536 : 1); i < s0 + 1536 && i < L; ++i) {
            mx = std::max(mx, (double)fabsf(x[i] - x[i - 1])); mr = std::max(mr, (double)std::max(fabsf(A[i] - A[i - 1]), fabsf(B1[i] - B1[i - 1]))); } return 20 * log10(mx / mr); };
        fJump = std::max(fJump, jump(F)); gJump = std::max(gJump, jump(G));
        if (on && e1 >= s0 + 1536 + (size_t)(0.5 * sr)) {
            // G 重新打开后的收敛：最早的 t，使 [t, 稳态末) 每个 384 样本块都低于 −40 dB，且这一段至少 0.5 s；
            // 找不到就记为未收敛（不再把观察窗口的终点当作收敛时刻——10-13 Codex 审查指出）
            ++nReopen; size_t t = e1; const size_t minHold = (size_t)(0.5 * sr);
            for (size_t u = e1; u >= s0 + 1536 + 384; u -= 384) { if (devDb(G, B1, u - 384, u) > -40) break; t = u - 384; }
            if (e1 - t >= minHold) settleMax = std::max(settleMax, (double)(t - s0) / sr); else ++nUnsettled;
        }
        ++nsw;
    }
    bool fin = true; for (const std::vector<float>* v : std::initializer_list<const std::vector<float>*>{&A, &B1, &C2, &D1, &E2, &F, &G}) for (float x : *v) if (!std::isfinite(x)) fin = false;
    printf("{\"samples\":%zu,\"finite\":%s,\"mic_db\":%.2f,\"A_db\":%.2f,\"B_db\":%.2f,\"C_db\":%.2f,\"D_db\":%.2f,\"E_db\":%.2f,"
           "\"switches\":%d,\"F_steady_vs_ref_db\":%.1f,\"G_off_vs_A_db\":%.1f,\"F_jump_db\":%.2f,\"G_jump_db\":%.2f,\"G_reopen\":%d,\"G_unsettled\":%d,\"G_settle_max_s\":%.3f}\n",
           L, fin ? "true" : "false", rmsDb(mic, 0, L), rmsDb(A, 0, L), rmsDb(B1, 0, L), rmsDb(C2, 0, L), rmsDb(D1, 0, std::min(L, D1.size())), rmsDb(E2, 0, std::min(L, E2.size())),
           nsw, fSteady, gSteady, fJump, gJump, nReopen, nUnsettled, settleMax);
    ans48_destroy(n); return 0;
}
