// aec48 × ans48 联调（10-13）：真实回声素材上把两个库串起来跑几种接法，写出各路输出并报切换处的突变。
// 只在给了 ANS48_AEC48_DIR 时编（同 ans48_aecchain）。用法：
//   ans48_aecjoint <mic.pcm> <ref.pcm> <采样率> <输出目录>      （int16 单声道原始 PCM，mic 与 ref 等长对齐）
// 写出（int16 PCM，都已挪回 mic 的时间轴）：
//   A_aec       aec48 单独（不挂回调）
//   B_spec_g1   aec48 + 频谱档 γ1（挂在回调上）          C_spec_g2   同，γ2
//   D_serial_g1 aec48 输出合成成声音后再串推流档 γ1（旧接法，多 1024 样本延迟，已挪齐）  E_serial_g2  同，γ2
//   F_tog_gain  频谱档 γ1，每 2 s 在强度 1 ↔ 0 之间切（推荐的临时关法）
//   G_tog_hook  频谱档 γ1，每 2 s 用 aec48 的挂钩开关关 / 开（关的那段 ans48 收不到帧）
// 标准输出一行 JSON：各路电平；实际执行 / 已分析的切换数；稳态逐样本（浮点位）比较；帧级核对（强度切换时每帧与一直开着逐位相同或原样返回）；
// 挂钩重开的收敛分类（收敛 / 未收敛 / 观察不足 / 近静音）。定义见 main 里切换分析一节。
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

// 帧级记录：包一层 ans48_spec_frame，逐帧记下进 / 出的频谱和当时的强度（10-13 挑战式审查后加：输出波形上的过渡是 aec48 合成时的重叠相加，
// 频谱档自己有没有出异常要在帧上核——强度切换时每一帧必须要么与一直开着那次的同一帧逐位相同，要么原样返回）
struct Rec { Ans48* n = nullptr; std::vector<float> in, out, g; long curBlk = -1; std::vector<long> blk; };   // blk：每次回调发生在第几块
static void recHook(void* u, float* re, float* im, int bins) {
    Rec* r = (Rec*)u; r->blk.push_back(r->curBlk); r->in.insert(r->in.end(), re, re + bins); r->in.insert(r->in.end(), im, im + bins);
    r->g.push_back(ans48_get_strength(r->n));
    ans48_spec_frame(r->n, re, im, bins);
    r->out.insert(r->out.end(), re, re + bins); r->out.insert(r->out.end(), im, im + bins);
}
// 跑 aec48（可挂频谱档），返回挪回 mic 时间轴的输出
static std::vector<float> runChain(const std::vector<float>& mic, const std::vector<float>& ref, int sr, Ans48* n, float g, Tog tog, int togBlocks, int* applied = nullptr, Rec* rec = nullptr) {
    const int B = 384; Aec48* a = aec48_create(sr, B); if (!a) { fprintf(stderr, "aec48_create 失败\n"); exit(2); }
    const int lat = aec48_latency_samples(a);
    if (n) { ans48_reset(n); ans48_set_strength(n, g);
        if (rec) { rec->n = n; rec->in.clear(); rec->out.clear(); rec->g.clear(); rec->blk.clear(); aec48_set_spec_hook(a, recHook, rec); } else aec48_set_spec_hook(a, ans48_spec_frame, n); }
    std::vector<float> y; y.reserve(mic.size() + B); std::vector<float> out(B);
    for (size_t i = 0, k = 0; i + B <= mic.size(); i += B, ++k) {
        if (tog != NONE && k > 0 && k % togBlocks == 0) {
            const bool on = (k / togBlocks) % 2 == 0;
            if (tog == GAIN) ans48_set_strength(n, on ? g : 0.f); else aec48_set_spec_hook_enabled(a, on ? 1 : 0);
            if (applied) ++*applied;   // 实际执行的切换，在执行时计数（10-13 挑战式审查：分析口径的计数会漏掉末尾的切换）
        }
        if (rec) rec->curBlk = (long)k;
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
    Rec recB, recF, recG;
    const auto B1 = runChain(mic, ref, sr, n, 1.f, NONE, 0, nullptr, &recB);
    const auto C2 = runChain(mic, ref, sr, n, 2.f, NONE, 0);
    int appliedF = 0, appliedG = 0;
    const auto F = runChain(mic, ref, sr, n, 1.f, GAIN, togBlocks, &appliedF, &recF);
    const auto G = runChain(mic, ref, sr, n, 1.f, HOOK, togBlocks, &appliedG, &recG);
    // 帧级核对
    const int BN = nfft / 2 + 1; const size_t nfB = recB.g.size(), nfF = recF.g.size(), nfG = recG.g.size();
    long frOn = 0, frOff = 0, frBad = 0;
    for (size_t c = 0; c < std::min(nfB, nfF); ++c) {
        const float* fo = &recF.out[c * 2 * BN]; const float* fi = &recF.in[c * 2 * BN]; const float* bo = &recB.out[c * 2 * BN]; const float* bi = &recB.in[c * 2 * BN];
        const bool inSame = !memcmp(fi, bi, 2 * BN * 4);   // 两次跑进来的频谱应相同（aec48 不受回调输出影响）
        if (recF.g[c] == 1.f) { ++frOn; if (!inSame || memcmp(fo, bo, 2 * BN * 4)) ++frBad; }
        else { ++frOff; if (memcmp(fo, fi, 2 * BN * 4)) ++frBad; }
    }
    // 挂钩开关的调用序列（10-13 挑战式审查：只比总数时，关着时多调一次、开着时少调一次会相互抵消）：
    // 逐块核对 G 的回调是否恰好发生在开着的块上（缺 / 多 / 重复分开报），并把每次回调进来的频谱与 B 同一块的逐位比。
    // 开关在第 k 块 process 之前设置；按 aec48 说明「下一块起生效」，这里以实测的序列为准：先按「第 k 块即生效」比，不符的另报。
    long gMissing = 0, gExtra = 0, gDup = 0, gInNeq = 0;
    {
        const long nb = (long)(mic.size() / 384);
        std::vector<long> bIdx(nb, -1); for (size_t c = 0; c < recB.blk.size(); ++c) if (recB.blk[c] >= 0 && recB.blk[c] < nb) bIdx[recB.blk[c]] = (long)c;
        std::vector<int> seen(nb, 0); for (long b : recG.blk) if (b >= 0 && b < nb) seen[b]++;
        for (long k = 1; k < nb; ++k) {
            const bool on = ((k / togBlocks) % 2) == 0;
            if (seen[k] > 1) gDup += seen[k] - 1;
            if (on && !seen[k]) ++gMissing;
            if (!on && seen[k]) ++gExtra;
        }
        for (size_t c = 0; c < recG.blk.size(); ++c) {
            const long b = recG.blk[c]; if (b < 0 || b >= nb || bIdx[b] < 0) { ++gInNeq; continue; }
            if (memcmp(&recG.in[c * 2 * BN], &recB.in[(size_t)bIdx[b] * 2 * BN], 2 * BN * 4)) ++gInNeq;
        }
    }
    const long expectG = [&] { long k = 0; const size_t nb = mic.size() / 384; for (size_t b = 1; b < nb; ++b) { if (((b / togBlocks) % 2) == 0) ++k; } return k; }();
    std::vector<float> D1, E2;
    for (float g : {1.f, 2.f}) {   // 旧接法：aec48 输出（波形）→ 推流档
        Ans48* s = ans48_create(sr, ANS48_MODE_STREAM); ans48_set_strength(s, g); const int L = ans48_latency_samples(s);
        std::vector<float> o(A.size()); ans48_process(s, A.data(), o.data(), (int)A.size()); ans48_destroy(s);
        std::vector<float> z(o.begin() + std::min<size_t>(L, o.size()), o.end()); (g == 1.f ? D1 : E2) = z;
    }
    wr(od + "/A_aec.pcm", A); wr(od + "/B_spec_g1.pcm", B1); wr(od + "/C_spec_g2.pcm", C2); wr(od + "/D_serial_g1.pcm", D1);
    wr(od + "/E_serial_g2.pcm", E2); wr(od + "/F_tog_gain.pcm", F); wr(od + "/G_tog_hook.pcm", G);

    // ---- 切换分析（10-13 挑战式审查后重写）----
    // 第 j 段（每段 togBlocks 块）开 = j 偶数；切换点 s0 = j·seg（mic 时间轴，aec 延迟已挪掉）。帧窗 1536 样本：
    // 切换后算出的第一帧覆盖切换点之前约 1000 个样本，所以过渡带取 [s0 − 1536, s0 + 1536)，稳态取过渡带之后到下一过渡带之前。
    //   逐样本相同：稳态区间里按浮点位逐个比（不是能量比），报比较样本数、不相等样本数、最大绝对误差；
    //   切换本身有没有引入异常：在帧上核（见 main 开头的帧级核对），输出波形上的过渡是 aec48 合成时的重叠相加。
    //   （曾试过「过渡带输出应落在开 / 关两路之间」的逐样本指标：重叠相加时每个样本由几帧叠成，正常过渡也不必落在两路之间，
    //     分不清毛刺与正常过渡，已删，10-13。）
    //   挂钩重开的收敛：块级（384 样本）误差能量相对 B 的同块能量 < −40 dB、并持续 ≥ 0.5 s 才算收敛；B 那块低于 −70 dBFS 的近静音块不参与判定。
    const size_t seg = (size_t)togBlocks * 384, L = std::min({A.size(), B1.size(), F.size(), G.size()}), P = mic.size() / 384 * 384;
    const double sil = pow(10.0, -70 / 10.0) * 384;   // 近静音块的能量门槛（−70 dBFS）
    auto blkE = [&](const std::vector<float>& x, const std::vector<float>& r, size_t a, double* er) { double e = 0, s = 0; for (size_t i = a; i < a + 384; ++i) { e += (double)(x[i] - r[i]) * (x[i] - r[i]); s += (double)r[i] * r[i]; } *er = e; return s; };
    long cmpN = 0, neqF = 0, neqG = 0; double maxF = 0, maxG = 0, settleMax = 0;
    int analyzed = 0, ro = 0, roConv = 0, roUnconv = 0, roShort = 0, roSilent = 0, roThin = 0; std::string winRows;
    for (size_t j = 1; j * seg < P; ++j) {
        const size_t s0 = j * seg; if (s0 + 1536 >= L) break;           // 输出里看不到这次切换之后的样本（被 aec 延迟截掉）：不分析，计数里已含
        ++analyzed;
        const bool on = j % 2 == 0; const std::vector<float>& ref_ = on ? B1 : A;
        const bool nextSw = (j + 1) * seg < P; const size_t e1 = nextSw ? std::min((j + 1) * seg - 1536, L) : L;
        // 稳态逐样本（浮点位）比
        if (e1 > s0 + 1536) for (size_t i = s0 + 1536; i < e1; ++i) {
            ++cmpN;
            if (memcmp(&F[i], &ref_[i], 4)) { ++neqF; maxF = std::max(maxF, (double)fabsf(F[i] - ref_[i])); }
            if (!on && memcmp(&G[i], &A[i], 4)) { ++neqG; maxG = std::max(maxG, (double)fabsf(G[i] - A[i])); }
        }
        // 挂钩重开的收敛
        if (on) {
            ++ro; const size_t hold = (size_t)(0.5 * sr);
            // 按时间窗记误差（浮点域；每窗累加非近静音块的误差能量与 B 能量；窗内没有可听块、或窗超出观察区间就记 null）。窗从过渡带之后算起。
            // 放在「观察不足」的提前退出之前：观察不到 0.5 s 的重开，已完整看到的前几个窗也要计入（10-13 Codex 审查）
            {
                static const double W[6] = {0.0, 0.1, 0.3, 0.6, 1.0, 1.5}; std::string row = "[";
                for (int w = 0; w < 5; ++w) {
                    const size_t x0 = s0 + 1536 + (size_t)(W[w] * sr), x1 = s0 + 1536 + (size_t)(W[w + 1] * sr); double E = 0, S2 = 0;
                    if (x1 <= e1) for (size_t u = x0; u + 384 <= x1; u += 384) { double er; const double sr_ = blkE(G, B1, u, &er); if (sr_ < sil) continue; E += er; S2 += sr_; }
                    char b[32]; if (x1 <= e1 && S2 > 0) snprintf(b, sizeof b, "%.2f", 10 * log10(E / S2 + 1e-30)); else snprintf(b, sizeof b, "null");
                    row += (w ? "," : ""); row += b;
                }
                winRows += (winRows.empty() ? "" : ",") + row + "]";
            }
            if (e1 < s0 + 1536 + hold) { ++roShort; continue; }
            // 从段尾往前找最长的「可听块全部达标」尾段。近静音块不参与判定，也不算进持续时长——持续时长只累计可听且达标的块
            // （10-13 Codex 审查：原写法把近静音尾段算进 0.5 s，一个可听块都没达标也能报收敛）
            size_t t = e1 - (e1 - s0 - 1536) % 384; bool anyLoud = false, failed = false; size_t conv = t, loudPass = 0;
            for (size_t u = t; u >= s0 + 1536 + 384; u -= 384) {
                double er; const double sr_ = blkE(G, B1, u - 384, &er);
                if (sr_ < sil) { conv = u - 384; continue; }          // 近静音块
                anyLoud = true; if (10 * log10(er / sr_ + 1e-30) > -40) { failed = true; break; }
                conv = u - 384; loudPass += 384;
            }
            if (!anyLoud) ++roSilent;
            else if (loudPass >= hold) { ++roConv; settleMax = std::max(settleMax, (double)(conv - s0) / sr); }
            else if (failed) ++roUnconv;
            else ++roThin;                                             // 可听块全都达标，但合计不到 0.5 s：证据不足
        }
    }
    // 各路电平在同一区间上比（旧接法扣掉 1024 样本延迟后尾部短一截；10-13 挑战式审查：区间不同会把内容差混进接法差）
    const size_t Lc = std::min({L, C2.size(), D1.size(), E2.size()});
    bool fin = true; for (const std::vector<float>* v : std::initializer_list<const std::vector<float>*>{&A, &B1, &C2, &D1, &E2, &F, &G}) for (float x : *v) if (!std::isfinite(x)) fin = false;
    printf("{\"samples\":%zu,\"level_samples\":%zu,\"finite\":%s,\"mic_db\":%.2f,\"A_db\":%.2f,\"B_db\":%.2f,\"C_db\":%.2f,\"D_db\":%.2f,\"E_db\":%.2f,"
           "\"switches_applied_F\":%d,\"switches_applied_G\":%d,\"switches_analyzed\":%d,"
           "\"steady_cmp\":%ld,\"steady_neq_F\":%ld,\"steady_maxabs_F\":%.3g,\"steady_neq_G_off\":%ld,\"steady_maxabs_G_off\":%.3g,"
           
           "\"reopen\":%d,\"reopen_converged\":%d,\"reopen_unconverged\":%d,\"reopen_short\":%d,\"reopen_silent\":%d,\"reopen_thin\":%d,\"settle_max_s\":%.3f,"
           "\"frames_B\":%zu,\"frames_F\":%zu,\"frames_F_on\":%ld,\"frames_F_off\":%ld,\"frames_F_bad\":%ld,\"frames_G\":%zu,\"frames_G_expect\":%ld,"
           "\"G_missing\":%ld,\"G_extra\":%ld,\"G_dup\":%ld,\"G_in_neq_B\":%ld,\"reopen_win_db\":[%s]}\n",
           L, Lc, fin ? "true" : "false", rmsDb(mic, 0, Lc), rmsDb(A, 0, Lc), rmsDb(B1, 0, Lc), rmsDb(C2, 0, Lc), rmsDb(D1, 0, Lc), rmsDb(E2, 0, Lc),
           appliedF, appliedG, analyzed, cmpN, neqF, maxF, neqG, maxG, ro, roConv, roUnconv, roShort, roSilent, roThin, settleMax,
           nfB, nfF, frOn, frOff, frBad, nfG, expectG, gMissing, gExtra, gDup, gInNeq, winRows.c_str());
    ans48_destroy(n); return 0;
}
