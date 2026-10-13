// 与 aec48 串联核对（10-13，ANS48_SPEC_TIER_HANDOFF.md §2.3）。两个库互不链接，只有这个测试程序同时链两个，
// 编法：cmake -DANS48_AEC48_DIR=<aec48-android 目录>（要有 include/aec48.h 与 build-host/libaec48.so）。
//   1. aec48 直通（aec48_set_enabled 0）+ 挂频谱档：回调里的频谱 = 麦克风按契约分帧的频谱，输出应等于
//      Python 参照（mk_ans48_specvec.py，同一契约）再晚 aec48_latency_samples 个样本；
//   2. aec48 开（参考信号为零）：频谱档强度 0 时与不挂回调逐样本相同；
//   3. 每块耗时（aec48_stats）：不挂 / 挂频谱档，平均与最大。
// 用法：ans48_aecchain <向量目录> <id>
#include "aec48.h"
#include "ans48.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if AEC48_SPEC_CONTRACT_VERSION != ANS48_SPEC_CONTRACT_VERSION
#error "aec48 与 ans48 的联动接口契约版本不一致"
#endif

static int fails = 0;
static void check(bool ok, const std::string& w) { printf("%s  %s\n", ok ? "通过" : "不通过", w.c_str()); if (!ok) ++fails; }
static std::vector<float> readPcm(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("读不到 %s\n", p.c_str()); exit(2); }
    std::vector<short> s; short b[4096]; size_t n; while ((n = fread(b, 2, 4096, f)) > 0) s.insert(s.end(), b, b + n); fclose(f);
    std::vector<float> x(s.size()); for (size_t i = 0; i < s.size(); ++i) x[i] = s[i] / 32768.f; return x;
}
static short toS16(float x) { float v = x * 32767.f; return (short)(v > 32767.f ? 32767 : (v < -32768.f ? -32768 : lrintf(v))); }

struct Run { std::vector<float> y; std::vector<double> blockMs; };
static Run runAec(const std::vector<float>& mic, int sr, bool aecOn, Ans48* hook) {
    const int blk = 384; Aec48* a = aec48_create(sr, blk); Run r;
    if (!a) { printf("aec48_create 失败\n"); exit(2); }
    aec48_set_enabled(a, aecOn ? 1 : 0);
    if (hook) check(aec48_set_spec_hook(a, ans48_spec_frame, hook) == 0, "aec48_set_spec_hook 挂上频谱档");
    std::vector<float> ref(blk, 0.f), out(blk); r.y.reserve(mic.size());
    for (size_t i = 0; i + blk <= mic.size(); i += blk) {
        const auto t0 = std::chrono::steady_clock::now();
        aec48_process(a, &mic[i], ref.data(), out.data(), blk);
        r.blockMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        r.y.insert(r.y.end(), out.begin(), out.end());
    }
    aec48_destroy(a); return r;
}
static void timing(const char* nm, std::vector<double> v) {
    std::sort(v.begin(), v.end()); double s = 0; for (double d : v) s += d;
    printf("      %s：每块平均 %.3f ms，p99 %.3f ms，最大 %.3f ms（%zu 块）\n", nm, s / v.size(), v[v.size() * 99 / 100], v.back(), v.size());
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("用法：%s <向量目录> <id>\n", argv[0]); return 2; }
    const std::string dir = argv[1], id = argv[2];
    for (int sr : {48000, 44100}) {
        const std::string tag = sr == 48000 ? "48k" : "441";
        const std::vector<float> x = readPcm(dir + "/" + id + "_in" + tag + ".pcm");
        int fsr, nfft, hop, win; { Aec48* t = aec48_create(sr, 384); aec48_spec_format(t, &fsr, &nfft, &hop, &win); aec48_destroy(t); }
        Ans48* n = ans48_create_spec(fsr, nfft, hop, win);
        check(n != nullptr, std::to_string(sr) + " Hz：按 aec48_spec_format 建频谱档（" + std::to_string(nfft) + " / " + std::to_string(hop) + " / 窗 " + std::to_string(win) + "）");
        if (!n) continue;
        int lat; { Aec48* t = aec48_create(sr, 384); lat = aec48_latency_samples(t); aec48_destroy(t); }
        // 1. 直通 + 频谱档 γ1 对 Python 参照（晚 lat 个样本）
        ans48_set_strength(n, 1.f); ans48_reset(n);
        const Run r1 = runAec(x, sr, false, n);
        const std::vector<float> e = readPcm(dir + "/" + id + "_spec_" + tag + "_g1.pcm");
        double de = 0, ee = 0; int mx = 0;
        for (size_t t = 768; t + 1536 < e.size() && t + lat < r1.y.size(); ++t) {
            const int a = toS16(r1.y[t + lat]), b = (int)lrintf(e[t] * 32768.f), d = abs(a - b); if (d > mx) mx = d; de += (double)(a - b) * (a - b); ee += (double)b * b; }
        char buf[200]; snprintf(buf, sizeof buf, "%s 直通 + 频谱档 γ1 对 Python 参照（晚 %d 样本）：差值能量 %.1f dB，最大差 %d LSB", tag.c_str(), lat, 10 * log10(de / ee + 1e-30), mx);
        check(10 * log10(de / ee + 1e-30) < -90 && mx <= 1, buf);
        // 2. AEC 开：强度 0 与不挂逐样本相同
        const Run r0 = runAec(x, sr, true, nullptr);
        ans48_set_strength(n, 0.f); ans48_reset(n);
        const Run rz = runAec(x, sr, true, n);
        check(r0.y.size() == rz.y.size() && memcmp(r0.y.data(), rz.y.data(), r0.y.size() * 4) == 0, tag + " AEC 开：频谱档强度 0 与不挂回调逐样本相同");
        // 3. 耗时（主机读数，只看量级；手机另测）
        ans48_set_strength(n, 1.f); ans48_reset(n);
        const Run r2 = runAec(x, sr, true, n);
        timing((tag + " AEC 开、不挂").c_str(), r0.blockMs); timing((tag + " AEC 开、挂频谱档 γ1").c_str(), r2.blockMs);
        bool fin = true; for (float v : r2.y) if (!std::isfinite(v)) fin = false;
        check(fin, tag + " AEC 开 + 频谱档：输出全是有限数");
        ans48_destroy(n);
    }
    printf("%s：不通过 %d 项\n", fails ? "有不通过" : "全部通过", fails);
    return fails ? 1 : 0;
}
