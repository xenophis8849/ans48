// 频谱档自检（10-13）：按 aec48 联动接口契约版本 1 自己做分析 / 合成（与 aec48 同式：输入左补 768 个零后因果分帧、
// 周期 sqrt-Hann、不归一化 DFT；合成 IDFT/N × 窗、重叠相加、除以窗平方的重叠和），逐帧调 ans48_spec_frame，
// 结果与 Python 参照（mk_ans48_specvec.py）的波形比；另查接口边界。全部通过返回 0。
// 用法：ans48_speccheck <向量目录> <id>      （目录里要有 <id>_in48k.pcm、<id>_in441.pcm、<id>_spec_<率>_g<γ>.pcm）
#include "ans48.h"
#include "pffft.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int N = 1536, H = 384, PADL = 768, B = N / 2 + 1;
static int fails = 0;
static void check(bool ok, const std::string& what) { printf("%s  %s\n", ok ? "通过" : "不通过", what.c_str()); if (!ok) ++fails; }

static std::vector<float> readPcm(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("读不到 %s\n", p.c_str()); exit(2); }
    std::vector<short> s; short b[4096]; size_t n; while ((n = fread(b, 2, 4096, f)) > 0) s.insert(s.end(), b, b + n); fclose(f);
    std::vector<float> x(s.size()); for (size_t i = 0; i < s.size(); ++i) x[i] = s[i] / 32768.f; return x;
}
static short toS16(float x) { float v = x * 32767.f; return (short)(v > 32767.f ? 32767 : (v < -32768.f ? -32768 : lrintf(v))); }

// 按契约跑一遍：每帧调 fn（可为空 = 不改），返回合成后的输出（与输入同一时间轴、等长）
template <class Fn>
static std::vector<float> chain(const std::vector<float>& x, Fn fn) {
    const size_t L = x.size() / H * H;
    std::vector<float> w(N); for (int i = 0; i < N; ++i) w[i] = sqrtf(0.5f * (1.f - cosf(2.f * (float)M_PI * i / N)));
    PFFFT_Setup* s = pffft_new_setup(N, PFFFT_REAL);
    float* in = (float*)pffft_aligned_malloc(N * 4); float* out = (float*)pffft_aligned_malloc(N * 4); float* wk = (float*)pffft_aligned_malloc(N * 4);
    std::vector<double> num(PADL + L + N, 0.0), den(PADL + L + N, 0.0);
    std::vector<float> re(B), im(B);
    const size_t T = L / H - 1;   // 回调次数（第 1 块不调）
    for (size_t c = 0; c < T; ++c) {
        for (int i = 0; i < N; ++i) { const long j = (long)(c * H) + i - PADL; in[i] = (j >= 0 && j < (long)L ? x[j] : 0.f) * w[i]; }
        pffft_transform_ordered(s, in, out, wk, PFFFT_FORWARD);
        re[0] = out[0]; im[0] = 0; re[B - 1] = out[1]; im[B - 1] = 0;
        for (int k = 1; k < B - 1; ++k) { re[k] = out[2 * k]; im[k] = out[2 * k + 1]; }
        fn(re.data(), im.data());
        im[0] = 0; im[B - 1] = 0;                                   // 契约：回调后直流 / 奈奎斯特虚部置 0
        in[0] = re[0]; in[1] = re[B - 1]; for (int k = 1; k < B - 1; ++k) { in[2 * k] = re[k]; in[2 * k + 1] = im[k]; }
        pffft_transform_ordered(s, in, out, wk, PFFFT_BACKWARD);
        for (int i = 0; i < N; ++i) { num[c * H + i] += (double)w[i] * out[i] / N; den[c * H + i] += (double)w[i] * w[i]; }
    }
    pffft_aligned_free(in); pffft_aligned_free(out); pffft_aligned_free(wk); pffft_destroy_setup(s);
    std::vector<float> y(L); for (size_t t = 0; t < L; ++t) { const double d = den[t + PADL]; y[t] = (float)(num[t + PADL] / (d < 1e-11 ? 1e-11 : d)); }
    return y;
}

// 中段（跳过首 PADL、末 N 个样本：窗覆盖不全）对期望 int16：差值能量 dB、最大差 LSB
static void cmp(const std::vector<float>& y, const std::vector<float>& e, const std::string& what, double dbMax, int lsbMax) {
    const size_t L = std::min(y.size(), e.size()); double de = 0, ee = 0; int mx = 0;
    for (size_t t = PADL; t + N < L; ++t) { const int a = toS16(y[t]), b = (int)lrintf(e[t] * 32768.f); const int d = abs(a - b); if (d > mx) mx = d;
        const double v = (double)a - b; de += v * v; ee += (double)b * b; }
    const double db = 10 * log10(de / (ee + 1e-30) + 1e-30);
    char buf[256]; snprintf(buf, sizeof buf, "%s：差值能量 %.1f dB，最大差 %d LSB", what.c_str(), db, mx);
    check(db < dbMax && mx <= lsbMax, buf);
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("用法：%s <向量目录> <id>\n", argv[0]); return 2; }
    const std::string dir = argv[1], id = argv[2];
    check(ANS48_SPEC_CONTRACT_VERSION == 1, "契约版本 = 1（aec48 README「⚠ 联动接口」）");
    // 参数核对：与契约不符一律 NULL
    check(!ans48_create_spec(48000, 1024, 384, 1) && !ans48_create_spec(48000, 1536, 512, 1) && !ans48_create_spec(48000, 1536, 384, 0) &&
          !ans48_create_spec(32000, 1536, 384, 1), "格式与契约不符的参数一律返回 NULL");
    for (int sr : {48000, 44100}) {
        const std::string tag = sr == 48000 ? "48k" : "441";
        const std::vector<float> x = readPcm(dir + "/" + id + "_in" + (sr == 48000 ? "48k" : "441") + ".pcm");
        Ans48* a = ans48_create_spec(sr, 1536, 384, ANS48_WINDOW_SQRT_HANN_PERIODIC);
        check(a != nullptr, std::to_string(sr) + " Hz 建频谱档实例");
        if (!a) continue;
        check(ans48_latency_samples(a) == 0 && fabsf(ans48_max_strength(a) - 2.f) < 1e-6f, "延迟 0、强度上限 2");
        float f1 = 0, f2 = 0; check(ans48_process(a, &f1, &f2, 1) == -1, "频谱档上调 ans48_process 返回 -1");
        // bins 不符 / 空指针：原样返回
        std::vector<float> r(B, 0.5f), i2(B, 0.25f); ans48_spec_frame(a, r.data(), i2.data(), B - 1); ans48_spec_frame(nullptr, r.data(), i2.data(), B);
        check(r[10] == 0.5f && i2[10] == 0.25f, "bins 不符或 user 为空时频谱原样返回");
        for (float g : {1.f, 2.f, 0.5f}) {
            ans48_reset(a); ans48_set_strength(a, g);
            const std::vector<float> y = chain(x, [&](float* re, float* im) { ans48_spec_frame(a, re, im, B); });
            char gs[16]; snprintf(gs, sizeof gs, "%g", g);
            cmp(y, readPcm(dir + "/" + id + "_spec_" + tag + "_g" + gs + ".pcm"), std::to_string(sr) + " Hz γ" + gs + " 对 Python 参照", -90.0, 1);
        }
        // γ = 0：输出与「不挂回调」逐样本相同（网络照跑，掩码^0 = 1）
        ans48_reset(a); ans48_set_strength(a, 0.f);
        const std::vector<float> y0 = chain(x, [&](float* re, float* im) { ans48_spec_frame(a, re, im, B); });
        const std::vector<float> yb = chain(x, [](float*, float*) {});
        check(memcmp(y0.data(), yb.data(), y0.size() * 4) == 0, "γ = 0 时输出与不挂回调逐样本相同");
        // 复位后与新实例相同
        ans48_set_strength(a, 1.f); ans48_reset(a);
        const std::vector<float> ya = chain(x, [&](float* re, float* im) { ans48_spec_frame(a, re, im, B); });
        Ans48* b = ans48_create_spec(sr, 1536, 384, 1);
        const std::vector<float> yn = chain(x, [&](float* re, float* im) { ans48_spec_frame(b, re, im, B); });
        check(memcmp(ya.data(), yn.data(), ya.size() * 4) == 0, "reset 后与新实例逐样本相同");
        double st[3]; ans48_stats(a, st);
        printf("      统计：每帧平均 %.3f ms，最大 %.3f ms，已处理 %.0f 样本\n", st[0], st[1], st[2]);
        ans48_destroy(b); ans48_destroy(a);
    }
    printf("%s：不通过 %d 项\n", fails ? "有不通过" : "全部通过", fails);
    return fails ? 1 : 0;
}
