// 接口健壮性测试。主机与手机都能跑：ans48_robust <in48k.pcm> <in441.pcm>
// 每项打印"通过 / 不通过"，全部通过返回 0。主机版用替换 operator new 的方式做分配计数，第一项先确认这种替换能拦到库内部的分配。
#include "ans48.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

static std::atomic<long> g_news{0}; static std::atomic<bool> g_count{false};
void* operator new(size_t n) { if (g_count.load(std::memory_order_relaxed)) g_news.fetch_add(1); void* p = malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void operator delete(void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }

static int fails = 0;
static void check(bool ok, const std::string& what, const std::string& detail = "") { printf("%s  %s%s%s\n", ok ? "通过  " : "不通过", what.c_str(), detail.empty() ? "" : "：", detail.c_str()); if (!ok) ++fails; }
static std::vector<float> rdf(const char* p) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(2); } fseek(f, 0, SEEK_END); long n = ftell(f) / 2; fseek(f, 0, SEEK_SET); std::vector<short> v(n); if (fread(v.data(), 2, n, f) != (size_t)n) exit(2); fclose(f); std::vector<float> o(n); for (long i = 0; i < n; ++i) o[i] = v[i] / 32768.f; return o; }
static long rss_kb() { FILE* f = fopen("/proc/self/status", "r"); char l[256]; long v = -1; while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6); if (f) fclose(f); return v; }
static double maxdiff(const std::vector<float>& a, const std::vector<float>& b) { double d = 0; for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i])); return d; }
static double energy(const std::vector<float>& a, size_t from = 0) { double e = 0; for (size_t i = from; i < a.size(); ++i) e += (double)a[i] * a[i]; return e; }
// 用块长 blk 处理整段（可原地），返回输出
static std::vector<float> run(const std::vector<float>& x, int sr, int mode, int blk, float g = 1.f, bool inplace = false) {
    Ans48* a = ans48_create(sr, mode); ans48_set_strength(a, g); std::vector<float> o(x.size()), buf;
    for (size_t i = 0; i < x.size(); i += blk) { int c = (int)std::min<size_t>(blk, x.size() - i);
        if (inplace) { buf.assign(x.begin() + i, x.begin() + i + c); ans48_process(a, buf.data(), buf.data(), c); memcpy(&o[i], buf.data(), c * 4); }
        else ans48_process(a, &x[i], &o[i], c); }
    ans48_destroy(a); return o;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "用法：%s in48k.pcm in441.pcm\n", argv[0]); return 2; }
    auto x48 = rdf(argv[1]), x44 = rdf(argv[2]);
    printf("%s\n", ans48_version());
    struct Cfg { int sr, mode; const char* name; const std::vector<float>* x; };
    const Cfg cfgs[] = {{48000, ANS48_MODE_MONITOR, "耳返 48k", &x48}, {44100, ANS48_MODE_MONITOR, "耳返 44.1k", &x44}, {48000, ANS48_MODE_STREAM, "推流 48k", &x48}, {44100, ANS48_MODE_STREAM, "推流 44.1k", &x44}};

    // 前提：测试程序能拦到库内部的分配
    { g_news = 0; g_count = true; Ans48* a = ans48_create(48000, ANS48_MODE_STREAM); g_count = false; ans48_destroy(a);
      check(g_news.load() > 0, "前提：测试程序能拦到库内部的内存分配", "创建一次 operator new " + std::to_string(g_news.load()) + " 次"); }

    // 不支持的参数 → NULL；空指针与非法长度 → -1；空实例上的查询不崩
    check(!ans48_create(16000, 0) && !ans48_create(48000, 2) && !ans48_create(48000, -1), "不支持的采样率 / 档位返回 NULL");
    { Ans48* a = ans48_create(48000, 0); float b[4] = {0}; short s[4] = {0};
      check(ans48_process(a, nullptr, b, 4) == -1 && ans48_process(a, b, b, 0) == -1 && ans48_process_s16(a, s, nullptr, 4) == -1 && ans48_process(nullptr, b, b, 4) == -1, "空指针 / 长度 ≤0 返回 -1");
      ans48_destroy(a); ans48_destroy(nullptr); double st[3]; ans48_stats(nullptr, st);
      check(ans48_latency_samples(nullptr) == 0 && ans48_get_strength(nullptr) == 0.f, "对 NULL 实例的查询不崩、返回 0"); }

    for (const Cfg& c : cfgs) {
        const std::vector<float>& x = *c.x; const std::string nm = c.name;
        Ans48* a = ans48_create(c.sr, c.mode); const int L = ans48_latency_samples(a); const float mx = ans48_max_strength(a); const float g0 = ans48_get_strength(a); ans48_destroy(a);
        const int want = c.mode == ANS48_MODE_MONITOR ? 256 : 1024;
        check(L == want && mx == 2.f   /* 0.3.0 起两档上限都是 2（推流档原为 1） */, nm + "：延迟 = 窗长、强度上限符合档位", "延迟 " + std::to_string(L) + "，上限 " + std::to_string(mx));
        check(g0 == 1.f, nm + "：不设强度时默认 1", "默认 " + std::to_string(g0));   // 10-14 需求方：落端方不指定时三档默认强度都是 1

        // 延迟实测：强度 0 时输出 = 输入挪后 L 个样本
        { auto o = run(x, c.sr, c.mode, 480, 0.f); double d = 0; for (size_t i = 0; i + L < x.size(); ++i) d = std::max(d, (double)std::fabs(o[i + L] - x[i]));
          double head = 0; for (int i = 0; i < L; ++i) head = std::max(head, (double)std::fabs(o[i]));
          check(d < 1e-5 && head < 1e-3, nm + "：强度 0 → 输出 = 输入延迟 " + std::to_string(L) + " 样本（启动段 < −60 dBFS）", "最大差 " + std::to_string(d) + "，开头最大 " + std::to_string(head)); }

        // 任意块长、原地处理与逐 hop 处理逐样本相同
        { auto ref = run(x, c.sr, c.mode, c.mode == ANS48_MODE_MONITOR ? 128 : 512); double worst = 0; std::string why;
          for (int blk : {1, 7, 96, 383, 480, 1000, 4096, (int)x.size()}) for (int ip = 0; ip < 2; ++ip) { auto o = run(x, c.sr, c.mode, blk, 1.f, ip == 1); double d = maxdiff(o, ref); if (d > worst) { worst = d; why = "块长 " + std::to_string(blk) + (ip ? " 原地" : ""); } }
          check(worst == 0, nm + "：块长 1 / 7 / 96 / 383 / 480 / 1000 / 4096 / 整段，原地与否，输出逐样本相同", worst == 0 ? "" : "最大差 " + std::to_string(worst) + "（" + why + "）"); }

        // int16 接口与 float 一致（≤1 LSB），含原地
        { std::vector<short> s(x.size()); for (size_t i = 0; i < x.size(); ++i) s[i] = (short)lrintf(x[i] * 32768.f);
          std::vector<float> xf(x.size()); for (size_t i = 0; i < x.size(); ++i) xf[i] = s[i] / 32768.f;
          auto ref = run(xf, c.sr, c.mode, 480); long worst = 0;
          for (int ip = 0; ip < 2; ++ip) { Ans48* b = ans48_create(c.sr, c.mode); std::vector<short> o(s.size());
            for (size_t i = 0; i < s.size(); i += 333) { int k = (int)std::min<size_t>(333, s.size() - i); if (ip) { memcpy(&o[i], &s[i], k * 2); ans48_process_s16(b, &o[i], &o[i], k); } else ans48_process_s16(b, &s[i], &o[i], k); }
            ans48_destroy(b);
            for (size_t i = 0; i < s.size(); ++i) { float v = ref[i] * 32767.f; long q = (long)(v > 32767.f ? 32767 : (v < -32768.f ? -32768 : lrintf(v))); worst = std::max(worst, std::labs(q - o[i])); } }
          check(worst <= 1, nm + "：int16 接口与 float 一致（含原地）", "最大差 " + std::to_string(worst) + " LSB"); }

        // 强度：越大压得越多（输出能量单调不增）；超范围截断；NaN 忽略
        { auto o1 = run(x, c.sr, c.mode, 480, 1.f), oh = run(x, c.sr, c.mode, 480, 0.5f), om = run(x, c.sr, c.mode, 480, mx);
          double e0 = energy(x), eh = energy(oh, L), e1 = energy(o1, L), em = energy(om, L);
          check(eh <= e0 * 1.0001 && e1 <= eh && em <= e1, nm + "：强度 0.5 / 1 / 上限 输出能量依次不增", std::to_string(10 * log10(eh / e0)) + " / " + std::to_string(10 * log10(e1 / e0)) + " / " + std::to_string(10 * log10(em / e0)) + " dB");
          Ans48* b = ans48_create(c.sr, c.mode); float r1 = ans48_set_strength(b, 9.f), r2 = ans48_set_strength(b, -3.f), r3 = ans48_set_strength(b, 0.7f), r4 = ans48_set_strength(b, NAN); ans48_destroy(b);
          check(r1 == mx && r2 == 0.f && std::fabs(r3 - 0.7f) < 1e-6 && std::fabs(r4 - 0.7f) < 1e-6, nm + "：强度超上限截到上限、负数截到 0、NaN 忽略"); }

        // reset 后与新实例逐样本相同
        { Ans48* b = ans48_create(c.sr, c.mode); std::vector<float> o(x.size()); ans48_process(b, x.data(), o.data(), (int)std::min<size_t>(x.size(), 48000)); ans48_reset(b);
          for (size_t i = 0; i < x.size(); i += 480) ans48_process(b, &x[i], &o[i], (int)std::min<size_t>(480, x.size() - i)); ans48_destroy(b);
          check(maxdiff(o, run(x, c.sr, c.mode, 480)) == 0, nm + "：reset 之后与新实例输出逐样本相同"); }

        // 处理期不分配内存（float 与 int16）
        { Ans48* b = ans48_create(c.sr, c.mode); std::vector<float> o(480); std::vector<short> s(480), so(480);
          ans48_process(b, x.data(), o.data(), 480);
          g_news = 0; g_count = true;
          for (size_t i = 0; i + 480 <= x.size(); i += 480) { ans48_process(b, &x[i], o.data(), 480); ans48_process_s16(b, s.data(), so.data(), 480); ans48_set_strength(b, 0.5f + (i % 3) * 0.3f); }
          g_count = false; ans48_destroy(b);
          check(g_news.load() == 0, nm + "：处理期 operator new 次数为 0（float / int16 / 改强度）", std::to_string(g_news.load())); }
    }

    // 反复创建 / 销毁不漏内存
    { for (int i = 0; i < 20; ++i) ans48_destroy(ans48_create(48000, ANS48_MODE_STREAM)); long r0 = rss_kb();
      for (int i = 0; i < 400; ++i) { ans48_destroy(ans48_create(48000, ANS48_MODE_STREAM)); ans48_destroy(ans48_create(44100, ANS48_MODE_MONITOR)); } long r1 = rss_kb();
      check(r1 - r0 < 2048, "800 次创建 / 销毁后常驻内存增长 < 2 MB", std::to_string(r1 - r0) + " KB"); }

    // 8 线程同时创建、处理，输出逐样本相同
    { std::vector<std::vector<float>> outs(8); std::vector<std::thread> th; std::vector<float> seg(x48.begin(), x48.begin() + std::min<size_t>(x48.size(), 96000));
      for (int k = 0; k < 8; ++k) th.emplace_back([&, k] { outs[k] = run(seg, 48000, k % 2, 480); });
      for (auto& t : th) t.join(); bool ok = true; for (int k = 2; k < 8; ++k) ok = ok && maxdiff(outs[k % 2], outs[k]) == 0;
      check(ok, "8 线程并发创建与处理，同档位各实例输出逐样本相同"); }

    // 处理中从另一线程改强度：不崩、生效值始终在范围内
    { Ans48* b = ans48_create(48000, ANS48_MODE_MONITOR); std::atomic<bool> go{true}; std::atomic<int> bad{0};
      std::thread t([&] { int k = 0; while (go) { float r = ans48_set_strength(b, (k++ % 50) * 0.1f - 1.f); if (r < 0.f || r > 2.f) bad++; } });
      std::vector<float> o(480); for (int r = 0; r < 3; ++r) for (size_t i = 0; i + 480 <= x48.size(); i += 480) ans48_process(b, &x48[i], o.data(), 480);
      go = false; t.join(); float g = ans48_get_strength(b); ans48_destroy(b);
      check(bad == 0 && g >= 0.f && g <= 2.f, "处理中另一线程反复改强度：不崩、生效值始终在 [0, 2]"); }

    printf("%s：不通过 %d 项\n", fails ? "有问题" : "全部通过", fails); return fails ? 1 : 0;
}
