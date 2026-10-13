#include "ulunas_kernel.h"
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace ulk {

// ---- 极简 JSON 取数（只够读权重清单，不是通用解析器）--------
// 清单是我们自己生成的、结构固定，所以不引第三方库。**只在加载期跑一次。**
namespace {

struct J {
    const std::string& s;
    size_t i = 0;
    explicit J(const std::string& t) : s(t) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t')) ++i; }
    bool lit(char c) { ws(); if (i < s.size() && s[i]==c) { ++i; return true; } return false; }
    std::string str() {
        ws(); if (i >= s.size() || s[i] != '"') return {};
        ++i; std::string o;
        while (i < s.size() && s[i] != '"') { if (s[i]=='\\') ++i; o += s[i++]; }
        ++i; return o;
    }
    long num() { ws(); char* e = nullptr; long v = strtol(s.c_str()+i, &e, 10); i = e - s.c_str(); return v; }
};

}  // namespace

static bool parseManifest(const std::string& js, Weights* w) {
    // 找 "tensors": [ {name,shape,offset,count}, ... ]，其余键单独找。
    auto findInt = [&](const char* key, int dflt) {
        size_t p = js.find(std::string("\"") + key + "\"");
        if (p == std::string::npos) return dflt;
        p = js.find(':', p); if (p == std::string::npos) return dflt;
        return (int)strtol(js.c_str() + p + 1, nullptr, 10);
    };
    w->nFft = findInt("n_fft", 256);
    w->hop  = findInt("hop", 128);
    w->fs   = findInt("fs", 48000);
    w->bins = findInt("bins", 129);

    size_t p = js.find("\"tensors\"");
    if (p == std::string::npos) return false;
    J j(js); j.i = js.find('[', p) + 1;
    while (true) {
        j.ws();
        if (j.i >= js.size() || js[j.i] == ']') break;
        if (!j.lit('{')) break;
        std::string name; std::vector<int> shape; long off = 0, cnt = 0;
        while (true) {
            std::string k = j.str();
            if (k.empty()) break;
            j.lit(':');
            if (k == "name") name = j.str();
            else if (k == "shape") {
                j.lit('[');
                while (!j.lit(']')) { shape.push_back((int)j.num()); j.lit(','); }
            } else if (k == "offset") off = j.num();
            else if (k == "count") cnt = j.num();
            if (!j.lit(',')) break;
        }
        j.lit('}'); j.lit(',');
        long prod = shape.empty() ? 0 : 1; for (int d : shape) prod *= d;
        if (off < 0 || cnt <= 0 || prod != cnt || (size_t)(off + cnt) > w->data.size()) return false;   // 越界 / 形状与长度不符
        Tensor t; t.n = cnt; t.shape = shape;
        t.p = w->data.data() + off;
        w->byName[name] = t;
    }
    return !w->byName.empty();
}

bool Weights::loadFromMemory(const void* bin, size_t bytes, const std::string& jsonText) {
    data.assign((const float*)bin, (const float*)bin + bytes / 4);
    return parseManifest(jsonText, this);
}

bool Weights::load(const std::string& binPath, const std::string& jsonPath) {
    FILE* f = fopen(binPath.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    data.resize(n / 4);
    size_t rd = fread(data.data(), 1, n, f);
    fclose(f);
    if ((long)rd != n) return false;
    FILE* g = fopen(jsonPath.c_str(), "rb");
    if (!g) return false;
    fseek(g, 0, SEEK_END); long m = ftell(g); fseek(g, 0, SEEK_SET);
    std::string js(m, '\0');
    rd = fread(&js[0], 1, m, g);
    fclose(g);
    if ((long)rd != m) return false;
    return parseManifest(js, this);
}

const Tensor& Weights::at(const std::string& n) const {
    auto it = byName.find(n);
    if (it == byName.end()) throw std::runtime_error("ans48: missing weight " + n);   // 库里不退出宿主进程：由 ans48_create 接住返回 NULL
    return it->second;
}

// ---- 按算子类别计时（只在 -DULK_OPPROF 的剖析构建里有，库本身不编进去；ans48-dev/opprof.cpp 用）----
#ifdef ULK_OPPROF
long long g_opns[kOpCats] = {0};
struct OpTimer { int k; long long t0; explicit OpTimer(int k_) : k(k_) { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); t0 = t.tv_sec * 1000000000LL + t.tv_nsec; }
    ~OpTimer() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g_opns[k] += t.tv_sec * 1000000000LL + t.tv_nsec - t0; } };
#define OPT(k) OpTimer opTimer_(k)
#else
#define OPT(k)
#endif

// ---- 基本算子 ----------------------------------------------------------

// 参数先限到 [−80, 80]，expf 不会溢出成无穷大：-O3 -ffast-math 下 glibc 会换成不处理溢出的向量版 expf，
// 掩码接近 0 的频点（x < −88）会得到错值。±80 处 sigmoid 离 0 / 1 不到 1e-34。
static inline float sigmoidf_(float x) { x = x < -80.f ? -80.f : (x > 80.f ? 80.f : x); return 1.0f / (1.0f + expf(-x)); }   // 精确版（只留作对照）
// 激活：tanh 用 Eigen 的 13/6 次有理逼近（全实数轴最大绝对误差 3.3e-7），sigmoid = 0.5·tanh(x/2) + 0.5（误差 2.3e-7）。
// 无 exp、可向量化，且自带限幅不会溢出。整张网络每帧有几千次 tanh / sigmoid（GRU 门控与注意力），逐个调 libm 是主要耗时。
static inline float ftanh(float x) {
    x = std::min(7.90531110763549805f, std::max(-7.90531110763549805f, x)); const float x2 = x * x;
    float p = x2 * -2.76076847742355e-16f + 2.00018790482477e-13f; p = p * x2 + -8.60467152213735e-11f; p = p * x2 + 5.12229709037114e-08f; p = p * x2 + 1.48572235717979e-05f; p = p * x2 + 6.37261928875436e-04f; p = p * x2 + 4.89352455891786e-03f; p = p * x;
    float q = x2 * 1.19825839466702e-06f + 1.18534705686654e-04f; q = q * x2 + 2.26843463243900e-03f; q = q * x2 + 4.89352518554385e-03f; return p / q; }
static inline float fsigm(float x) { return 0.5f * ftanh(0.5f * x) + 0.5f; }

typedef float v4f __attribute__((vector_size(16)));
static inline v4f ld4(const float* p) { v4f v; memcpy(&v, p, 16); return v; }
static inline void st4(float* p, v4f v) { memcpy(p, &v, 16); }
// 矩阵 × 向量：一次 4 行，输入向量的每个元素读一次用 4 次（各行仍按 i 依次累加，与逐行写法同序）
static inline void gemv4(const float* __restrict x, int nIn, const float* __restrict W, const float* __restrict b, int nOut, float* __restrict y) {
    int o = 0;
    for (; o + 4 <= nOut; o += 4) {
        const float *w0 = W + (size_t)o * nIn, *w1 = w0 + nIn, *w2 = w1 + nIn, *w3 = w2 + nIn;
        float s0 = b ? b[o] : 0.0f, s1 = b ? b[o + 1] : 0.0f, s2 = b ? b[o + 2] : 0.0f, s3 = b ? b[o + 3] : 0.0f;
        for (int i = 0; i < nIn; ++i) { const float v = x[i]; s0 += w0[i] * v; s1 += w1[i] * v; s2 += w2[i] * v; s3 += w3[i] * v; }
        y[o] = s0; y[o + 1] = s1; y[o + 2] = s2; y[o + 3] = s3;
    }
    for (; o < nOut; ++o) { const float* wr = W + (size_t)o * nIn; float s = b ? b[o] : 0.0f; for (int i = 0; i < nIn; ++i) s += wr[i] * x[i]; y[o] = s; }
}
template <int NI, int NO>
static inline void linearT(const float* __restrict x, const float* __restrict W, const float* __restrict b, float* __restrict y) {
    for (int o = 0; o < NO; ++o) { const float* wr = W + o * NI; float s = b ? b[o] : 0.0f; for (int i = 0; i < NI; ++i) s += wr[i] * x[i]; y[o] = s; }
}

void linear(const float* x, int nIn, const float* W, const float* b, int nOut, float* y) {
    if (nIn == 16 && nOut == 16) { linearT<16, 16>(x, W, b, y); return; }   // DPGRNN 逐频点全连接
    if (nIn >= 16) { gemv4(x, nIn, W, b, nOut, y); return; }
    for (int o = 0; o < nOut; ++o) {
        const float* wr = W + (size_t)o * nIn;
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < nIn; ++i) s += wr[i] * x[i];
        y[o] = s;
    }
}

// 小尺寸 GRU 单步（输入 / 隐层宽度编译期固定）：每帧几百次调用、每次只有几十到几百次乘加，
// 通用版的循环开销占大头；尺寸写死后编译器能整段展开并向量化。计算式与累加顺序与通用版相同。
template <int NI, int H>
static inline void gruStepT(const float* __restrict x, float* __restrict h, const float* __restrict Wih, const float* __restrict Whh,
                            const float* __restrict bih, const float* __restrict bhh) {
    float g[3 * H], gh[3 * H];
    for (int k = 0; k < 3 * H; ++k) { const float* wr = Wih + k * NI; float s = bih[k]; for (int i = 0; i < NI; ++i) s += wr[i] * x[i]; g[k] = s; }
    for (int k = 0; k < 3 * H; ++k) { const float* wr = Whh + k * H; float s = bhh[k]; for (int i = 0; i < H; ++i) s += wr[i] * h[i]; gh[k] = s; }
    for (int i = 0; i < H; ++i) {
        const float r = fsigm(g[i] + gh[i]), z = fsigm(g[H + i] + gh[H + i]), n = ftanh(g[2 * H + i] + r * gh[2 * H + i]);
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

void gruStep(const float* x, int nIn, float* h, int nHid,
             const float* Wih, const float* Whh, const float* bih, const float* bhh,
             float* g) {
    if (nIn == 8 && nHid == 4) { gruStepT<8, 4>(x, h, Wih, Whh, bih, bhh); return; }   // DPGRNN 频率方向
    if (nIn == 8 && nHid == 8) { gruStepT<8, 8>(x, h, Wih, Whh, bih, bhh); return; }   // DPGRNN 时间方向
    if (nIn == 4 && nHid == 4) { gruStepT<4, 4>(x, h, Wih, Whh, bih, bhh); return; }   // 频率注意力

    // g 布局：[0,nHid) = W_ir·x + b_ir 等三门的输入部分，随后就地变成门值
    const int H = nHid;
    gemv4(x, nIn, Wih, bih, 3 * H, g);
    // 隐状态那三份单独算：n 门要用 r ⊙ (W_hn·h + b_hn)，**b_hn 在括号里面**。
    // 只用到 3*H，而 H 最大是 2*C（C≤32）→ 64。原来开 3*512 = 6 KB 栈，
    // 每次调用都白摸一遍缓存；这个函数一帧要被调几百次。
    float gh[3 * 96];
    gemv4(h, H, Whh, bhh, 3 * H, gh);
    for (int i = 0; i < H; ++i) {
        float r = fsigm(g[i] + gh[i]);
        float z = fsigm(g[H + i] + gh[H + i]);
        float n = ftanh(g[2 * H + i] + r * gh[2 * H + i]);
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

void gruBidir(const float* x, int len, int nIn, int nHid,
              const float* Wih, const float* Whh, const float* bih, const float* bhh,
              const float* WihR, const float* WhhR, const float* bihR, const float* bhhR,
              float* out, float* scratch) {
    float* h = scratch;                    // nHid
    float* g = scratch + nHid;             // 3*nHid
    for (int i = 0; i < nHid; ++i) h[i] = 0.0f;
    for (int t = 0; t < len; ++t) {
        gruStep(x + (size_t)t * nIn, nIn, h, nHid, Wih, Whh, bih, bhh, g);
        for (int i = 0; i < nHid; ++i) out[(size_t)t * 2 * nHid + i] = h[i];
    }
    for (int i = 0; i < nHid; ++i) h[i] = 0.0f;
    for (int t = len - 1; t >= 0; --t) {
        gruStep(x + (size_t)t * nIn, nIn, h, nHid, WihR, WhhR, bihR, bhhR, g);
        for (int i = 0; i < nHid; ++i) out[(size_t)t * 2 * nHid + nHid + i] = h[i];
    }
}

void shuffle2(const float* x, int C2, int nF, float* y) {
    // (2C,F) → 交错。与 torch 版一致：chunk 成两半，stack 后 transpose 再展平。
    const int C = C2 / 2;
    for (int c = 0; c < C; ++c) {
        memcpy(y + (size_t)(2 * c) * nF, x + (size_t)c * nF, sizeof(float) * nF);
        memcpy(y + (size_t)(2 * c + 1) * nF, x + (size_t)(C + c) * nF, sizeof(float) * nF);
    }
}

void layerNorm(float* x, int nF, int C, const float* w, const float* b, float eps) {
    const int N = nF * C;
    double m = 0.0;
    for (int i = 0; i < N; ++i) m += x[i];
    m /= N;
    double v = 0.0;
    for (int i = 0; i < N; ++i) { double d = x[i] - m; v += d * d; }
    v /= N;
    const float inv = (float)(1.0 / sqrt(v + eps));
    for (int i = 0; i < N; ++i) x[i] = ((float)(x[i] - m)) * inv * w[i] + b[i];
}

// ---- FA ----------------------------------------------------------------

void FA::prepare(const Weights& w, const std::string& pfx, int nFreq) {
    nF = nFreq; r = 4;
    int rem = nF % r; padLen = rem ? (r - rem) : 0;
    H = (nF + padLen) / r;
    Wih = w.at(pfx + ".gru.weight_ih_l0").p;
    Whh = w.at(pfx + ".gru.weight_hh_l0").p;
    bih = w.at(pfx + ".gru.bias_ih_l0").p;
    bhh = w.at(pfx + ".gru.bias_hh_l0").p;
    WihR = w.at(pfx + ".gru.weight_ih_l0_reverse").p;
    WhhR = w.at(pfx + ".gru.weight_hh_l0_reverse").p;
    bihR = w.at(pfx + ".gru.bias_ih_l0_reverse").p;
    bhhR = w.at(pfx + ".gru.bias_hh_l0_reverse").p;
    fcW = w.at(pfx + ".fc.weight").p;
    fcB = w.at(pfx + ".fc.bias").p;
    pad.assign(nF + padLen, 0.0f);
    gruOut.assign((size_t)H * 2 * r, 0.0f);
    scratch.assign(4 * r + 16, 0.0f);
    if (r != 4) throw std::runtime_error("ans48: FA expects freq_comp_ratio 4");
    wx.assign(2 * 3 * 4 * 4, 0.0f); wh.assign(wx.size(), 0.0f); bx.assign(2 * 3 * 4, 0.0f); bh.assign(bx.size(), 0.0f);
    for (int d = 0; d < 2; ++d) {
        const float *Wi = d ? WihR : Wih, *Wh = d ? WhhR : Whh, *bi = d ? bihR : bih, *bhh_ = d ? bhhR : bhh;
        for (int g = 0; g < 3; ++g) for (int k = 0; k < 4; ++k) {
            const int row = g * 4 + k;
            bx[(d * 3 + g) * 4 + k] = bi[row]; bh[(d * 3 + g) * 4 + k] = bhh_[row];
            for (int i = 0; i < 4; ++i) { wx[((d * 3 + g) * 4 + i) * 4 + k] = Wi[row * 4 + i]; wh[((d * 3 + g) * 4 + i) * 4 + k] = Wh[row * 4 + i]; }
        }
    }
    fcT.assign(8 * 4, 0.0f);
    for (int o = 0; o < 4; ++o) for (int i = 0; i < 8; ++i) fcT[i * 4 + o] = fcW[o * 8 + i];
}

// 4 宽的向量版 tanh / sigmoid：与 ftanh / fsigm 同一组系数、同一个式子，逐元素算
static inline v4f vclamp(v4f x, float lo, float hi) { for (int k = 0; k < 4; ++k) x[k] = std::min(hi, std::max(lo, x[k])); return x; }
static inline v4f vtanh(v4f x) {
    x = vclamp(x, -7.90531110763549805f, 7.90531110763549805f); const v4f x2 = x * x;
    v4f p = x2 * -2.76076847742355e-16f + 2.00018790482477e-13f; p = p * x2 + -8.60467152213735e-11f; p = p * x2 + 5.12229709037114e-08f; p = p * x2 + 1.48572235717979e-05f; p = p * x2 + 6.37261928875436e-04f; p = p * x2 + 4.89352455891786e-03f; p = p * x;
    v4f q = x2 * 1.19825839466702e-06f + 1.18534705686654e-04f; q = q * x2 + 2.26843463243900e-03f; q = q * x2 + 4.89352518554385e-03f; return p / q; }
static inline v4f vsigm(v4f x) { return 0.5f * vtanh(0.5f * x) + 0.5f; }

// FA 的一步 GRU（输入 4、隐层 4），四个隐单元放一个向量里一起算。各门 = 偏置 + 按输入下标依次累加，式子与 gruStepT<4,4> 相同。
static inline v4f faStep(v4f xv, v4f h, const float* wx, const float* wh, const float* bx, const float* bh) {
    float xs[4], hs[4]; memcpy(xs, &xv, 16); memcpy(hs, &h, 16);
    v4f g[3], gh[3];
    for (int q = 0; q < 3; ++q) {
        v4f a = ld4(bx + q * 4), c = ld4(bh + q * 4);
        for (int i = 0; i < 4; ++i) { a += xs[i] * ld4(wx + (q * 4 + i) * 4); c += hs[i] * ld4(wh + (q * 4 + i) * 4); }
        g[q] = a; gh[q] = c;
    }
    const v4f rr = vsigm(g[0] + gh[0]), zz = vsigm(g[1] + gh[1]), nn = vtanh(g[2] + rr * gh[2]);
    return (1.0f - zz) * nn + zz * h;
}

// 10-13：正反两个方向互不依赖，放进同一个循环交错算（小核顺序执行，单方向时每步都在等上一步）；
// 一步内四个隐单元向量化。频点能量按通道逐行累加（每个频点仍是 0 + 按通道依次累加，再除以 C）。
void FA::run(const float* x, int C, float* out) {
    float* __restrict pd = pad.data();
    for (int f = 0; f < nF + padLen; ++f) pd[f] = 0.0f;
    for (int c = 0; c < C; ++c) { const float* __restrict row = x + (size_t)c * nF; for (int f = 0; f < nF; ++f) pd[f] += row[f] * row[f]; }
    for (int f = 0; f < nF; ++f) pd[f] = pd[f] / C;
    float* __restrict go = gruOut.data();
    v4f hf = {0, 0, 0, 0}, hb = hf;
    const float *wx0 = wx.data(), *wh0 = wh.data(), *bx0 = bx.data(), *bh0 = bh.data();
    const float *wx1 = wx0 + 48, *wh1 = wh0 + 48, *bx1 = bx0 + 12, *bh1 = bh0 + 12;
    for (int t = 0; t < H; ++t) {
        const int tb = H - 1 - t;
        hf = faStep(ld4(pd + 4 * t), hf, wx0, wh0, bx0, bh0);
        hb = faStep(ld4(pd + 4 * tb), hb, wx1, wh1, bx1, bh1);
        st4(go + (size_t)t * 8, hf); st4(go + (size_t)tb * 8 + 4, hb);
    }
    const v4f fb = ld4(fcB);
    for (int h = 0; h < H; ++h) {
        const float* in = go + (size_t)h * 8; v4f y = fb;
        for (int i = 0; i < 8; ++i) y += in[i] * ld4(fcT.data() + i * 4);
        const int f0 = h * 4;
        if (f0 + 4 <= nF) st4(out + f0, y);
        else { float ys[4]; memcpy(ys, &y, 16); for (int k = 0; f0 + k < nF; ++k) out[f0 + k] = ys[k]; }
    }
}

// ---- cTFA --------------------------------------------------------------

void CTFA::prepare(const Weights& w, const std::string& pfx, int channels, int nFreq) {
    C = channels; nF = nFreq; hid = 2 * C;
    Wih = w.at(pfx + ".ta_gru.weight_ih_l0").p;
    Whh = w.at(pfx + ".ta_gru.weight_hh_l0").p;
    bih = w.at(pfx + ".ta_gru.bias_ih_l0").p;
    bhh = w.at(pfx + ".ta_gru.bias_hh_l0").p;
    fcW = w.at(pfx + ".ta_fc.weight").p;
    fcB = w.at(pfx + ".ta_fc.bias").p;
    fa.prepare(w, pfx + ".fa", nF);
    h.assign(hid, 0.0f);
    zt.assign(C, 0.0f); at.assign(C, 0.0f);
    gates.assign(3 * hid, 0.0f);
    af.assign(nF, 0.0f);
}

void CTFA::reset() { std::fill(h.begin(), h.end(), 0.0f); }

void CTFA::run(float* x, float* out, const int* outRow, const float* res) {
  {
    OPT(kOpTa);
    for (int c = 0; c < C; ++c) {
        float s = 0.0f;
        const float* row = x + (size_t)c * nF;
        for (int f = 0; f < nF; ++f) s += row[f] * row[f];
        zt[c] = s / nF;
    }
    gruStep(zt.data(), C, h.data(), hid, Wih, Whh, bih, bhh, gates.data());
    linear(h.data(), hid, fcW, fcB, C, at.data());
    for (int c = 0; c < C; ++c) at[c] = fsigm(at[c]);
  }
  {
    OPT(kOpFa);
    fa.run(x, C, af.data());
    for (int f = 0; f < nF; ++f) af[f] = fsigm(af[f]);
  }
    OPT(kOpMul);
    if (!out) {
        for (int c = 0; c < C; ++c) {
            float a = at[c];
            float* row = x + (size_t)c * nF;
            for (int f = 0; f < nF; ++f) row[f] *= a * af[f];
        }
        return;
    }
    for (int c = 0; c < C; ++c) {
        const float a = at[c]; const float* __restrict row = x + (size_t)c * nF; float* __restrict o = out + (size_t)outRow[c] * nF;
        if (res) { const float* __restrict r = res + (size_t)c * nF; for (int f = 0; f < nF; ++f) o[f] = row[f] * (a * af[f]) + r[f]; }
        else for (int f = 0; f < nF; ++f) o[f] = row[f] * (a * af[f]);
    }
}

// ---- AffinePReLU -------------------------------------------------------

void AffinePReLU::prepare(const Weights& w, const std::string& pfx, int channels, int nFreq) {
    C = channels; nF = nFreq;
    aw = w.at(pfx + ".affine_weight").p;
    ab = w.at(pfx + ".affine_bias").p;
    slope = w.at(pfx + ".slope_weight").p;
}

void AffinePReLU::run(float* x) const {
    for (int c = 0; c < C; ++c) row(x + (size_t)c * nF, c);
}

// ---- Conv / Deconv -----------------------------------------------------

void Conv::prepare(const Weights& w, const std::string& pfx, int cinN, int coutN,
                   int ktN, int kfN, int sfN, int groupsN, int finN, bool isDeconv) {
    cin = cinN; cout = coutN; kt = ktN; kf = kfN; sf = sfN; groups = groupsN;
    fin = finN; deconv = isDeconv;
    W = w.at(pfx + ".weight").p;
    b = w.has(pfx + ".bias") ? w.at(pfx + ".bias").p : nullptr;
    const int pf = kf / 2;
    fout = deconv ? (fin - 1) * sf - 2 * pf + kf
                  : (fin + 2 * pf - kf) / sf + 1;
    if (groups == cin && cin == cout && sf == 1 && kt * kf > kMaxTaps) throw std::runtime_error("ans48: depthwise conv taps exceed kMaxTaps");
    hist.assign((size_t)(kt - 1) * cin * fin, 0.0f);
    zb.assign(cout, 0.0f);
    pwPack.clear();
    if (!deconv && kt == 1 && kf == 1 && sf == 1 && fin == fout)
        for (int g = 0; g < groups; ++g) pwPack.push_back(packPw(W + (size_t)g * (cout / groups) * (cin / groups), cout / groups, cin / groups));
    padr.assign((size_t)std::max((size_t)kt * (fin + kf), (size_t)(2 * fout + kf)), 0.0f);   // 深度卷积每个时间帧一行补零行
    accr.assign((size_t)2 * fin + kf, 0.0f);
    if (sf == 2) {
        // 补零行：普通卷积读到 xp[2(fout−1) + kf−1 + 7]（步长 2 的向量读一次取 8 个），转置卷积读到 x[m + e/2 + 3]、m < fout/2+1
        // 转置：抽头位置偏移 e/2 = (r + pf − k)/2 可能为负，左边留 kf 个零兜住
        padL = deconv ? kf : pf;
        // 普通：偶 / 奇两段各 fout + kf + 8（读到 f + k/2 + 15，f < fout）；转置：一段
        padRowLen = deconv ? padL + std::max(fin, fout / 2 + 1) + kf + 8 : 2 * (fout + kf + 8);
        if (deconv && padRowLen < padL + fin) padRowLen = padL + fin + 8;
        padAll.assign((size_t)cin * kt * padRowLen, 0.0f);
        tapW.assign((size_t)(cin / groups) * kt * kf, 0.0f); tapP.assign(tapW.size(), nullptr);
        accr.assign((size_t)2 * (fout / 2 + 1 + 4) + 8, 0.0f);
    }
}

void Conv::reset() { std::fill(hist.begin(), hist.end(), 0.0f); }

// 逐点卷积（1×1，步长 1）：一次 4 个输出通道 × 4 个频点，累加值留在寄存器里。原写法每累加一个输入通道
// 都把整行输出读出再写回（读写多于计算）。每个输出元素仍是"偏置 + 按输入通道依次累加"，与原式同序。
// act 非空时每写完 4 行就地做激活（第 c0+o 个通道），趁这几行还在缓存里
// 逐点卷积权重的转置打包：每 4 个输出通道一块，块内 [输入通道][4]（不满 4 的尾行不打包，走原写法）
std::vector<float> packPw(const float* w, int O, int I) {
    const int OB = O / 4; std::vector<float> wt((size_t)OB * I * 4);
    for (int ob = 0; ob < OB; ++ob) for (int i = 0; i < I; ++i) for (int k = 0; k < 4; ++k) wt[((size_t)ob * I + i) * 4 + k] = w[(size_t)(ob * 4 + k) * I + i];
    return wt;
}

// wt 非空（10-13）：4 个输出 × 8 个频点一块，按输入通道读 1 个权重向量（4 个输出的权重，来自 packPw）+ 2 个输入向量、算 8 个向量乘加；
// 原写法每个通道读 1 个输入向量 + 4 个标量权重、只算 4 个，小核上读多于算（微基准：S10 / vivo 小核快约 22%，输出逐位相同）。
static void pw(const float* __restrict w, const float* __restrict b, const float* __restrict x, float* __restrict y, int O, int I, int N,
               const AffinePReLU* act = nullptr, int c0 = 0, const float* __restrict wt = nullptr) {
    const int N4 = N & ~3; int o = 0;
    if (wt) {
        const int N8 = N & ~7;
        for (; o + 4 <= O; o += 4) {
            const float* __restrict wb = wt + (size_t)(o / 4) * I * 4;
            float *y0 = y + o * N, *y1 = y0 + N, *y2 = y1 + N, *y3 = y2 + N;
            int n = 0;
            for (; n < N8; n += 8) {
                v4f a0 = {b[o], b[o], b[o], b[o]}, a1 = {b[o + 1], b[o + 1], b[o + 1], b[o + 1]}, a2 = {b[o + 2], b[o + 2], b[o + 2], b[o + 2]}, a3 = {b[o + 3], b[o + 3], b[o + 3], b[o + 3]};
                v4f e0 = a0, e1 = a1, e2 = a2, e3 = a3;
                for (int i = 0; i < I; ++i) {
                    const v4f wv = ld4(wb + i * 4); const v4f xa = ld4(x + i * N + n), xb = ld4(x + i * N + n + 4);
                    a0 += xa * wv[0]; a1 += xa * wv[1]; a2 += xa * wv[2]; a3 += xa * wv[3];
                    e0 += xb * wv[0]; e1 += xb * wv[1]; e2 += xb * wv[2]; e3 += xb * wv[3];
                }
                st4(y0 + n, a0); st4(y1 + n, a1); st4(y2 + n, a2); st4(y3 + n, a3);
                st4(y0 + n + 4, e0); st4(y1 + n + 4, e1); st4(y2 + n + 4, e2); st4(y3 + n + 4, e3);
            }
            for (; n < N4; n += 4) {
                v4f a0 = {b[o], b[o], b[o], b[o]}, a1 = {b[o + 1], b[o + 1], b[o + 1], b[o + 1]}, a2 = {b[o + 2], b[o + 2], b[o + 2], b[o + 2]}, a3 = {b[o + 3], b[o + 3], b[o + 3], b[o + 3]};
                for (int i = 0; i < I; ++i) { const v4f wv = ld4(wb + i * 4); const v4f xa = ld4(x + i * N + n); a0 += xa * wv[0]; a1 += xa * wv[1]; a2 += xa * wv[2]; a3 += xa * wv[3]; }
                st4(y0 + n, a0); st4(y1 + n, a1); st4(y2 + n, a2); st4(y3 + n, a3);
            }
            for (; n < N; ++n) { float a[4] = {b[o], b[o + 1], b[o + 2], b[o + 3]};
                for (int i = 0; i < I; ++i) { const float xv = x[i * N + n]; for (int k = 0; k < 4; ++k) a[k] += wb[i * 4 + k] * xv; }
                y0[n] = a[0]; y1[n] = a[1]; y2[n] = a[2]; y3[n] = a[3]; }
            if (act) { act->row(y0, c0 + o); act->row(y1, c0 + o + 1); act->row(y2, c0 + o + 2); act->row(y3, c0 + o + 3); }
        }
    }
    for (; o + 4 <= O; o += 4) {
        const float *w0 = w + o * I, *w1 = w0 + I, *w2 = w1 + I, *w3 = w2 + I; float *y0 = y + o * N, *y1 = y0 + N, *y2 = y1 + N, *y3 = y2 + N;
        for (int n = 0; n < N4; n += 4) {
            v4f a0 = {b[o], b[o], b[o], b[o]}, a1 = {b[o + 1], b[o + 1], b[o + 1], b[o + 1]}, a2 = {b[o + 2], b[o + 2], b[o + 2], b[o + 2]}, a3 = {b[o + 3], b[o + 3], b[o + 3], b[o + 3]};
            for (int i = 0; i < I; ++i) { const v4f xv = ld4(x + i * N + n); a0 += w0[i] * xv; a1 += w1[i] * xv; a2 += w2[i] * xv; a3 += w3[i] * xv; }
            st4(y0 + n, a0); st4(y1 + n, a1); st4(y2 + n, a2); st4(y3 + n, a3);
        }
        for (int n = N4; n < N; ++n) { float a0 = b[o], a1 = b[o + 1], a2 = b[o + 2], a3 = b[o + 3];
            for (int i = 0; i < I; ++i) { const float xv = x[i * N + n]; a0 += w0[i] * xv; a1 += w1[i] * xv; a2 += w2[i] * xv; a3 += w3[i] * xv; }
            y0[n] = a0; y1[n] = a1; y2[n] = a2; y3[n] = a3; }
        if (act) { act->row(y0, c0 + o); act->row(y1, c0 + o + 1); act->row(y2, c0 + o + 2); act->row(y3, c0 + o + 3); }
    }
    for (; o < O; ++o) { const float* wo = w + o * I; float* yo = y + o * N;
        for (int n = 0; n < N4; n += 4) { v4f a = {b[o], b[o], b[o], b[o]}; for (int i = 0; i < I; ++i) a += wo[i] * ld4(x + i * N + n); st4(yo + n, a); }
        for (int n = N4; n < N; ++n) { float a = b[o]; for (int i = 0; i < I; ++i) a += wo[i] * x[i * N + n]; yo[n] = a; }
        if (act) act->row(yo, c0 + o); }
}

// 深度卷积一行（抽头数编译期固定）：rows[t] 是第 t 个时间帧的补零行，w 按 (t, k) 排；每个输出点 = 偏置 + 按 (t, k) 依次累加。
// 一次 16 个输出点、4 个互不依赖的累加器：小核（顺序执行）上乘加的结果要等几拍才能接着用，只有 2 个累加器时
// 每个抽头都在等上一个（10-13 simpleperf 实测热点就是这条依赖链，S10 小核只有 0.4 乘加/ns）。
template <int KT, int K>
static inline void dwRowT(const float* const* rows, const float* __restrict w, float bv, float* __restrict yo, int fout) {
    float wv[KT * K]; for (int q = 0; q < KT * K; ++q) wv[q] = w[q];
    int f = 0;
    for (; f + 16 <= fout; f += 16) {
        v4f a0 = {bv, bv, bv, bv}, a1 = a0, a2 = a0, a3 = a0;
        for (int t = 0; t < KT; ++t) for (int k = 0; k < K; ++k) {
            const float wq = wv[t * K + k]; const float* p = rows[t] + k + f;
            a0 += wq * ld4(p); a1 += wq * ld4(p + 4); a2 += wq * ld4(p + 8); a3 += wq * ld4(p + 12);
        }
        st4(yo + f, a0); st4(yo + f + 4, a1); st4(yo + f + 8, a2); st4(yo + f + 12, a3);
    }
    for (; f + 4 <= fout; f += 4) {
        v4f a0 = {bv, bv, bv, bv};
        for (int t = 0; t < KT; ++t) for (int k = 0; k < K; ++k) a0 += wv[t * K + k] * ld4(rows[t] + k + f);
        st4(yo + f, a0);
    }
    for (; f < fout; ++f) { float a = bv; for (int t = 0; t < KT; ++t) for (int k = 0; k < K; ++k) a += wv[t * K + k] * rows[t][k + f]; yo[f] = a; }
}

// 若干抽头的连续相关一行：y[f] = b + Σ_q w[q]·p[q][f]（按 q 依次累加）。一次 16 点、4 个累加器。
static void contigRow(const float* const* p, const float* w, int nt, float bv, float* __restrict y, int n) {
    int f = 0;
    for (; f + 16 <= n; f += 16) {
        v4f a0 = {bv, bv, bv, bv}, a1 = a0, a2 = a0, a3 = a0;
        for (int q = 0; q < nt; ++q) { const float wq = w[q]; const float* s = p[q] + f; a0 += wq * ld4(s); a1 += wq * ld4(s + 4); a2 += wq * ld4(s + 8); a3 += wq * ld4(s + 12); }
        st4(y + f, a0); st4(y + f + 4, a1); st4(y + f + 8, a2); st4(y + f + 12, a3);
    }
    for (; f + 4 <= n; f += 4) { v4f a0 = {bv, bv, bv, bv}; for (int q = 0; q < nt; ++q) a0 += w[q] * ld4(p[q] + f); st4(y + f, a0); }
    for (; f < n; ++f) { float a = bv; for (int q = 0; q < nt; ++q) a += w[q] * p[q][f]; y[f] = a; }
}
// 步长 2 的一行：y[f] = b + Σ_q w[q]·p[q][2f]（深度卷积用：每个通道只读自己那一行，拆奇偶的拷贝摊不开，直接交错读）
static inline v4f ld4s2(const float* p) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    v4f v; float32x4_t e = vld2q_f32(p).val[0]; memcpy(&v, &e, 16); return v;
#else
    v4f v = {p[0], p[2], p[4], p[6]}; return v;
#endif
}
static void stride2Row(const float* const* p, const float* w, int nt, float bv, float* __restrict y, int n) {
    int f = 0;
    for (; f + 16 <= n; f += 16) {
        v4f a0 = {bv, bv, bv, bv}, a1 = a0, a2 = a0, a3 = a0;
        for (int q = 0; q < nt; ++q) { const float wq = w[q]; const float* s = p[q] + 2 * f; a0 += wq * ld4s2(s); a1 += wq * ld4s2(s + 8); a2 += wq * ld4s2(s + 16); a3 += wq * ld4s2(s + 24); }
        st4(y + f, a0); st4(y + f + 4, a1); st4(y + f + 8, a2); st4(y + f + 12, a3);
    }
    for (; f + 4 <= n; f += 4) { v4f a0 = {bv, bv, bv, bv}; for (int q = 0; q < nt; ++q) a0 += w[q] * ld4s2(p[q] + 2 * f); st4(y + f, a0); }
    for (; f < n; ++f) { float a = bv; for (int q = 0; q < nt; ++q) a += w[q] * p[q][2 * f]; y[f] = a; }
}

void Conv::run(const float* x, float* y, const AffinePReLU* post) {
    if (!deconv && kt == 1 && kf == 1 && sf == 1 && fin == fout) {   // 逐点卷积快速路径（按组）
        OPT(kOpPw);
        const int cinG = cin / groups, coutG = cout / groups; const float* bb = b ? b : zb.data();
        for (int g = 0; g < groups; ++g)
            pw(W + (size_t)g * coutG * cinG, bb + g * coutG, x + (size_t)g * cinG * fin, y + (size_t)g * coutG * fout, coutG, cinG, fout, post, g * coutG,
               pwPack.empty() ? nullptr : pwPack[g].data());
        return;
    }
    OPT(kOpConv);
    const int pf = kf / 2;
    const int cinG = cin / groups, coutG = cout / groups;
    const size_t frameSz = (size_t)cin * fin;

    auto frameAt = [&](int t) -> const float* {
        // t ∈ [0,kt)：0 是最旧的一帧，kt-1 是当前帧
        return (t == kt - 1) ? x : (hist.data() + (size_t)t * frameSz);
    };

    // 深度卷积（每个输入通道对应一个输出通道、步长 1，普通与转置）：输入行先拷进左右补零的缓冲，
    // 再对每个抽头整行乘加——没有边界判断，内循环连续可向量化。转置卷积 = 抽头倒序的相关，左补零宽度 kf−1−pf。
    if (groups == cin && cin == cout && sf == 1 && fin == fout) {
        // 10-13：累加值留在寄存器里跨全部抽头（原写法每个抽头整行读出、加、写回，读写多于计算；
        // 实测 S10 小核只有 0.3 乘加/ns，逐点卷积是 1.5）。各时间帧先拷进各自的补零行，
        // 每个输出点仍是「偏置 + 按 (t, k) 依次累加」，与原式同序。
        const int K = kf, L = deconv ? (K - 1 - pf) : pf, PL = fin + K - 1, NT = kt * K;
        float* __restrict xpb = padr.data();
        const float* src[kMaxTaps]; float wv[kMaxTaps];
        const int F8 = fout & ~7, F4 = fout & ~3;
        for (int c = 0; c < cin; ++c) {
            float* __restrict yo = y + (size_t)c * fout; const float bv = b ? b[c] : 0.0f;
            for (int t = 0; t < kt; ++t) {
                float* xp = xpb + (size_t)t * PL;
                const float* xr = frameAt(t) + (size_t)(inMap.empty() ? c : inMap[c]) * fin;
                for (int q = 0; q < L; ++q) xp[q] = 0.0f;
                memcpy(xp + L, xr, sizeof(float) * fin);
                for (int q = L + fin; q < PL; ++q) xp[q] = 0.0f;
                const float* wr = W + ((size_t)c * kt + (deconv ? kt - 1 - t : t)) * K;
                for (int k = 0; k < K; ++k) { wv[t * K + k] = deconv ? wr[K - 1 - k] : wr[k]; src[t * K + k] = xp + k; }
            }
            const float* rows[3] = {xpb, kt > 1 ? xpb + PL : xpb, kt > 2 ? xpb + 2 * PL : xpb};
            if (kt == 1 && K == 5) dwRowT<1, 5>(rows, wv, bv, yo, fout);
            else if (kt == 2 && K == 3) dwRowT<2, 3>(rows, wv, bv, yo, fout);
            else {
                int f = 0;
                for (; f < F8; f += 8) {
                    v4f a0 = {bv, bv, bv, bv}, a1 = a0;
                    for (int q = 0; q < NT; ++q) { a0 += wv[q] * ld4(src[q] + f); a1 += wv[q] * ld4(src[q] + f + 4); }
                    st4(yo + f, a0); st4(yo + f + 4, a1);
                }
                for (; f < F4; f += 4) { v4f a0 = {bv, bv, bv, bv}; for (int q = 0; q < NT; ++q) a0 += wv[q] * ld4(src[q] + f); st4(yo + f, a0); }
                for (; f < fout; ++f) { float a = bv; for (int q = 0; q < NT; ++q) a += wv[q] * src[q][f]; yo[f] = a; }
            }
            if (post) post->row(yo, c);
        }
        if (kt > 1) {
            for (int t = 0; t + 1 < kt - 1; ++t)
                memcpy(hist.data() + (size_t)t * frameSz, hist.data() + (size_t)(t + 1) * frameSz, sizeof(float) * frameSz);
            memcpy(hist.data() + (size_t)(kt - 2) * frameSz, x, sizeof(float) * frameSz);
        }
        return;
    }

    // 步长 2（普通与转置，任意分组）：同样先补零、按抽头整行乘加，没有边界判断。
    //   普通：y[f] += w[k]·xp[2f + k]，xp 左补 pf 个零；
    //   转置：输入点 i 落到输出 2i + k − pf，写进左边多留 pf 个位置的累加行 acc[2i + k]，最后取 acc[f + pf]。
    if (sf == 2) {
        // 10-13 改写：输入行（每个输入通道 × 每个时间帧）先一次性补零放进 padAll，各输出通道共用（原写法每个输出通道都重拷一遍，
        // 首层 12 个输出通道就拷 12 遍）；一次 16 个输出点、4 个累加器（小核顺序执行，累加器少就卡在依赖链上）。
        //   普通：y[f] = b + Σ_(ci,t,k) w·xp[2f + k]，xp 左补 pf 个零；
        //   转置：y[f] = b + Σ_(ci,t,k) w·x[i]，其中 2i + k = f + pf。按 f 的奇偶拆成两条连续序列 f = 2m + r：
        //         只有 k ≡ r + pf (mod 2) 的抽头有贡献，i = m + (r + pf − k)/2，于是每条序列都是普通的连续相关（散射改聚集），
        //         最后两条交错写回。每个输出点的累加顺序仍是 (ci, t, k)，偏置最后加，与原式同序。
        const int K = kf, cinG = cin / groups, coutG = cout / groups;
        const int RL = padRowLen;                                   // 每行补零后的长度（prepare 里定）
        float* __restrict pa = padAll.data();
        if (!deconv && groups == cin && cin == cout) {
            // 步长 2 的深度卷积：每个通道一行补零行（左 pf 个零），交错读
            for (int c = 0; c < cin; ++c) {
                const float bv = b ? b[c] : 0.0f; float* __restrict yo = y + (size_t)c * fout;
                int nt = 0;
                for (int t = 0; t < kt; ++t) {
                    float* r = pa + (size_t)t * RL;
                    const float* xr = frameAt(t) + (size_t)(inMap.empty() ? c : inMap[c]) * fin;
                    for (int q = 0; q < pf; ++q) r[q] = 0.0f;
                    memcpy(r + pf, xr, sizeof(float) * fin);
                    for (int q = pf + fin; q < RL; ++q) r[q] = 0.0f;
                    const float* wr = W + ((size_t)c * kt + t) * K;
                    for (int k = 0; k < K; ++k) { tapW[nt] = wr[k]; tapP[nt] = r + k; ++nt; }
                }
                stride2Row(tapP.data(), tapW.data(), nt, bv, yo, fout);
                if (post) post->row(yo, c);
            }
        } else if (!deconv) {
            // 普通卷积：补零行 xp（左 pf 个零）拆成偶数位 E[m] = xp[2m] 与奇数位 O[m] = xp[2m+1] 两段存，
            // 于是 xp[2f + k] = (k 偶 ? E : O)[f + k/2]，每个抽头都是连续读（小核上步长 2 的交错读 vld2 很慢，10-13 实测）
            const int RH = RL / 2;
            for (int c = 0; c < cin; ++c) for (int t = 0; t < kt; ++t) {
                float* E = pa + ((size_t)c * kt + t) * RL; float* O = E + RH;
                const float* xr = frameAt(t) + (size_t)(inMap.empty() ? c : inMap[c]) * fin;
                // E[m] = xr[2m − pf]、O[m] = xr[2m + 1 − pf]，越界处补零；有效区间算好后整段无分支拷
                const int e0 = (pf + 1) / 2, e1 = std::min(RH, (fin - 1 + pf) / 2 + 1);   // E 的有效 m：0 ≤ 2m−pf < fin
                const int o0 = pf / 2, o1 = std::min(RH, (fin - 2 + pf) / 2 + 1);         // O 的有效 m：0 ≤ 2m+1−pf < fin
                for (int m = 0; m < e0; ++m) E[m] = 0.0f;
                { const float* xs = xr + 2 * e0 - pf; for (int m = e0; m < e1; ++m) E[m] = xs[2 * (m - e0)]; }
                for (int m = std::max(e0, e1); m < RH; ++m) E[m] = 0.0f;
                for (int m = 0; m < o0; ++m) O[m] = 0.0f;
                { const float* xs = xr + 2 * o0 + 1 - pf; for (int m = o0; m < o1; ++m) O[m] = xs[2 * (m - o0)]; }
                for (int m = std::max(o0, o1); m < RH; ++m) O[m] = 0.0f;
            }
            for (int o = 0; o < cout; ++o) {
                const int g = o / coutG; const float bv = b ? b[o] : 0.0f; float* __restrict yo = y + (size_t)o * fout;
                int nt = 0;
                for (int ci = 0; ci < cinG; ++ci) for (int t = 0; t < kt; ++t) {
                    const float* E = pa + ((size_t)(g * cinG + ci) * kt + t) * RL;
                    const float* wr = W + ((((size_t)o * cinG + ci) * kt + t) * K);
                    for (int k = 0; k < K; ++k) { tapW[nt] = wr[k]; tapP[nt] = (k & 1 ? E + RH : E) + k / 2; ++nt; }
                }
                contigRow(tapP.data(), tapW.data(), nt, bv, yo, fout);
                if (post) post->row(yo, o);
            }
        } else {
            for (int c = 0; c < cin; ++c) for (int t = 0; t < kt; ++t) {
                float* r = pa + ((size_t)c * kt + t) * RL;
                const float* xr = frameAt(t) + (size_t)(inMap.empty() ? c : inMap[c]) * fin;
                for (int q = 0; q < padL; ++q) r[q] = 0.0f;
                memcpy(r + padL, xr, sizeof(float) * fin);
                for (int q = padL + fin; q < RL; ++q) r[q] = 0.0f;
            }
            float* __restrict yr = accr.data();                     // 两条奇偶子序列的暂存
            const int M0 = (fout + 1) / 2, M1 = fout / 2;
            for (int o = 0; o < cout; ++o) {
                const int g = o / coutG, j = o % coutG; const float bv = b ? b[o] : 0.0f;
                for (int rpar = 0; rpar < 2; ++rpar) {
                    int nt = 0;
                    for (int ci = 0; ci < cinG; ++ci) { const int c = g * cinG + ci;
                        for (int t = 0; t < kt; ++t) {
                            const float* wr = W + ((((size_t)c * coutG + j) * kt + (kt - 1 - t)) * K);
                            const float* r = pa + ((size_t)c * kt + t) * RL + padL;            // r[i] = x[i]
                            for (int k = 0; k < K; ++k) {
                                const int e = rpar + pf - k; if (e & 1) continue;               // 奇偶不符：无贡献
                                tapW[nt] = wr[k]; tapP[nt] = r + e / 2; ++nt;                     // x[m + e/2]
                            }
                        } }
                    contigRow(tapP.data(), tapW.data(), nt, 0.0f, yr + (size_t)rpar * (M0 + 4), rpar ? M1 : M0);
                }
                float* yo = y + (size_t)o * fout;
                for (int m = 0; m < M1; ++m) { yo[2 * m] = bv + yr[m]; yo[2 * m + 1] = bv + yr[M0 + 4 + m]; }
                if (M0 > M1) yo[2 * M1] = bv + yr[M1];
                if (post) post->row(yo, o);
            }
        }
        if (kt > 1) {
            for (int t = 0; t + 1 < kt - 1; ++t)
                memcpy(hist.data() + (size_t)t * frameSz, hist.data() + (size_t)(t + 1) * frameSz, sizeof(float) * frameSz);
            memcpy(hist.data() + (size_t)(kt - 2) * frameSz, x, sizeof(float) * frameSz);
        }
        return;
    }

    for (int o = 0; o < cout; ++o) {
        float* yo = y + (size_t)o * fout;
        for (int f = 0; f < fout; ++f) yo[f] = b ? b[o] : 0.0f;
    }

    if (!deconv) {
        // 普通卷积：当前帧对应权重的最后一个时间抽头。
        //
        // **频率内循环拆成"内部 + 两侧边缘"**：原来每个抽头都带 `fi>=0 && fi<fin`
        // 的判断，编译器因此不敢向量化整条内循环。逐块耗时是平摊的（最大的一块
        // 也只占 17%），所以能普遍生效的改动才有意义，这就是其中最直接的一条。
        //
        // 内部区间：f*sf-pf >= 0 且 f*sf-pf+kf-1 <= fin-1
        int lo = 0;
        while (lo < fout && lo * sf - pf < 0) ++lo;
        int hi = fout;                       // 开区间
        while (hi > lo && (hi - 1) * sf - pf + kf - 1 >= fin) --hi;

        for (int o = 0; o < cout; ++o) {
            const int g = o / coutG;
            float* yo = y + (size_t)o * fout;
            for (int ci = 0; ci < cinG; ++ci) {
                const int c = g * cinG + ci;
                for (int t = 0; t < kt; ++t) {
                    const float* xr = frameAt(t) + (size_t)c * fin;
                    const float* wr = W + ((((size_t)o * cinG + ci) * kt + t) * kf);
                    // 左边缘
                    for (int f = 0; f < lo; ++f) {
                        float s = 0.0f;
                        const int base = f * sf - pf;
                        for (int k = 0; k < kf; ++k) {
                            const int fi = base + k;
                            if (fi >= 0 && fi < fin) s += xr[fi] * wr[k];
                        }
                        yo[f] += s;
                    }
                    // 内部：无边界判断
                    if (kf == 1) {
                        const float w0 = wr[0];
                        const float* xp = xr - pf;
                        if (sf == 1) {
                            for (int f = lo; f < hi; ++f) yo[f] += xp[f] * w0;
                        } else {
                            for (int f = lo; f < hi; ++f) yo[f] += xp[f * sf] * w0;
                        }
                    } else if (sf == 1) {
                        const float* xp = xr - pf;
                        for (int f = lo; f < hi; ++f) {
                            float s = 0.0f;
                            const float* xf = xp + f;
                            for (int k = 0; k < kf; ++k) s += xf[k] * wr[k];
                            yo[f] += s;
                        }
                    } else {
                        for (int f = lo; f < hi; ++f) {
                            float s = 0.0f;
                            const float* xf = xr + f * sf - pf;
                            for (int k = 0; k < kf; ++k) s += xf[k] * wr[k];
                            yo[f] += s;
                        }
                    }
                    // 右边缘
                    for (int f = hi; f < fout; ++f) {
                        float s = 0.0f;
                        const int base = f * sf - pf;
                        for (int k = 0; k < kf; ++k) {
                            const int fi = base + k;
                            if (fi >= 0 && fi < fin) s += xr[fi] * wr[k];
                        }
                        yo[f] += s;
                    }
                }
            }
        }
    } else {
        // 转置卷积：ZeroPad(kt-1) + padding=(kt-1) 之后输出恰好一帧，
        // 且**时间抽头是反的**——当前帧用 W[0]，最旧的一帧用 W[kt-1]。
        const int cinPerG = cin / groups;
        if (sf == 1) {
            // **散射改聚集。** 散射写法（对每个输入点往输出上撒 kf 个位置）没法向量化，
            // 而 sf==1 时可以反解成 out[f] = Σ_k in[f+pf−k]·W[k]，就是一次相关，
            // 内循环连续、可向量化。sf==2 有奇偶约束，仍走散射。
            for (int c = 0; c < cin; ++c) {
                const int g = c / cinPerG;
                for (int t = 0; t < kt; ++t) {
                    const float* xr = frameAt(t) + (size_t)c * fin;
                    const int wt = kt - 1 - t;
                    for (int j = 0; j < coutG; ++j) {
                        const int o = g * coutG + j;
                        float* yo = y + (size_t)o * fout;
                        const float* wr = W + ((((size_t)c * coutG + j) * kt + wt) * kf);
                        int lo = kf - 1 - pf; if (lo < 0) lo = 0;
                        int hi = fin - pf;    if (hi > fout) hi = fout;
                        for (int f = 0; f < lo && f < fout; ++f) {
                            float acc = 0.0f;
                            for (int k = 0; k < kf; ++k) {
                                const int idx = f + pf - k;
                                if (idx >= 0 && idx < fin) acc += xr[idx] * wr[k];
                            }
                            yo[f] += acc;
                        }
                        for (int f = lo; f < hi; ++f) {
                            float acc = 0.0f;
                            const float* xf = xr + f + pf;
                            for (int k = 0; k < kf; ++k) acc += xf[-k] * wr[k];
                            yo[f] += acc;
                        }
                        for (int f = (hi > lo ? hi : lo); f < fout; ++f) {
                            float acc = 0.0f;
                            for (int k = 0; k < kf; ++k) {
                                const int idx = f + pf - k;
                                if (idx >= 0 && idx < fin) acc += xr[idx] * wr[k];
                            }
                            yo[f] += acc;
                        }
                    }
                }
            }
            if (kt > 1) {
                for (int t = 0; t + 1 < kt - 1; ++t)
                    memcpy(hist.data() + (size_t)t * frameSz,
                           hist.data() + (size_t)(t + 1) * frameSz, sizeof(float) * frameSz);
                memcpy(hist.data() + (size_t)(kt - 2) * frameSz, x, sizeof(float) * frameSz);
            }
            if (post) post->run(y);
            return;
        }
        for (int c = 0; c < cin; ++c) {
            const int g = c / cinPerG;
            for (int t = 0; t < kt; ++t) {
                const float* xr = frameAt(t) + (size_t)c * fin;
                const int wt = kt - 1 - t;
                for (int j = 0; j < coutG; ++j) {
                    const int o = g * coutG + j;
                    float* yo = y + (size_t)o * fout;
                    const float* wr = W + ((((size_t)c * coutG + j) * kt + wt) * kf);
                    for (int fi = 0; fi < fin; ++fi) {
                        const float v = xr[fi];
                        if (v == 0.0f) continue;
                        for (int k = 0; k < kf; ++k) {
                            const int f = fi * sf + k - pf;
                            if (f >= 0 && f < fout) yo[f] += v * wr[k];
                        }
                    }
                }
            }
        }
    }

    if (kt > 1) {
        // 历史左移一帧，把当前帧放到末尾
        for (int t = 0; t + 1 < kt - 1; ++t)
            memcpy(hist.data() + (size_t)t * frameSz,
                   hist.data() + (size_t)(t + 1) * frameSz, sizeof(float) * frameSz);
        memcpy(hist.data() + (size_t)(kt - 2) * frameSz, x, sizeof(float) * frameSz);
    }
    if (post) post->run(y);
}

}  // namespace ulk

namespace ulk {

// ---- 组合块 ------------------------------------------------------------

void XConvBlock::reset() { conv.reset(); ctfa.reset(); }

void XConvBlock::run(const float* x, float* y) {
    conv.run(x, useShuffle ? tmp.data() : y, useAct ? &act : nullptr);
    if (useShuffle) ctfa.run(tmp.data(), y, perm.data());   // 输出洗牌并进 cTFA 的最后一遍
    else ctfa.run(y);
}

void XDWSBlock::reset() { pconv.reset(); dconv.reset(); ctfa.reset(); }

void XDWSBlock::run(const float* x, float* y) {
    pconv.run(x, t1.data(), &act1);
    // 不要 std::swap(t1,t2)：两个缓冲尺寸不一定相同，交换之后后面按名字取到的是错的那块（会越界）。
    const float* dz = t1.data();
    if (shuffleP && dconv.inMap.empty()) { shuffle2(t1.data(), cmid, fmid, t2.data()); dz = t2.data(); }   // 一般情况下洗牌已并进 dconv 的读取
    dconv.run(dz, y, useAct2 ? &act2 : nullptr);
    ctfa.run(y);
}

void XMBBlock::reset() { pconv1.reset(); dconv.reset(); pconv2.reset(); ctfa.reset(); }

void XMBBlock::run(const float* x, float* y) {
    pconv1.run(x, t1.data(), &act1);
    const float* dz = t1.data();
    if (shuffleP && dconv.inMap.empty()) { shuffle2(t1.data(), cmid, fmid, t2.data()); dz = t2.data(); }
    dconv.run(dz, t3.data(), &act2);
    // 洗牌的目的地必须是**按 cout×fout 分配**的那块，不能借用中间缓冲——
    // 借用一次就越界（t2 是按 cmid×fmid 分配的）。
    float* dst = shuffleOut ? t4.data() : y;
    pconv2.run(t3.data(), dst);
    // 残差：形状相同才加，与 torch 版的 `if x.shape == input.shape` 一致
    const bool res = cin == cout && fin == fout;
    if (shuffleOut) { ctfa.run(dst, y, perm.data(), res ? x : nullptr); return; }   // cTFA 乘、残差、洗牌一遍做完
    ctfa.run(dst);
    if (res) {
        const size_t n = (size_t)cout * fout;
        for (size_t i = 0; i < n; ++i) dst[i] += x[i];
    }
}

// ---- DPGRNN ------------------------------------------------------------

void DPGRNN::prepare(const Weights& w, const std::string& pfx, int channels, int width) {
    C = channels; F = width; hid = C;
    auto P = [&](const std::string& s) { return w.at(pfx + s).p; };
    i1Wih = P(".intra_rnn.rnn1.weight_ih_l0");   i1Whh = P(".intra_rnn.rnn1.weight_hh_l0");
    i1bih = P(".intra_rnn.rnn1.bias_ih_l0");     i1bhh = P(".intra_rnn.rnn1.bias_hh_l0");
    i1WihR = P(".intra_rnn.rnn1.weight_ih_l0_reverse");
    i1WhhR = P(".intra_rnn.rnn1.weight_hh_l0_reverse");
    i1bihR = P(".intra_rnn.rnn1.bias_ih_l0_reverse");
    i1bhhR = P(".intra_rnn.rnn1.bias_hh_l0_reverse");
    i2Wih = P(".intra_rnn.rnn2.weight_ih_l0");   i2Whh = P(".intra_rnn.rnn2.weight_hh_l0");
    i2bih = P(".intra_rnn.rnn2.bias_ih_l0");     i2bhh = P(".intra_rnn.rnn2.bias_hh_l0");
    i2WihR = P(".intra_rnn.rnn2.weight_ih_l0_reverse");
    i2WhhR = P(".intra_rnn.rnn2.weight_hh_l0_reverse");
    i2bihR = P(".intra_rnn.rnn2.bias_ih_l0_reverse");
    i2bhhR = P(".intra_rnn.rnn2.bias_hh_l0_reverse");
    ifcW = P(".intra_fc.weight");  ifcB = P(".intra_fc.bias");
    ilnW = P(".intra_ln.weight");  ilnB = P(".intra_ln.bias");
    e1Wih = P(".inter_rnn.rnn1.weight_ih_l0");   e1Whh = P(".inter_rnn.rnn1.weight_hh_l0");
    e1bih = P(".inter_rnn.rnn1.bias_ih_l0");     e1bhh = P(".inter_rnn.rnn1.bias_hh_l0");
    e2Wih = P(".inter_rnn.rnn2.weight_ih_l0");   e2Whh = P(".inter_rnn.rnn2.weight_hh_l0");
    e2bih = P(".inter_rnn.rnn2.bias_ih_l0");     e2bhh = P(".inter_rnn.rnn2.bias_hh_l0");
    efcW = P(".inter_fc.weight");  efcB = P(".inter_fc.bias");
    elnW = P(".inter_ln.weight");  elnB = P(".inter_ln.bias");

    const int half = C / 2, hh = half / 2;
    hInter.assign((size_t)F * C, 0.0f);      // 每个频点一份时间方向的状态，(C,F) 排
    xfc.assign((size_t)F * C, 0.0f);
    cat.assign((size_t)F * C, 0.0f);
    if (hh > 16 || 3 * hh > 48) throw std::runtime_error("ans48: dpgrnn intra hidden too large");
    pj.assign((size_t)3 * hh * F, 0.0f); gx.assign((size_t)3 * half * F, 0.0f); gh.assign((size_t)3 * half * F, 0.0f);
    if (hh == 4) {
        pj4.assign((size_t)4 * 12 * F, 0.0f); uT4.assign(4 * 48, 0.0f); bh4.assign(4 * 12, 0.0f);
        const float* Us[4] = {i1Whh, i1WhhR, i2Whh, i2WhhR}; const float* Bs[4] = {i1bhh, i1bhhR, i2bhh, i2bhhR};
        for (int k = 0; k < 4; ++k) for (int q = 0; q < 3; ++q) for (int j = 0; j < 4; ++j) {
            bh4[k * 12 + q * 4 + j] = Bs[k][q * 4 + j];
            for (int i = 0; i < 4; ++i) uT4[k * 48 + (q * 4 + i) * 4 + j] = Us[k][(q * 4 + j) * 4 + i];
        }
    }
    pkIntra[0] = packPw(i1Wih, 3 * hh, half); pkIntra[1] = packPw(i1WihR, 3 * hh, half);
    pkIntra[2] = packPw(i2Wih, 3 * hh, half); pkIntra[3] = packPw(i2WihR, 3 * hh, half);
    pkIfc = packPw(ifcW, C, C); pkEfc = packPw(efcW, C, C);
    pkEx[0] = packPw(e1Wih, 3 * half, half); pkEx[1] = packPw(e2Wih, 3 * half, half);
    pkEh[0] = packPw(e1Whh, 3 * half, half); pkEh[1] = packPw(e2Whh, 3 * half, half);
    auto tr = [&](const float* src, std::vector<float>& dst) { dst.assign((size_t)F * C, 0.0f); for (int f = 0; f < F; ++f) for (int c = 0; c < C; ++c) dst[(size_t)c * F + f] = src[(size_t)f * C + c]; };
    tr(ilnW, ilnWT); tr(ilnB, ilnBT); tr(elnW, elnWT); tr(elnB, elnBT);
}

void DPGRNN::reset() { std::fill(hInter.begin(), hInter.end(), 0.0f); }

// 频率方向 GRU 的递推：输入投影 P（[3H][F]，含 b_ih）已算好，这里只剩 W_hh·h。H 编译期已知时（HH>0）循环全展开。
template <int HH>
static inline void intraRec(const float* __restrict U, const float* __restrict bh, const float* __restrict P, int F, int dir, float* __restrict out, int hrt = 0) {
    const int H = HH > 0 ? HH : hrt; float h[16] = {0}, g[48];
    for (int s = 0; s < F; ++s) {
        const int f = dir ? F - 1 - s : s;
        for (int k = 0; k < 3 * H; ++k) { const float* ur = U + k * H; float a = bh[k]; for (int i = 0; i < H; ++i) a += ur[i] * h[i]; g[k] = a; }
        for (int i = 0; i < H; ++i) {
            const float rg = fsigm(P[(size_t)i * F + f] + g[i]), zg = fsigm(P[(size_t)(H + i) * F + f] + g[H + i]);
            const float ng = ftanh(P[(size_t)(2 * H + i) * F + f] + rg * g[2 * H + i]);
            h[i] = (1.0f - zg) * ng + zg * h[i]; out[(size_t)i * F + f] = h[i];
        }
    }
}

// 频率方向递推，隐层 4（10-13）：两组 × 正反两个方向共 4 条递推互不依赖，放进同一个循环交错算（小核顺序执行，
// 单条递推每步都在等上一步）；一步内 4 个隐单元向量化。各门 = b_hh + 按隐单元下标依次累加 U·h，再与输入投影相加，
// 式子与 intraRec<4> 相同。P4：4 条各自的输入投影 [3·4][F]；uT：[条][门][i][4]；bh4：[条][门][4]；out[k]：第 k 条的输出行首。
static void intraRec4x(const float* __restrict uT, const float* __restrict bh4, const float* __restrict P4, int F, float* const* out) {
    v4f h[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    const size_t PS = (size_t)12 * F;
    for (int s = 0; s < F; ++s) {
        for (int k = 0; k < 4; ++k) {
            const int f = (k & 1) ? F - 1 - s : s;                 // k = 组·2 + 方向
            const float* u = uT + k * 48; const float* bb = bh4 + k * 12; const float* P = P4 + k * PS;
            float hs[4]; memcpy(hs, &h[k], 16);
            v4f g[3];
            for (int q = 0; q < 3; ++q) { v4f a = ld4(bb + q * 4); for (int i = 0; i < 4; ++i) a += hs[i] * ld4(u + (q * 4 + i) * 4); g[q] = a; }
            const v4f pr = {P[(size_t)0 * F + f], P[(size_t)1 * F + f], P[(size_t)2 * F + f], P[(size_t)3 * F + f]};
            const v4f pz = {P[(size_t)4 * F + f], P[(size_t)5 * F + f], P[(size_t)6 * F + f], P[(size_t)7 * F + f]};
            const v4f pn = {P[(size_t)8 * F + f], P[(size_t)9 * F + f], P[(size_t)10 * F + f], P[(size_t)11 * F + f]};
            const v4f rg = vsigm(pr + g[0]), zg = vsigm(pz + g[1]), ng = vtanh(pn + rg * g[2]);
            h[k] = (1.0f - zg) * ng + zg * h[k];
            float o[4]; memcpy(o, &h[k], 16); float* ob = out[k];
            ob[f] = o[0]; ob[(size_t)F + f] = o[1]; ob[(size_t)2 * F + f] = o[2]; ob[(size_t)3 * F + f] = o[3];
        }
    }
}

// 全部在 (C,F) 布局里算（原写法前后各做一次整块转置）：输入投影、门值、两个全连接都是 [行][F] 的逐点卷积（pw），
// 时间方向 GRU 在 F 个频点上一次算完（逐元素、可向量化）；频率方向 GRU 的递推里只剩隐状态那一小块。
// 各量的计算式与累加顺序同旧写法；LayerNorm 对全部 F×C 个元素求均值方差，与布局无关（权重按 (C,F) 预先转置）。
void DPGRNN::run(float* x) {
    OPT(kOpDp);
    const int half = C / 2, hh = half / 2;
    float* __restrict P = pj.data();                     // [3·hh][F]
    float* __restrict ct = cat.data();                   // [C][F]
    // --- intra：频率方向，双向，两路各吃一半通道 ---
    if (hh == 4) {
        float* outs[4];
        for (int r = 0; r < 2; ++r) {
            const float* X = x + (size_t)r * half * F;
            for (int dir = 0; dir < 2; ++dir) {
                const int k = r * 2 + dir;
                const float* W = r ? (dir ? i2WihR : i2Wih) : (dir ? i1WihR : i1Wih); const float* bi = r ? (dir ? i2bihR : i2bih) : (dir ? i1bihR : i1bih);
                { OPT(kOpDpPw); pw(W, bi, X, pj4.data() + (size_t)k * 12 * F, 12, half, F, nullptr, 0, pkIntra[k].data()); }
                outs[k] = ct + (size_t)(r * 2 * hh + dir * hh) * F;
            }
        }
        { OPT(kOpDpRec); intraRec4x(uT4.data(), bh4.data(), pj4.data(), F, outs); }
    } else
    for (int r = 0; r < 2; ++r) {
        const float* X = x + (size_t)r * half * F;
        const float* Wf = r ? i2Wih : i1Wih; const float* Uf = r ? i2Whh : i1Whh; const float* bf = r ? i2bih : i1bih; const float* cf = r ? i2bhh : i1bhh;
        const float* Wb = r ? i2WihR : i1WihR; const float* Ub = r ? i2WhhR : i1WhhR; const float* bb = r ? i2bihR : i1bihR; const float* cb = r ? i2bhhR : i1bhhR;
        for (int dir = 0; dir < 2; ++dir) {
            const float* W = dir ? Wb : Wf; const float* U = dir ? Ub : Uf; const float* bi = dir ? bb : bf; const float* bhh_ = dir ? cb : cf;
            { OPT(kOpDpPw); pw(W, bi, X, P, 3 * hh, half, F, nullptr, 0, pkIntra[r * 2 + dir].data()); }            // 整条序列的输入投影
            float* out = ct + (size_t)(r * 2 * hh + dir * hh) * F;
            { OPT(kOpDpRec); if (hh == 4) intraRec<4>(U, bhh_, P, F, dir, out); else intraRec<0>(U, bhh_, P, F, dir, out, hh); }
        }
    }
    { OPT(kOpDpPw); pw(ifcW, ifcB, ct, xfc.data(), C, C, F, nullptr, 0, pkIfc.data()); }
    { OPT(kOpDpLn); layerNorm(xfc.data(), F, C, ilnWT.data(), ilnBT.data(), 1e-8f); }
    for (size_t i = 0; i < (size_t)F * C; ++i) x[i] += xfc[i];

    // --- inter：时间方向，单向，每个频点一份状态；T=1 所以就是一步，F 个频点一起算 ---
    float* __restrict Gx = gx.data(); float* __restrict Gh = gh.data();   // [3·half][F]
    for (int r = 0; r < 2; ++r) {
        const float* X = x + (size_t)r * half * F; float* __restrict Hs = hInter.data() + (size_t)r * half * F;
        { OPT(kOpDpPw); pw(r ? e2Wih : e1Wih, r ? e2bih : e1bih, X, Gx, 3 * half, half, F, nullptr, 0, pkEx[r].data());
        pw(r ? e2Whh : e1Whh, r ? e2bhh : e1bhh, Hs, Gh, 3 * half, half, F, nullptr, 0, pkEh[r].data()); }
        OPT(kOpDpGate);
        for (int i = 0; i < half; ++i) {
            const float *xr = Gx + (size_t)i * F, *xz = Gx + (size_t)(half + i) * F, *xn = Gx + (size_t)(2 * half + i) * F;
            const float *hr = Gh + (size_t)i * F, *hz = Gh + (size_t)(half + i) * F, *hn = Gh + (size_t)(2 * half + i) * F;
            float* hs = Hs + (size_t)i * F;
            for (int f = 0; f < F; ++f) { const float rg = fsigm(xr[f] + hr[f]), zg = fsigm(xz[f] + hz[f]), ng = ftanh(xn[f] + rg * hn[f]); hs[f] = (1.0f - zg) * ng + zg * hs[f]; }
        }
    }
    { OPT(kOpDpPw); pw(efcW, efcB, hInter.data(), xfc.data(), C, C, F, nullptr, 0, pkEfc.data()); }
    { OPT(kOpDpLn); layerNorm(xfc.data(), F, C, elnWT.data(), elnBT.data(), 1e-8f); }
    for (size_t i = 0; i < (size_t)F * C; ++i) x[i] += xfc[i];
}

}  // namespace ulk

namespace ulk {

// ---- 网络装配 ----------------------------------------------------------
//
// 结构照 ulunas.py 的构造参数逐条对：
//   types=[0,2,1,2,1] strides=[2,2,1,1,1] groups=[1,2,2,2,2]
//   channels=[12,24,24,32,16] kernels=[(3,3),(2,3),(2,3),(1,5),(1,5)] widths=[65,33,33,33,33]
// 解码器倒着建（i=4..1 用 types[i]，末块用 types[0]），全部 use_deconv。
// **stride==2 时 AffinePReLU 的宽度不是 width**：编码器是 width*2-1，
// 解码器是 width/2+1 —— 这两行照 XDWSBlock/XMBBlocks 里的 in_width 抄。

static std::vector<int> shuffleSrcToDst(int C2) {   // shuffle2：源行 c → 目的行 2c，源行 C+c → 目的行 2c+1
    std::vector<int> d(C2); const int C = C2 / 2; for (int c = 0; c < C; ++c) { d[c] = 2 * c; d[C + c] = 2 * c + 1; } return d;
}
static std::vector<int> shuffleDstToSrc(int C2) {   // 反过来：目的行 2c ← 源行 c，2c+1 ← C+c
    std::vector<int> m(C2); const int C = C2 / 2; for (int c = 0; c < C; ++c) { m[2 * c] = c; m[2 * c + 1] = C + c; } return m;
}
// 激活并进卷积输出后按卷积的输出行宽逐行做，两者的形状必须一致
static void checkAct(const Conv& cv, const AffinePReLU& a) {
    if (a.C != cv.cout || a.nF != cv.fout) throw std::runtime_error("ans48: activation shape does not match conv output");
}
// 深度卷积能把前面的洗牌并进读取（只在会走快速路径时）
static void mapInput(Conv& dc, int C2) { if (dc.sf == 2 || dc.fin == dc.fout) dc.inMap = shuffleDstToSrc(C2); }

static void mkXConv(XConvBlock* B, const Weights& w, const std::string& pfx,
                    int cin, int cout, int width, int kt, int kf, int sf,
                    int groups, int fin, bool deconv, bool isLast) {
    B->conv.prepare(w, pfx + ".ops.1", cin, cout, kt, kf, sf, groups, fin, deconv);
    B->cout = cout; B->fout = B->conv.fout;
    B->useAct = !isLast;
    if (B->useAct) { B->act.prepare(w, pfx + ".ops.3", cout, width); checkAct(B->conv, B->act); }
    B->ctfa.prepare(w, pfx + ".ops.4", cout, width);
    B->useShuffle = (!isLast && groups == 2);
    if (B->useShuffle) B->perm = shuffleSrcToDst(cout);
    B->tmp.assign((size_t)cout * B->fout, 0.0f);
}

static void mkXDWS(XDWSBlock* B, const Weights& w, const std::string& pfx,
                   int cin, int cout, int width, int kt, int kf, int sf,
                   int groups, int fin, bool deconv) {
    const int inWidth = (sf == 2) ? (deconv ? width / 2 + 1 : width * 2 - 1) : width;
    B->pconv.prepare(w, pfx + ".pconv.0", cin, cout, 1, 1, 1, groups, fin, false);
    B->cmid = cout; B->fmid = B->pconv.fout;
    B->act1.prepare(w, pfx + ".pconv.2", cout, inWidth);
    B->shuffleP = (groups == 2);
    B->dconv.prepare(w, pfx + ".dconv.1", cout, cout, kt, kf, sf, cout, B->fmid, deconv);
    B->cout = cout; B->fout = B->dconv.fout;
    B->act2.prepare(w, pfx + ".dconv.3", cout, width);
    checkAct(B->pconv, B->act1); if (B->useAct2) checkAct(B->dconv, B->act2);
    B->ctfa.prepare(w, pfx + ".dconv.4", cout, width);
    if (B->shuffleP) mapInput(B->dconv, cout);
    B->t1.assign((size_t)B->cmid * B->fmid, 0.0f);
    B->t2.assign((size_t)B->cmid * B->fmid, 0.0f);
}

static void mkXMB(XMBBlock* B, const Weights& w, const std::string& pfx,
                  int cin, int cout, int width, int kt, int kf, int sf,
                  int groups, int fin, bool deconv, bool isLast) {
    const int inWidth = (sf == 2) ? (deconv ? width / 2 + 1 : width * 2 - 1) : width;
    B->cin = cin; B->fin = fin;
    B->pconv1.prepare(w, pfx + ".pconv1.0", cin, cout, 1, 1, 1, groups, fin, false);
    B->cmid = cout; B->fmid = B->pconv1.fout;
    B->act1.prepare(w, pfx + ".pconv1.2", cout, inWidth);
    B->shuffleP = (groups == 2);
    B->dconv.prepare(w, pfx + ".dconv.1", cout, cout, kt, kf, sf, cout, B->fmid, deconv);
    B->act2.prepare(w, pfx + ".dconv.3", cout, width);
    checkAct(B->pconv1, B->act1); checkAct(B->dconv, B->act2);
    B->pconv2.prepare(w, pfx + ".pconv2.0", cout, cout, 1, 1, 1, groups, B->dconv.fout, false);
    B->cout = cout; B->fout = B->pconv2.fout;
    B->ctfa.prepare(w, pfx + ".pconv2.2", cout, width);
    B->shuffleOut = (!isLast && groups == 2);
    if (B->shuffleP) mapInput(B->dconv, cout);
    if (B->shuffleOut) B->perm = shuffleSrcToDst(cout);
    B->t1.assign((size_t)B->cmid * B->fmid, 0.0f);
    B->t2.assign((size_t)B->cmid * B->fmid, 0.0f);
    B->t3.assign((size_t)cout * B->dconv.fout, 0.0f);
    B->t4.assign((size_t)B->cout * B->fout, 0.0f);
}

bool Net::prepare() {
    bins = w.bins;
    // **宽度和 ERB 分界从权重形状读，不写死。**
    // 窗长不同的模型 ERB 分界与各级宽度不同（例如窗 128 为 33/32、33/17/17/17/17），写死 65/33 会接不上。
    // 各级宽度 = 该级卷积后那个 AffinePReLU 的第二维；ERB 分界由 erb_fc 的形状定。
    auto widthOf = [&](const char* n) { return w.at(n).shape.back(); };
    const int W0 = widthOf("encoder.en_convs.0.ops.3.affine_weight");
    const int W1 = widthOf("encoder.en_convs.1.dconv.3.affine_weight");
    const int W2 = widthOf("encoder.en_convs.2.dconv.3.affine_weight");
    const int W3 = widthOf("encoder.en_convs.3.dconv.3.affine_weight");
    const int W4 = widthOf("encoder.en_convs.4.dconv.3.affine_weight");
    {
        const Tensor& e = w.at("erb.erb_fc.weight");     // (erb_high, nfreq - erb_low)
        erbHigh = e.shape[0];
        erbLow = bins - e.shape[1];
        if (e.shape.size() != 2 || erbLow <= 0 || erbHigh <= 0) throw std::runtime_error("ans48: bad erb shape");
    }
    erbW = w.at("erb.erb_fc.weight").p; ierbW = w.at("erb.ierb_fc.weight").p;
    auto nzRange = [](const float* M, int R, int Cn, std::vector<int>& S, std::vector<int>& E) {
        S.assign(R, 0); E.assign(R, 0);
        for (int r = 0; r < R; ++r) { int a = Cn, b = 0; for (int c = 0; c < Cn; ++c) if (M[(size_t)r * Cn + c] != 0.0f) { a = std::min(a, c); b = c + 1; } S[r] = a < b ? a : 0; E[r] = a < b ? b : 0; }
    };
    nzRange(erbW, erbHigh, bins - erbLow, erbS, erbE); nzRange(ierbW, bins - erbLow, erbHigh, ierbS, ierbE);   // 每帧不再按名字查（名字超过短串长度，查一次就是一次堆分配）
    const int WF = erbLow + erbHigh;                     // 末块的 final_width
    // 首块输入宽度 = ERB 压缩后的特征宽度（erbLow + erbHigh），不是频点数 bins。
    // 窗 256 / 128 两者恰好相等（129 / 65），窗 1024 不相等（513 ≠ 129）。
    const int F0 = WF;

    mkXConv(&en0, w, "encoder.en_convs.0", 1, 12, W0, 3, 3, 2, 1, F0, false, false);
    mkXMB (&en1, w, "encoder.en_convs.1", 12, 24, W1, 2, 3, 2, 2, en0.fout, false, false);
    mkXDWS(&en2, w, "encoder.en_convs.2", 24, 24, W2, 2, 3, 1, 2, en1.fout, false);
    mkXMB (&en3, w, "encoder.en_convs.3", 24, 32, W3, 1, 5, 1, 2, en2.fout, false, false);
    mkXDWS(&en4, w, "encoder.en_convs.4", 32, 16, W4, 1, 5, 1, 2, en3.fout, false);
    dp0.prepare(w, "dpgrnn.0", 16, W4);
    dp1.prepare(w, "dpgrnn.1", 16, W4);
    mkXDWS(&de0, w, "decoder.de_convs.0", 16, 32, W3, 1, 5, 1, 2, en4.fout, true);
    mkXMB (&de1, w, "decoder.de_convs.1", 32, 24, W2, 1, 5, 1, 2, de0.fout, true, false);
    mkXDWS(&de2, w, "decoder.de_convs.2", 24, 24, W1, 2, 3, 1, 2, de1.fout, true);
    mkXMB (&de3, w, "decoder.de_convs.3", 24, 12, W0, 2, 3, 2, 2, de2.fout, true, false);
    mkXConv(&de4, w, "decoder.de_convs.4", 12, 1, WF, 3, 3, 2, 1, de3.fout, true, true);

    feat.assign(F0, 0.0f);
    e0.assign((size_t)en0.cout * en0.fout, 0.0f);
    e1.assign((size_t)en1.cout * en1.fout, 0.0f);
    e2.assign((size_t)en2.cout * en2.fout, 0.0f);
    e3.assign((size_t)en3.cout * en3.fout, 0.0f);
    e4.assign((size_t)en4.cout * en4.fout, 0.0f);
    d.assign((size_t)std::max(std::max(de0.cout * de0.fout, de1.cout * de1.fout),
                              std::max(de2.cout * de2.fout, de3.cout * de3.fout)), 0.0f);
    skipTmp.assign(d.size() + F0, 0.0f);
    skipAdd.assign(d.size() + F0, 0.0f);
    lastOut.assign((size_t)de4.fout, 0.0f);
    return true;
}

void Net::reset() {
    en0.reset(); en1.reset(); en2.reset(); en3.reset(); en4.reset();
    dp0.reset(); dp1.reset();
    de0.reset(); de1.reset(); de2.reset(); de3.reset(); de4.reset();
}

const char* Profile::name(int i) {
    static const char* n[Profile::kSlots] = {
        "erb.bm", "en0", "en1", "en2", "en3", "en4",
        "dpgrnn0", "dpgrnn1", "de0", "de1", "de2", "de3", "de4+bs"};
    return (i >= 0 && i < Profile::kSlots) ? n[i] : "?";
}

static inline long long nowNs() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

void Net::run(const float* logmag, float* mask) {
    long long t0 = profiling ? nowNs() : 0;
    int slot = 0;
    auto tick = [&]() {
        if (!profiling) return;
        long long t = nowNs();
        prof.ns[slot++] += t - t0;
        t0 = t;
    };
    // ERB.bm：前 erbLow 个频点原样，其余投影
    memcpy(feat.data(), logmag, sizeof(float) * erbLow);
    {   // ERB 投影：只算每行的非零段（零元素对和没有贡献，结果与满矩阵乘相同）
        const float* xin = logmag + erbLow; const int Cn = bins - erbLow;
        for (int r = 0; r < erbHigh; ++r) { const float* wr = erbW + (size_t)r * Cn; float s = 0.0f; for (int c = erbS[r]; c < erbE[r]; ++c) s += wr[c] * xin[c]; feat[erbLow + r] = s; }
    }
    tick();

    en0.run(feat.data(), e0.data()); tick();
    en1.run(e0.data(), e1.data()); tick();
    en2.run(e1.data(), e2.data()); tick();
    en3.run(e2.data(), e3.data()); tick();
    en4.run(e3.data(), e4.data()); tick();

    // dpgrnn 就地作用在 e4 上；但 e4 还要当跳连用，所以先拷一份
    std::vector<float>& x = skipTmp;
    memcpy(x.data(), e4.data(), sizeof(float) * e4.size());
    dp0.run(x.data()); tick();
    if (keepTaps) tapDp[0].assign(x.data(), x.data() + e4.size());
    dp1.run(x.data()); tick();
    if (keepTaps) tapDp[1].assign(x.data(), x.data() + e4.size());

    auto addSkip = [](float* dst, const float* a, const float* b, size_t n) {
        for (size_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
    };
    float* t = skipAdd.data();
    addSkip(t, x.data(), e4.data(), e4.size());
    de0.run(t, d.data()); tick();
    if (keepTaps) tapDe[0].assign(d.data(), d.data() + (size_t)de0.cout * de0.fout);

    addSkip(t, d.data(), e3.data(), (size_t)en3.cout * en3.fout);
    de1.run(t, d.data()); tick();
    if (keepTaps) tapDe[1].assign(d.data(), d.data() + (size_t)de1.cout * de1.fout);

    addSkip(t, d.data(), e2.data(), (size_t)en2.cout * en2.fout);
    de2.run(t, d.data()); tick();
    if (keepTaps) tapDe[2].assign(d.data(), d.data() + (size_t)de2.cout * de2.fout);

    addSkip(t, d.data(), e1.data(), (size_t)en1.cout * en1.fout);
    de3.run(t, d.data()); tick();
    if (keepTaps) tapDe[3].assign(d.data(), d.data() + (size_t)de3.cout * de3.fout);

    addSkip(t, d.data(), e0.data(), (size_t)en0.cout * en0.fout);
    float* last = lastOut.data();
    de4.run(t, last);
    if (keepTaps) tapDe[4].assign(last, last + de4.fout);   // sigmoid 之前（= torch 解码末块输出）

    for (int i = 0; i < de4.fout; ++i) last[i] = fsigm(last[i]);

    // ERB.bs：前 erbLow 原样，其余反投影
    memcpy(mask, last, sizeof(float) * erbLow);
    {   // ERB 反投影：同上
        const float* xin = last + erbLow; const int Rn = bins - erbLow;
        for (int r = 0; r < Rn; ++r) { const float* wr = ierbW + (size_t)r * erbHigh; float s = 0.0f; for (int c = ierbS[r]; c < ierbE[r]; ++c) s += wr[c] * xin[c]; mask[erbLow + r] = s; }
    }
    tick();
    if (profiling) prof.calls++;
}

}  // namespace ulk
