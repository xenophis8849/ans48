// UL-UNAS 结构降噪网络的流式推理（手写实现）。窗长、频点数、ERB 分界与各级宽度都从权重清单读，不写死。
//
// 不用通用推理框架的原因：逐帧推理时每帧算力很小（窗 256 约 0.4 MMACs），而图里的形状搬运节点很多，
// 框架的算子调度开销远大于实际计算。这里把整张图写成一段直筒代码：没有算子调度、没有中间张量分配，
// 所有缓冲在 prepare() 里一次分配好。T 恒为 1（逐帧），因果卷积的时间维靠历史缓冲，GRU 靠显式状态。
//
// 正确性靠逐块核对：权重导出时同时录下参照网络每一块在两帧输入下的输出，与本实现逐块比较。
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>

namespace ulk {

struct Tensor {
    const float* p = nullptr;
    int64_t n = 0;
    std::vector<int> shape;
    float operator[](int i) const { return p[i]; }
};

/** 权重包：一段连续 float + 名字索引（权重导出工具产出）。 */
struct Weights {
    std::vector<float> data;
    std::unordered_map<std::string, Tensor> byName;
    bool load(const std::string& binPath, const std::string& jsonPath);
    bool loadFromMemory(const void* bin, size_t binBytes,
                        const std::string& jsonText);
    const Tensor& at(const std::string& n) const;
    bool has(const std::string& n) const { return byName.count(n) != 0; }
    int nFft = 256, hop = 128, fs = 48000, bins = 129;
};

// ---- 基本算子（全部 T=1，形状在编译期不定但在 prepare 期固定）----------

/** 单步 GRU。PyTorch 的门序是 r, z, n。
 *  n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))  —— **b_hn 在括号里面**，
 *  写到外面是最常见的 GRU 移植错误。 */
void gruStep(const float* x, int nIn, float* h, int nHid,
             const float* Wih, const float* Whh, const float* bih, const float* bhh,
             float* gates /*3*nHid 暂存*/);

/** 双向 GRU 扫一条序列（用于频率方向）。out 为 (len, 2*nHid)。 */
void gruBidir(const float* x, int len, int nIn, int nHid,
              const float* Wih, const float* Whh, const float* bih, const float* bhh,
              const float* WihR, const float* WhhR, const float* bihR, const float* bhhR,
              float* out, float* scratch);

/** y = W·x + b。W 为 (out, in) 行主序。 */
void linear(const float* x, int nIn, const float* W, const float* b, int nOut, float* y);

/** 频率注意力 FA：对通道求平方均值 → 补齐到 r 的倍数 → 双向 GRU → fc → 裁回。 */
struct FA {
    int nF = 0, padLen = 0, H = 0, r = 4;
    const float *Wih = nullptr, *Whh = nullptr, *bih = nullptr, *bhh = nullptr;
    const float *WihR = nullptr, *WhhR = nullptr, *bihR = nullptr, *bhhR = nullptr;
    const float *fcW = nullptr, *fcB = nullptr;
    std::vector<float> pad, gruOut, scratch;
    void prepare(const Weights& w, const std::string& pfx, int nFreq);
    void run(const float* x, int C, float* out /*nF*/);
};

/** 因果时频注意力 cTFA。ta_gru 是**时间方向**的，所以带状态。 */
struct CTFA {
    int C = 0, nF = 0, hid = 0;
    const float *Wih = nullptr, *Whh = nullptr, *bih = nullptr, *bhh = nullptr;
    const float *fcW = nullptr, *fcB = nullptr;
    std::vector<float> h, zt, at, gates, af;
    FA fa;
    void prepare(const Weights& w, const std::string& pfx, int channels, int nFreq);
    void reset();
    void run(float* x /*就地，(C,nF)*/);
};

/** 逐通道逐频点仿射 + PReLU。 */
struct AffinePReLU {
    int C = 0, nF = 0;
    const float *aw = nullptr, *ab = nullptr, *slope = nullptr;
    void prepare(const Weights& w, const std::string& pfx, int channels, int nFreq);
    void run(float* x);
};

/** 二维卷积，时间维 kt（吃历史缓冲）、频率维 kf（补零 kf/2）、频率步长 sf。 */
struct Conv {
    int cin = 0, cout = 0, kt = 1, kf = 1, sf = 1, groups = 1;
    int fin = 0, fout = 0;
    bool deconv = false;
    const float* W = nullptr;
    const float* b = nullptr;
    std::vector<float> hist;      // (kt-1, cin, fin) 历史帧
    void prepare(const Weights& w, const std::string& pfx, int cinN, int coutN,
                 int ktN, int kfN, int sfN, int groupsN, int finN, bool isDeconv);
    void reset();
    void run(const float* x /*(cin,fin)*/, float* y /*(cout,fout)*/);
};

/** 通道洗牌：(2C,F) 交错两半。 */
void shuffle2(const float* x, int C2, int nF, float* y);

/** LayerNorm，对 (F,C) 两维一起归一化，逐元素仿射。 */
void layerNorm(float* x, int nF, int C, const float* w, const float* b, float eps);

// ---- 组合块 ----------------------------------------------------------

struct XConvBlock {   // ZeroPad → conv → (bn 已折叠) → AffinePReLU → cTFA → Shuffle?
    Conv conv; AffinePReLU act; CTFA ctfa;
    bool useAct = true, useShuffle = false;
    int cout = 0, fout = 0;
    std::vector<float> tmp;
    void reset();
    void run(const float* x, float* y);
};

struct XDWSBlock {    // pconv(1×1) → AffinePReLU → shuffle? → dconv → AffinePReLU → cTFA
    Conv pconv, dconv; AffinePReLU act1, act2; CTFA ctfa;
    bool shuffleP = false, useAct2 = true;
    int cmid = 0, fmid = 0, cout = 0, fout = 0;
    std::vector<float> t1, t2;
    void reset();
    void run(const float* x, float* y);
};

struct XMBBlock {     // pconv1 → act → shuffle? → dconv → act → pconv2 → cTFA → (+res) → shuffle?
    Conv pconv1, dconv, pconv2; AffinePReLU act1, act2; CTFA ctfa;
    bool shuffleP = false, shuffleOut = false;
    int cin = 0, fin = 0, cmid = 0, fmid = 0, cout = 0, fout = 0;
    std::vector<float> t1, t2, t3, t4;
    void reset();
    void run(const float* x, float* y);
};

struct DPGRNN {
    int C = 0, F = 0, hid = 0;
    // intra：频率方向双向；inter：时间方向单向（带状态）
    const float *i1Wih = nullptr, *i1Whh = nullptr, *i1bih = nullptr, *i1bhh = nullptr;
    const float *i1WihR = nullptr, *i1WhhR = nullptr, *i1bihR = nullptr, *i1bhhR = nullptr;
    const float *i2Wih = nullptr, *i2Whh = nullptr, *i2bih = nullptr, *i2bhh = nullptr;
    const float *i2WihR = nullptr, *i2WhhR = nullptr, *i2bihR = nullptr, *i2bhhR = nullptr;
    const float *ifcW = nullptr, *ifcB = nullptr, *ilnW = nullptr, *ilnB = nullptr;
    const float *e1Wih = nullptr, *e1Whh = nullptr, *e1bih = nullptr, *e1bhh = nullptr;
    const float *e2Wih = nullptr, *e2Whh = nullptr, *e2bih = nullptr, *e2bhh = nullptr;
    const float *efcW = nullptr, *efcB = nullptr, *elnW = nullptr, *elnB = nullptr;
    // 全部预分配：处理期不许分配内存。
    std::vector<float> hInter, xfc, y1, y2, cat, tmp, gates, scratch, in1, in2, o1, o2;
    void prepare(const Weights& w, const std::string& pfx, int channels, int width);
    void reset();
    void run(float* x /*就地 (C,F)*/);
};

/** 逐块耗时（纳秒累计）。**先量再优化**——这个图里 17 个卷积、26 个矩阵乘、
 *  28 个 GRU，凭直觉挑错地方优化会白干一整轮。 */
struct Profile {
    static constexpr int kSlots = 13;
    long long ns[kSlots] = {0};
    long long calls = 0;
    static const char* name(int i);
    void reset() { for (int i = 0; i < kSlots; ++i) ns[i] = 0; calls = 0; }
};

/** 整个网络。输入一帧 log10|spec|（bins 个），输出一帧掩码（bins 个）。 */
struct Net {
    Weights w;
    int bins = 129, erbLow = 65, erbHigh = 64;
    const float* erbW = nullptr; const float* ierbW = nullptr;   // ERB 投影 / 反投影矩阵（prepare 时取好）
    // 编码器 types=[0,2,1,2,1] → XConv, XMB, XDWS, XMB, XDWS
    XConvBlock en0; XMBBlock en1; XDWSBlock en2; XMBBlock en3; XDWSBlock en4;
    DPGRNN dp0, dp1;
    // 解码器倒着建：i=4..1 用 types[i]，最后一块用 types[0]
    // → XDWS, XMB, XDWS, XMB, XConv（全部 use_deconv）
    XDWSBlock de0; XMBBlock de1; XDWSBlock de2; XMBBlock de3; XConvBlock de4;
    std::vector<float> feat, e0, e1, e2, e3, e4, d, skipTmp, skipAdd, lastOut;
    std::vector<std::vector<float>> enOut;
    Profile prof;
    bool profiling = false;
    // 只给逐块核对用：打开后把 DPGRNN 两块与解码五块的输出另存（编码五块本来就在 e0–e4）。会分配内存，音频回调里别开。
    bool keepTaps = false;
    std::vector<float> tapDp[2], tapDe[5];
    bool prepare();
    void reset();
    void run(const float* logmag, float* mask);
};

}  // namespace ulk
