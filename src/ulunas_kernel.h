// UL-UNAS 结构降噪网络的流式推理（手写实现）。窗长、频点数、ERB 分界与各级宽度都从权重清单读，不写死。
//
// 不用通用推理框架的原因：逐帧推理时每帧算力很小（窗 256 约 0.4 MMACs），而图里的形状搬运节点很多，
// 框架的算子调度开销远大于实际计算。这里把整张图写成一段直筒代码：没有算子调度、没有中间张量分配，
// 所有缓冲在 prepare() 里一次分配好。T 恒为 1（逐帧），因果卷积的时间维靠历史缓冲，GRU 靠显式状态。
//
// 正确性靠逐块核对：权重导出时同时录下参照网络每一块在两帧输入下的输出，与本实现逐块比较。
#pragma once
#include <algorithm>
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
    // 向量化用的转置权重（10-13）：每个方向、每个门 (r,z,n) 一组 4 列，每列 4 个数 = 该门 4 个隐单元对第 i 个输入的权重
    std::vector<float> wx, wh, bx, bh, fcT;   // wx/wh: [方向 2][门 3][输入 4][4]；bx/bh: [方向 2][门 3][4]；fcT: [输入 8][4]
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
    // out 非空：结果不写回 x，而是写到 out 的第 outRow[c] 行，并加上 res（残差，可空）——把输出洗牌和残差并进最后一遍
    void run(float* x /*就地，(C,nF)*/, float* out = nullptr, const int* outRow = nullptr, const float* res = nullptr);
};

/** 逐通道逐频点仿射 + PReLU。 */
struct AffinePReLU {
    int C = 0, nF = 0;
    const float *aw = nullptr, *ab = nullptr, *slope = nullptr;
    void prepare(const Weights& w, const std::string& pfx, int channels, int nFreq);
    void run(float* x) const;
    // 只处理第 c 个通道的一行（卷积写完一行就地做，省一遍整块读写）
    inline void row(float* __restrict r, int c) const {
        const float* __restrict aws = aw + (size_t)c * nF; const float* __restrict abs_ = ab + (size_t)c * nF; const float sl = slope[c];
        for (int f = 0; f < nF; ++f) { const float v = r[f]; r[f] = aws[f] * v + abs_[f] + (std::max(v, 0.0f) + sl * std::min(v, 0.0f)); }   // 无分支（= v > 0 ? v : sl·v）
    }
};

constexpr int kMaxTaps = 16;   // 深度卷积 kt×kf 的上限（模型里最大 2×3 / 1×5；prepare 里校验）
/** 二维卷积，时间维 kt（吃历史缓冲）、频率维 kf（补零 kf/2）、频率步长 sf。 */
struct Conv {
    std::vector<float> zb;   // 无偏置时的零偏置（逐点快速路径用）
    std::vector<std::vector<float>> pwPack;   // 逐点快速路径：每组一份转置打包的权重（packPw）
    std::vector<float> padr; // 深度卷积 / 步长 2 快速路径的补零行
    std::vector<float> accr; // 步长 2 转置卷积：奇偶两条输出子序列的暂存
    std::vector<float> padAll;          // 步长 2：全部输入行（通道 × 时间帧）补零后放一起，各输出通道共用
    int padL = 0, padRowLen = 0;
    std::vector<float> tapW; std::vector<const float*> tapP;   // 步长 2：一个输出通道的全部抽头（权重 + 对应输入位置）
    std::vector<int> inMap;  // 非空：第 c 个输入通道取输入的第 inMap[c] 行（把前面的通道洗牌并进读取，只有深度卷积 / 步长 2 快速路径支持）
    int cin = 0, cout = 0, kt = 1, kf = 1, sf = 1, groups = 1;
    int fin = 0, fout = 0;
    bool deconv = false;
    const float* W = nullptr;
    const float* b = nullptr;
    std::vector<float> hist;      // (kt-1, cin, fin) 历史帧
    void prepare(const Weights& w, const std::string& pfx, int cinN, int coutN,
                 int ktN, int kfN, int sfN, int groupsN, int finN, bool isDeconv);
    void reset();
    // post 非空：输出每写完一行就地接激活（省掉激活单独一遍整块读写）
    void run(const float* x /*(cin,fin)*/, float* y /*(cout,fout)*/, const AffinePReLU* post = nullptr);
};

/** 逐点卷积权重转置打包（见 ulunas_kernel.cpp 的 pw）。 */
std::vector<float> packPw(const float* w, int O, int I);

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
    std::vector<int> perm;   // 同 XMBBlock::perm
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
    std::vector<int> perm;   // shuffle2 的行映射：输出第 2c 行 ← 第 c 行，第 2c+1 行 ← 第 C+c 行（这里存反向：源行 → 目的行）
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
    std::vector<float> hInter, xfc, cat;
    // (C,F) 布局版用：输入投影 / 门值 [3H][F]、两个 LayerNorm 的转置权重（原权重按 (F,C) 排）
    std::vector<float> pj, gx, gh, ilnWT, ilnBT, elnWT, elnBT;
    std::vector<float> pkIntra[4], pkIfc, pkEfc, pkEx[2], pkEh[2];
    std::vector<float> pj4, uT4, bh4;   // 隐层 4 时：4 条频率递推各自的输入投影、转置的 U、b_hh（intraRec4x）   // 各逐点投影 / 全连接的转置打包权重（packPw）
    void prepare(const Weights& w, const std::string& pfx, int channels, int width);
    void reset();
    void run(float* x /*就地 (C,F)*/);
};

/** 逐块耗时（纳秒累计）。**先量再优化**——这个图里 17 个卷积、26 个矩阵乘、
 *  28 个 GRU，凭直觉挑错地方优化会白干一整轮。 */
// 按算子类别计时的类别（只在 -DULK_OPPROF 构建里计数）
enum { kOpPw = 0, kOpConv, kOpTa, kOpFa, kOpMul, kOpDp, kOpTop, kOpDpRec = kOpTop, kOpDpPw, kOpDpLn, kOpDpGate, kOpCats };   // kOpTop 之后的是 DPGRNN 内部的细分（嵌套，不另加总）
#ifdef ULK_OPPROF
extern long long g_opns[kOpCats];
#endif

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
    std::vector<int> erbS, erbE, ierbS, ierbE;                   // 两个矩阵每行非零元素的起止（三角滤波器很窄，大部分是 0）
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
