// 网络的对外接口（10-14）：库里可能同时有几份内核（armeabi-v7a 上多一份 -mfpu=neon-vfpv4 编的），
// ans48.cpp 只经这个接口用网络，建实例时由 newNet() 按 CPU 能力选一份（net_select.cpp）。
#pragma once
#include <cstddef>

namespace ulk_if {

struct INet {
    virtual ~INet() {}
    virtual bool load(const void* bin, size_t bytes, const char* json) = 0;   // 从内存读权重
    virtual bool prepare() = 0;                                               // 成功后下面四个数有效
    virtual void reset() = 0;
    virtual void run(const float* logmag, float* mask) = 0;                   // 一帧：bins 个 log 幅度 → bins 个掩码
    int nFft = 0, hop = 0, fs = 0, bins = 0;
};

INet* newNet();   // 失败抛 std::bad_alloc

}  // namespace ulk_if
