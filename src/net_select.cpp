// 建网络实例时按 CPU 能力选内核（10-14）。只有 armeabi-v7a 编了两份（见 ulunas_kernel_vfpv4.cpp），其余 ABI 只有基础版。
#include "net_iface.h"
#if defined(ULK_HAVE_VFPV4) && !defined(ULK_FORCE_BASE)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

namespace ulk { ulk_if::INet* newNetImpl(); }
#if defined(ULK_HAVE_VFPV4) && !defined(ULK_FORCE_BASE)
namespace ulk_vfpv4 { ulk_if::INet* newNetImpl(); }
#endif

ulk_if::INet* ulk_if::newNet() {
#if defined(ULK_HAVE_VFPV4) && !defined(ULK_FORCE_BASE)
    const unsigned long h = getauxval(AT_HWCAP);   // 内核报的 CPU 特性位；读不到时为 0，落到基础版
    if ((h & HWCAP_NEON) && (h & HWCAP_VFPv4)) return ulk_vfpv4::newNetImpl();
#endif
    return ulk::newNetImpl();
}
