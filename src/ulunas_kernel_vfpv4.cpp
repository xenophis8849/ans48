// armeabi-v7a 专用（10-14）：同一份内核以 -mfpu=neon-vfpv4 再编一份（CMake 给这个文件单独加该选项），放进 ulk_vfpv4 命名空间。
// VFPv4 带融合乘加（vfma），-ffast-math 下编译器把 a·b + c 合成一条；不带它时每次乘加拆两条。小核按节奏实测快 6–10%。
// Cortex-A8 / A9 等老核没有 VFPv4，跑到 vfma 会非法指令崩溃，所以不能整库这么编，只能运行时按 HWCAP 选（net_select.cpp）。
// 输出与基础版不逐位相同（融合乘加少一次舍入），差在 −100 dB 以下。
#define ULK_NS ulk_vfpv4
#include "ulunas_kernel.cpp"
