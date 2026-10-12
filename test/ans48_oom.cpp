// 内存不足时 ans48_create 必须返回 NULL、不结束宿主进程（ans48.h 的约定）。
// 做法：在子进程里让建实例过程中的第 k 次 malloc 失败，k 从 1 扫到建实例用到的最后一次；
// 子进程被信号结束（崩溃）即判失败。只在 Linux 主机上编（借 glibc 的 __libc_malloc 转发）。
// 用法：ans48_oom
#include "ans48.h"
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern "C" void* __libc_malloc(size_t);
static long gFailAt = -1, gCount = 0;
extern "C" void* malloc(size_t n) {   // 覆盖 malloc（operator new、pffft 的对齐分配都走它）
    if (gFailAt >= 0 && ++gCount == gFailAt) return nullptr;
    return __libc_malloc(n);
}

// 子进程退出码：0 = 建成且能处理（k 超出了分配次数），2 = 返回 NULL，3 = 建成但分配确实失败过（失败被吞掉）
static int child(int sr, int mode, long k) {
    gCount = 0; gFailAt = k;
    Ans48* a = ans48_create(sr, mode);
    const long used = gCount; gFailAt = -1;
    if (!a) return 2;
    std::vector<float> x(4800, 0.01f), y(4800);
    ans48_process(a, x.data(), y.data(), (int)x.size());
    ans48_destroy(a);
    return used >= k ? 3 : 0;
}

int main() {
    int bad = 0;
    for (int sr : {48000, 44100}) for (int mode : {ANS48_MODE_MONITOR, ANS48_MODE_STREAM}) {
        long k = 1, nulls = 0, swallowed = 0; const int bad0 = bad;
        for (;; ++k) {
            fflush(stdout);
            const pid_t p = fork();
            if (p == 0) _exit(child(sr, mode, k));
            int st = 0; waitpid(p, &st, 0);
            if (WIFSIGNALED(st)) { printf("不通过  %d Hz %s：第 %ld 次分配失败时进程被信号 %d 结束\n", sr, mode ? "推流" : "耳返", k, WTERMSIG(st)); ++bad; continue; }
            const int rc = WEXITSTATUS(st);
            if (rc == 0) break;
            if (rc == 2) ++nulls; else if (rc == 3) ++swallowed;
            else { printf("不通过  %d Hz %s：第 %ld 次，子进程退出码 %d\n", sr, mode ? "推流" : "耳返", k, rc); ++bad; }
            if (k > 100000) { printf("不通过：分配次数超过 10 万，停止\n"); ++bad; break; }
        }
        printf("%s  %d Hz %s：建实例共 %ld 次分配，逐次让其失败：返回 NULL %ld 次，失败被吞掉但实例可用 %ld 次\n",
               bad > bad0 ? "不通过" : (swallowed ? "注意" : "通过"), sr, mode ? "推流" : "耳返", k - 1, nulls, swallowed);
    }
    printf(bad ? "有问题：不通过 %d 项\n" : "全部通过：不通过 %d 项\n", bad);
    return bad ? 1 : 0;
}
