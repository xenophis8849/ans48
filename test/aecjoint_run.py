#!/usr/bin/env python3
"""与 aec48 联调的批量跑与汇总（README「联调结果」的数字由它复现）。
素材：公开的 ICASSP 2023 AEC Challenge 盲测集（48 kHz wav，<目录>/{doubletalk,farend-singletalk,nearend-singletalk}/<id>_{mic,lpb}.wav）。
选条：双讲 13 条、远端单讲 6 条、近端单讲 6 条，都是下面写死的清单。运行前逐条核对 mic / lpb 两个文件都在，缺任何一个就退出非零，
  不会自动换成别的条（10-13 挑战式审查：原写法单讲按排序取前 6，目录缺失时会静默少一类）。
用法：python3 test/aecjoint_run.py <ans48_aecjoint 可执行文件> <盲测集目录> <输出目录>
每条写一行 JSON 到 <输出目录>/joint.jsonl，最后打印汇总。

汇总口径（README 引用的就是这些）：
  切换 / 帧 / 样本计数：各条相加。
  挂钩重开后的误差：ans48_aecjoint 对每次重开、每个时间窗（从过渡带之后算起 0–0.1 / 0.1–0.3 / 0.3–0.6 / 0.6–1.0 / 1.0–1.5 s），
    在浮点域累加「与一直挂着的误差能量」和「一直挂着那路的能量」（一直挂着那块低于 −70 dBFS 的近静音块两项都不计），取两者之比（dB）；
    窗内没有可听块或窗超出观察区间记为空。每次重开都出一行（含观察不足的），这里对全部条、全部重开的同一时间窗取中位数、
    90% 分位与最大值，空值不计。
  收敛分类：观察区间是过渡带之后到下一次关的过渡带之前（每 2 s 切一次，约 1.94 s），与误差时间窗（到 1.5 s）不是同一个。
  电平：各路在同一区间上比（取各路共同长度）。
  任一条联调程序失败：打印失败条目并以非零状态退出（汇总不完整）。
"""
import os, sys, json, subprocess
import numpy as np
import soundfile as sf

DT = [
    "JM5Sga0EXUmCPWbwCCo0RQ_doubletalk",
    "6zWc25PukkSXo5b7et6MaA_doubletalk-with-movement",
    "cQzyYTt2ukK8r3OOuxW5Ag_doubletalk-with-movement",
    "64329f44-6522-4714-b8f4-a563fd0058cb_doubletalk-with-movement",
    "64329f44-6522-4714-b8f4-a563fd0058cb_doubletalk",
    "70ea0c6a-c665-40cf-83fe-cba942ee1858_doubletalk",
    "65114de8-234f-456f-89dc-4997bccb06dc_doubletalk",
    "bd8328b8-ed2d-4aed-9daa-eeda6ed8309a_doubletalk",
    "d3ec0695-254c-4f25-a880-03353ef8756a_doubletalk",
    "d16fb5d4-49fe-4021-b799-b263533ab0e8_doubletalk-with-movement",
    "85a1dbc4-460c-44cd-a2d6-a49158c2881d_doubletalk",
    "62baeabb-7a30-4edf-9205-ea99a6c268b8_doubletalk",
    "L3qjZrpNz0ald0vg2F6P8A_doubletalk",
]


FE = ["00RUTexyKEaW0VCzjL9QYg", "03ra86gHhkWD5ttLpOXWkA", "04281ceb-b127-417e-a2f8-50ccd1a958be",
      "043c0213-3af9-4ccb-ad28-68df92069627", "051aec4b-cdfb-48b7-8c9e-0325262b920b", "0DDAYMovP0eS1P4rWEh0QA"]
NE = ["00d2fda6-46a5-4380-933d-335f4e00bd86", "00t5hG031EmZTpJtcnhS0g", "0132ca07-7891-461a-9944-c78f0208e1f0",
      "02a565aa-ca43-4023-8362-0e38d9f45202", "02f2lbI4f0Kj67Vu4oqmJg", "03ra86gHhkWD5ttLpOXWkA"]


def main():
    exe, bs, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    items = [("双讲", f"{bs}/doubletalk/{n}_mic.wav") for n in DT]
    items += [("远端单讲", f"{bs}/farend-singletalk/{n}_farend-singletalk_mic.wav") for n in FE]
    items += [("近端单讲", f"{bs}/nearend-singletalk/{n}_nearend-singletalk_mic.wav") for n in NE]
    missing = [p for _, m in items for p in (m, m.replace("_mic.wav", "_lpb.wav")) if not os.path.isfile(p)]
    if missing:
        print(f"素材不全，缺 {len(missing)} 个文件：", *missing[:10], sep="\n  "); sys.exit(2)
    rows, fails = [], []
    for kind, mic in items:
        name = os.path.basename(mic)[:-len("_mic.wav")]; d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        for src, dst in ((mic, "mic.pcm"), (mic.replace("_mic.wav", "_lpb.wav"), "ref.pcm")):
            x, sr = sf.read(src, dtype="int16"); assert sr == 48000, (src, sr)
            (x if x.ndim == 1 else x[:, 0]).tofile(os.path.join(d, dst))
        r = subprocess.run([exe, f"{d}/mic.pcm", f"{d}/ref.pcm", "48000", d], capture_output=True, text=True,
                           env=dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(os.path.abspath(exe))))
        if r.returncode != 0:
            print("失败", name, r.stderr[-300:], flush=True); fails.append(name); continue
        j = json.loads(r.stdout.strip().splitlines()[-1]); j.update(kind=kind, name=name); rows.append(j)
        print(kind, name, flush=True)
    with open(os.path.join(out, "joint.jsonl"), "w") as f:
        for j in rows:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    summarize(rows)
    if fails or len(rows) != len(items):   # 有条失败：汇总不完整，退出码非零（10-13 Codex 审查）
        print(f"汇总不完整：{len(items)} 条里失败 {len(fails)} 条：{fails}", flush=True); sys.exit(1)


def summarize(r):
    S = lambda k: sum(x[k] for x in r)
    print(f"条数 {len(r)}，输出全部有限 {all(x['finite'] for x in r)}")
    print(f"切换：执行 {S('switches_applied_F')}（强度）/ {S('switches_applied_G')}（挂钩），已分析 {S('switches_analyzed')}")
    print(f"稳态浮点位比较 {S('steady_cmp')} 个样本，不相等：强度切换 {S('steady_neq_F')}，挂钩关段 {S('steady_neq_G_off')}")
    print(f"强度切换帧级：开 {S('frames_F_on')} + 关 {S('frames_F_off')} 帧，例外 {S('frames_F_bad')}")
    print(f"挂钩回调序列：回调 {S('frames_G')} 次（应 {S('frames_G_expect')}），缺 {S('G_missing')}、多 {S('G_extra')}、重复 {S('G_dup')}，进来的频谱与一直挂着同一块不等 {S('G_in_neq_B')}")
    print(f"挂钩重开 {S('reopen')} 次：收敛 {S('reopen_converged')}、未收敛 {S('reopen_unconverged')}、证据不足 {S('reopen_thin')}、观察不足 {S('reopen_short')}、近静音 {S('reopen_silent')}；收敛最长 {max(x['settle_max_s'] for x in r):.2f} s")
    W = ["0–0.1", "0.1–0.3", "0.3–0.6", "0.6–1.0", "1.0–1.5"]
    for w in range(5):
        v = np.array([row[w] for x in r for row in x["reopen_win_db"] if row[w] is not None])
        if len(v):
            print(f"  重开后 {W[w]} s：误差中位 {np.median(v):.1f} dB，90% 分位 {np.percentile(v, 90):.1f} dB，最大 {v.max():.1f} dB（{len(v)} 个重开）")
    fe = [x for x in r if x["kind"] == "远端单讲"]
    print("远端单讲 频谱档 γ1 − aec48 单独（dB）:", [round(x["B_db"] - x["A_db"], 2) for x in fe])
    print(f"电平：γ1 中位 {np.median([x['B_db'] - x['A_db'] for x in r]):.2f} dB，γ2 中位 {np.median([x['C_db'] - x['A_db'] for x in r]):.2f} dB；"
          f"旧接法与频谱档 γ1 相差 ≤ 0.2 dB 的 {sum(abs(x['D_db'] - x['B_db']) <= 0.2 for x in r)} 条，最大差 {max(abs(x['D_db'] - x['B_db']) for x in r):.2f} dB")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        summarize([json.loads(l) for l in open(sys.argv[1])])
    else:
        main()
