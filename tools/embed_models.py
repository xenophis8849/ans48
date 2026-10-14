#!/usr/bin/env python3
"""把 models/ 下的权重编成 src/models.cpp（C 数组），库从内存加载，接入方不用另外放模型文件。
models/<档位>_<采样率>/ 下各有 kernel_weights.bin（float32 连续）与 kernel_manifest.json（张量名 / 形状 / 偏移 + 窗长等超参）。
频谱档（窗 1536，挂在 aec48 后）同推流档，两个采样率各一份（ERB 投影随采样率变，在 bin 里）。
耳返档（窗 256）0.4.0 起同推流档：小核瘦身模型的 ERB 是 33 + 32（原模型 65 + 64 时高频段频点与频带一一对应、投影与采样率无关，两个采样率可共用一份 bin；
现在投影随采样率变），所以也合成一个 bin：网络张量一份 + 两个采样率各自的 ERB 张量。
用法：tools/embed_models.py <模型标识>"""
import sys, hashlib, json
mid = sys.argv[1]
def arr(name, path, text=False):
    b = open(path, "rb").read(); h = hashlib.sha256(b).hexdigest()[:16]
    if text: b = b + b"\0"
    rows = ",\n".join(",".join(str(x) for x in b[i:i + 32]) for i in range(0, len(b), 32))
    n = len(b) - (1 if text else 0)
    return (f"// {path}  {n} 字节  sha256 前 16 位 {h}\nextern const unsigned char {name}[]; alignas(16) const unsigned char {name}[] = {{\n{rows}\n}};\n"
            f"extern const unsigned long {name}_size; const unsigned long {name}_size = {n};\n")
for d in ("w256_48k", "w256_44k", "w1024_48k", "w1024_44k", "w1536_48k", "w1536_44k"):
    m = json.load(open(f"models/{d}/kernel_manifest.json")); assert m["n_fft"] in (256, 1024, 1536) and m["fs"] in (44100, 48000), d
for d in ("w1536_48k", "w1536_44k"):   # 频谱档（挂在 aec48 后）：按联动契约分帧，窗型必须是 sqrt-Hann、帧移 384
    m = json.load(open(f"models/{d}/kernel_manifest.json")); assert m["hop"] == 384 and m.get("win") == "sqrt_hann", d
# 三个档：两个采样率的权重只有 ERB 投影两个张量不同（双采样率训的同一份网络；10-13 逐字节核过），
# 所以合成一个 bin：网络张量一份 + 两个采样率各自的 ERB 张量，两份清单各指各的偏移。库的加载代码不变（清单里的偏移即接口）。
# 原来两个采样率各存一整份，0.3.0 加频谱档后库从 2.8 MB 涨到 4.8 MB，大半是重复的网络权重。
import os, tempfile
def merge(tier):
    import numpy as np
    ma = json.load(open(f"models/{tier}_48k/kernel_manifest.json")); mb = json.load(open(f"models/{tier}_44k/kernel_manifest.json"))
    da = np.fromfile(f"models/{tier}_48k/kernel_weights.bin", np.float32); db = np.fromfile(f"models/{tier}_44k/kernel_weights.bin", np.float32)
    assert [t["name"] for t in ma["tensors"]] == [t["name"] for t in mb["tensors"]], tier
    buf = []; off = 0; outa = []; outb = []
    def put(arr):
        nonlocal off; o = off; buf.append(arr); off += arr.size; return o
    for ta, tb in zip(ma["tensors"], mb["tensors"]):
        xa = da[ta["offset"]:ta["offset"] + ta["count"]]; xb = db[tb["offset"]:tb["offset"] + tb["count"]]
        if ta["name"].startswith("erb."):
            outa.append(dict(ta, offset=put(xa))); outb.append(dict(tb, offset=put(xb)))
        else:
            assert np.array_equal(xa, xb), f"{tier} {ta['name']} 两个采样率不同，不能合并"
            o = put(xa); outa.append(dict(ta, offset=o)); outb.append(dict(tb, offset=o))
    d = tempfile.mkdtemp(); allf = np.concatenate(buf).astype(np.float32)
    allf.tofile(os.path.join(d, "w.bin"))
    for tag, m, ts in (("48k", ma, outa), ("44k", mb, outb)):
        json.dump(dict(m, tensors=ts, floats=int(allf.size)), open(os.path.join(d, f"{tag}.json"), "w"), ensure_ascii=False)
    print(f"{tier}：合并后 {allf.size * 4 / 1024:.0f} KB（原两份共 {(da.size + db.size) * 4 / 1024:.0f} KB）")
    return d
d256, d1024, d1536 = merge("w256"), merge("w1024"), merge("w1536")
out = ["// 自动生成（tools/embed_models.py），不要手改\nextern \"C\" {\n",
       arr("ans48_w256_bin", f"{d256}/w.bin"), arr("ans48_w256_48k_json", f"{d256}/48k.json", True), arr("ans48_w256_44k_json", f"{d256}/44k.json", True),
       arr("ans48_w1024_bin", f"{d1024}/w.bin"), arr("ans48_w1024_48k_json", f"{d1024}/48k.json", True), arr("ans48_w1024_44k_json", f"{d1024}/44k.json", True),
       arr("ans48_w1536_bin", f"{d1536}/w.bin"), arr("ans48_w1536_48k_json", f"{d1536}/48k.json", True), arr("ans48_w1536_44k_json", f"{d1536}/44k.json", True),
       f'extern const char ans48_model_id[]; const char ans48_model_id[] = "{mid}";\n}}\n']
open("src/models.cpp", "w").write("".join(out)); print("src/models.cpp 写好")
