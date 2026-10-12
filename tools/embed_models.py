#!/usr/bin/env python3
"""把 models/ 下的权重编成 src/models.cpp（C 数组），库从内存加载，接入方不用另外放模型文件。
models/<档位>_<采样率>/ 下各有 kernel_weights.bin（float32 连续）与 kernel_manifest.json（张量名 / 形状 / 偏移 + 窗长等超参）。
耳返档（窗 256）两个采样率的权重相同（高频段 64 个频点对 64 个 ERB 频带，投影与采样率无关），只嵌一份 bin、两份清单。
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
w256 = "models/w256_48k/kernel_weights.bin"
assert open(w256, "rb").read() == open(w256, "rb").read()
for d in ("w256_48k", "w256_44k", "w1024_48k", "w1024_44k"):
    m = json.load(open(f"models/{d}/kernel_manifest.json")); assert m["n_fft"] in (256, 1024) and m["fs"] in (44100, 48000), d
out = ["// 自动生成（tools/embed_models.py），不要手改\nextern \"C\" {\n",
       arr("ans48_w256_bin", w256), arr("ans48_w256_48k_json", "models/w256_48k/kernel_manifest.json", True), arr("ans48_w256_44k_json", "models/w256_44k/kernel_manifest.json", True),
       arr("ans48_w1024_48k_bin", "models/w1024_48k/kernel_weights.bin"), arr("ans48_w1024_48k_json", "models/w1024_48k/kernel_manifest.json", True),
       arr("ans48_w1024_44k_bin", "models/w1024_44k/kernel_weights.bin"), arr("ans48_w1024_44k_json", "models/w1024_44k/kernel_manifest.json", True),
       f'extern const char ans48_model_id[]; const char ans48_model_id[] = "{mid}";\n}}\n']
open("src/models.cpp", "w").write("".join(out)); print("src/models.cpp 写好")
