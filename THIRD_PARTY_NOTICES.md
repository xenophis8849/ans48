# 第三方声明

## pffft

`third_party/pffft/`，静态链接进 `libans48.so`。来源：<https://github.com/marton78/pffft>。

本库对 pffft 的修改：`src/pffft_priv_impl.h` 的 setup 创建函数检查两处内存分配，失败时返回 NULL（原实现直接解引用空指针）。

**分发包含 `libans48.so` 的软件（例如 App）时，须在随附文档或其他材料中附上以下版权声明、条款与免责声明**（pffft 许可对二进制分发的要求）。

```

Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )
Copyright (c) 2019  Hayati Ayguen ( h_ayguen@web.de )
Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

Copyright (c) 2004 the University Corporation for Atmospheric
Research ("UCAR"). All rights reserved. Developed by NCAR's
Computational and Information Systems Laboratory, UCAR,
www.cisl.ucar.edu.

Redistribution and use of the Software in source and binary forms,
with or without modification, is permitted provided that the
following conditions are met:

- Neither the names of NCAR's Computational and Information Systems
Laboratory, the University Corporation for Atmospheric Research,
nor the names of its sponsors or contributors may be used to
endorse or promote products derived from this Software without
specific prior written permission.  

- Redistributions of source code must retain the above copyright
notices, this list of conditions, and the disclaimer below.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions, and the disclaimer below in the
documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.

```

## UL-UNAS

本库降噪网络的结构（编码 / 解码卷积块、DPGRNN、时频注意力）与 ERB 滤波器组的计算方法取自 UL-UNAS，来源：<https://github.com/Xiaobin-Rong/ul-unas>。本库的推理代码为独立实现，模型权重为自行训练。

**分发包含 `libans48.so` 的软件（例如 App）时，须在随附文档或其他材料中附上以下版权声明与许可文本**（MIT 许可的要求）。

```
MIT License

Copyright (c) 2026 Xiaobin Rong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 测试向量

`test/vec/` 下的输入录音取自 Microsoft ICASSP 2023 Acoustic Echo Cancellation Challenge 盲测集（近端单讲的麦克风录音，前 10 s；44.1 kHz 版由 48 kHz 版多相重采样得到），来源：<https://github.com/microsoft/AEC-Challenge>。按 Microsoft 提供该数据集时的条款使用（"as is"，见上述仓库 README 的 Dataset licenses 一节），仅用于本库自检，不随库分发。其余 `*.pcm` 为参照实现处理上述录音的输出。
