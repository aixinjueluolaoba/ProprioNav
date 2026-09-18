# assets/ — 内置策略模型

`libncnn_rust.so` 把这两个文件用 `include_bytes!` 编进库里（见 `src/lib.rs` 的
`EMBEDDED_POLICY_PARAM` / `EMBEDDED_POLICY_BIN`）。调用方给的 `param_path` / `bin_path`
读不到文件时（没给、路径不存在、文件坏了），自动回退到这份内置模型，走 ncnn 的内存加载
`ncnn_net_load_param_memory` / `ncnn_net_load_model_memory`，不碰文件系统。
文件存在时依旧优先用文件，所以 `nav_init` 的签名与调用方行为都不变。

| 文件 | 大小 | md5 |
|---|---|---|
| `policy_v5_macro_fast.param` | 2028 | `781c584499d47ea329c5d4122c684ebe` |
| `policy_v5_macro_fast.bin` | 192476 | `471973ed9d73f285ec3ea5f1de3496d5` |

来源：`pipeline_out/policy_v5_macro_fast.{param,bin}`（V5 macro-fast 导出产物）。

换模型时：替换这两个文件 → 重编 `libncnn_rust.so`
（`bash ~/寻路/scripts/build_nav_arm64.sh`）→ 重新打包 APK。
LSTM hidden 维仍从 param 文本自动识别（`MemoryData ... 0=4*hidden`），内置模型同样适用。
