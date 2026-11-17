# TokenSpeed 网页参考压测

独立归档的 Codex skill、可迁移压测脚本和参考数据。入口为 [SKILL.md](SKILL.md)，执行口径与命令见 [protocol.md](references/protocol.md)。

## 文件

- `scripts/benchmark_grpc.py`：参数化的原生 gRPC 压测脚本。
- `original/web-fine-bench.py`：保留原始脚本，供追溯比较。
- `data/input-provenance.json`：64 条输入的长度和校验值；不是完整 token 数据。通过匹配的 tokenizer 重建并校验，脚本将导出完整输入。
- `data/web-performance-data.json`、`data/comparison.csv`：历史网页与对比数据。
- [6000D PD 配置简表](docs/6000D_PD_CONFIG_SUMMARY.md)：历史部署配置、实现库及验证边界；配套参数与实验总结也保存在 `docs/`。

## 使用

按 SKILL.md 和协议文档准备运行环境、模型 tokenizer 与已启动的服务，再执行脚本。原生 gRPC/protobuf 接口必须匹配；不能直接将 SMG HTTP 地址作为 gRPC 地址使用。

如需安装为 Codex skill，可将本仓库复制到 `~/.codex/skills/tokenspeed-web-reference-bench`；已有同名 skill 时先比较修改，勿直接覆盖。

## 两种实验不要混用

本 skill 复现共享前缀合成输入的**单批次网页参考测试**：128,928 token 输入、2,048 token 输出、真实 DSpark 接受率、三轮正式测量。

`docs/` 中 PD 配置来自 **2026-11-10 的递增前缀十轮实验**，用于部署参考，不代表网页测试采用相同数据、固定接受率或路由协议。部署文档中的源码相对路径指 TokenSpeed 仓库；历史日志、运行目录及图片引用是证据位置，不是本仓库的运行依赖。本仓库不包含服务源码、模型权重或完整历史运行产物。

校验归档文件：`shasum -a 256 -c SHA256SUMS`。整理时只做离线脚本与文件校验，未重新运行 GPU 压测。
