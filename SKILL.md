---
name: tokenspeed-web-reference-bench
description: 复现已归档 SM120 网页参考的单批次 PD 吞吐测试，使用共享前缀合成输入、DP16/EP16、真实接受率，汇总三轮 TPOT P50 和单 D 卡吞吐。不用于 100K prime 后十轮 agent 前缀测试。
---

# 网页参考 PD 压测

先读 [协议与运行方式](references/protocol.md)，确认用户要的是该协议而非十轮 agent 测试。

- 数据、脚本、参考结果都在本目录。`data/input-provenance.json` 仅含哈希和长度，不是完整 token IDs；使用脚本配合同版 tokenizer 重建，并核对全部64条哈希。哈希不符停止，不更新参考哈希来掩盖差异。
- `original/` 保留原始脚本；`scripts/benchmark_grpc.py` 仅移除固定 tokenizer/provenance 路径并导出已验证的 input_ids，不改计算公式。
- 运行前检查网页版本原生 gRPC/protobuf 服务契约。当前 TS headless ZMQ/SMG 不是该接口；不得通过删掉配置校验来宣称已复现。若使用 HTTP 适配器，单独标记入口差异。
- 输入128928、输出2048；真实 DSpark 接受率，取消模拟接受长度环境变量。DP16/EP16，固定 rank=index%16；不能用此证明自动负载均衡。
- 每档先完整预热一轮，再正式三轮，保留 prefix cache；预热单独输出目录。原始脚本自身没有自动排除预热。
- 三轮先各求请求TPOT中位数，再平均；吞吐先各轮实测再平均。成功、长度、rank负载窗口与接受率均需核验。负载窗口不等于逐GPU步满BS。
- 失败档保留原文件并停止后续测试，不把部分响应吞吐作为完整结果。该skill不授权自动重启/停止已有服务或安装依赖。
- 归档版本、参数、输入哈希、原始requests/loads/capture；输出目录不覆盖。

## 部署参考

参见 [6000D PD 配置简表](docs/6000D_PD_CONFIG_SUMMARY.md)。该文档记录历史十轮实验部署，执行本 skill 时仍以 references/protocol.md 的网页测试口径为准。
