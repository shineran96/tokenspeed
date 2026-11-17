# 十轮数据

采用已有 sessions128.json，文件 SHA256：
`6ae10b8740c882349cf99fe11c59ca83b966e3c387adeaade248ae9ff3c8bd7f`。

历史共享目录定位（仅为数据引用，不是脚本默认值）：
`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/tokenspeed_6000d_decode_optimized/deploy/dsv41_pd/.runtime/smg-agent10-c64-c128-20261106/sessions128.json`。

运行时通过 SESSIONS_FILE 指定可访问副本。先校验文件哈希及结构：顶层 list，每条包含 id、input_ids；每条至少 204800 tokens，目标并发不超过条数。请求取前 C 条，各轮取前 102400 + turn × 10240 个 token，turn=0..10。

原始来源为 LongBench，沿用历史高接受率候选；绑定 DeepSeek-V4.1-Flash-1017 tokenizer。历史接受率不是本次实际接受率。来源、扩展/重复规则须检查数据 manifest；不能仅凭文件名断言 128 条来自 128 个不同文档。缺少文件时先找归档副本，不自行生成后沿用原哈希。

[早期数据说明](legacy-data.md) 记录了 32 条、每条 143360 token 的四轮数据及来源，仅供追溯。它不足以支持十轮；重建更长数据时保存 tokenizer 版本、原文哈希、构造规则和新输入哈希，明确不能宣称与历史逐 token 相同。
