# 历史四轮数据来源（不能直接跑十轮）

以下原始说明仅用于追溯。143360-token 序列不足以运行至 200K；十轮数据要求见 data.md。

# 数据来源、格式与复现

## 直接引用现有数据

Mac 实验目录：
`/Users/wangchangran/remote/pd-prefix-agent-100k-140k-20261029`

远端共享实验目录：
`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/tokenspeed_master_6000d/local_work/pd-prefix-agent-100k-140k-20261029`

目录内：
- `sessions.json`：32 条、每条 143360 token 的完整序列。
- `manifest.json`：长度、来源、历史接受率、重复补足次数、各轮前缀 hash。
- `source-ranking.json`：旧的 128K 单请求筛选指标，不含旧版 token IDs。
- `validation.json`：构造校验结果。
- `prepare.py`：数据构造实现，路径依赖原实验目录布局。

本 skill 不复制这些文件。找不到原文件时先检查远端和本地同步目录，不自行造数据。用户授权迁移时再拷贝至新的实验目录并验证 hash。

`sessions.json` 字节 SHA256：
`cecee5f8f80c16e88b0ef225c0c6adf5e9991286a232edf7dcdb942d0c09bdcf`

每条保存 `rank/id/sub_domain/original_context_sha256/original_context_tokens/input_ids/input_lengths/output_tokens/prefix_sha256`，以及 `historical_average_accept_length/historical_draft_acceptance_rate_pct/new_acceptance_measured/context_repeats`。`new_acceptance_measured=false` 表示打包时尚未对新构造完成筛选，不是实时实验状态。

## 来源与构造

原始数据：
`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/LongBench/data.json`

旧版 64 选 32 的高接受率筛选文件（Mac）：
`/Users/wangchangran/remote/h20-dspark-128k-highaccept32-20261028/128k_high_accept32.json`

旧筛选每条为 131072 输入、1024 输出，取原始格式化提示前 65536+后 65536 token。用户允许重复输出，没有以答案质量过滤；不是无偏 LongBench 准确率评测。新数据沿用这 32 个不同来源和排序，**不沿用旧版截断后的 IDs**。

当前构造使用本机模型的 AutoTokenizer：
`/tmp/wcr-models/DeepSeek-V4.1-Flash-1017`，`trust_remote_code=True, local_files_only=True`，`add_special_tokens=False`。

先编码固定开头，再编码该来源完整 `context`，拼接后取前 143360 tokens。固定开头为：

```text
Benchmark session: {id}
Read the following progressively supplied document. Continue explaining the information relevant to this question using the available text.
{question}
Document:
```

原文不足时追加 `\nDocument repeated for benchmark length:\n` 加原文的 token 序列，直到够长。本批 4 条重复补足、28 条无需重复，manifest 明确标注。没有每轮重置问题或重新套聊天模板，旧前缀始终逐 token 一致。

这是 raw-token 前缀缓存性能负载。提示不使用模型完整 chat 模板，新增 chunk 可能截在文档中间；不得把它表述为自然完整 agent 对话。历史高接受率不保证新提示或各轮仍高接受率。

## 消费方式

```python
request_input_ids = session['input_ids'][:session['input_lengths'][turn]]
# turn 0..4; [102400,112640,122880,133120,143360]
```

各并发用排序前 C 条，分别保留 `source_id`，不带入任何模型生成输出。不同并发使用不同大小的样本子集，不能把组均值变化直接归因于并发。只要用户没有要求，不新增同样本对照实验。

需要重建时先复核原始 context SHA256，并保存 tokenizer 文件版本/hash；使用上述明确规则。可参考原实验的 prepare.py，但它的路径布局固定，迁移前须调整参数。
