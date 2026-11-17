# 参考协议与使用

来源（历史快照，未重新查询在线页面）：
http://33.29.36.157:8080/sm120_w4a8_online/comparison_20261103/SM120-W4A8-Performance-Report.html#fine-curves

P QCP8/EP8，D两机16卡DP16/EP16；NVFP4 dispatch / FP8 combine；真实DSpark接受率。64条合成输入共享128800 token前缀，各追加128 token；输入128928、输出2048，temperature=0、ignore_eos。随机种子20261028，包含tokenizer decode/re-tokenize，因此同种子不保证不同tokenizer版本一致，必须核对哈希。并发≤64取前N条，>64循环复用。

网页每rank并发1/2/4/6/8/12/16，即总C16/32/64/96/128/192/256；之前对比图只选1/2/4/8/16。数据 comparison.csv 同时含我们的历史结果，series=网页参考才是网页成绩；原始完整数据在web-performance-data.json。

## 依赖和接口

脚本需要匹配服务版本的 grpcio、grpcio-health-checking、protobuf、smg_grpc_proto、transformers，以及模型tokenizer。复用匹配的现有虚拟环境；没有给依赖随意指定新版本。接口必须提供Health、GetServerInfo、GetLoads、Generate及配对PD字段，server_args含deepep_dispatch_dtype/combine_dtype等。decode-log指向采样的D日志，需从运行脚本的机器可读；日志接受长度取算术平均，不是token加权draft接受率。

## 运行

以包根目录为工作目录，TOKENIZER/P_ADDR/D_ADDR/D_LOG由部署环境提供：

```bash
for b in 1 2 4 8 16; do
  python scripts/benchmark_grpc.py --tokenizer "$TOKENIZER" \
    --prefill-address "$P_ADDR" --decode-address "$D_ADDR" --decode-log "$D_LOG" \
    --per-rank "$b" --repeats 1 --output-tokens 2048 --timeout 1800 \
    --output-dir "results/b${b}-warmup" || exit 1
  python scripts/benchmark_grpc.py --tokenizer "$TOKENIZER" \
    --prefill-address "$P_ADDR" --decode-address "$D_ADDR" --decode-log "$D_LOG" \
    --per-rank "$b" --repeats 3 --output-tokens 2048 --timeout 1800 \
    --output-dir "results/b${b}-formal" || exit 1
done
```

先创建results父目录。勿在预热和正式轮之间清KV；必须记录测试开始时的cache状态。以上每档额外预热是我们的复现实验操作，原始网页脚本只有repeats循环，不能声称原脚本自动预热。服务必须已经就绪；本包不会启动或重启服务。

脚本每次生成并验证完整输入，成功后保存input_ids.json。当前包只包含生成方法及原始哈希，不包含预先恢复的完整输入，也未对当前在线服务做运行验收。

## 统计

每请求TPOT=(last-first)/(N-1)，每轮求P50；三轮P50算术平均。单请求TPS=1000/该均值(ms)。每轮输出吞吐=sum(N)/(max(last)-min(first))，再平均三轮；包含爬升和尾部、不含首次输出前时间，不等于C×TPS。单D卡吞吐=总吞吐/16，不计P卡。首个流式chunk可含多个token，沿用原脚本N-1定义。

脚本记录requests.json、loads.json、capture.json、results.jsonl及日志窗口；检查每轮ok=C、无errors、每请求2048输出、all_streams_completed。verified_target_window反映采样的GetLoads窗口，不等于GPU同步实际满BS。
