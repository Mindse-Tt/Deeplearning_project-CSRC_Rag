# M4.4 G0-G3 四组生成对比评估

- 评测子集: 50 条 (stratified from `data\eval\gold_130.jsonl`)
- 基座: `Qwen/Qwen2.5-0.5B-Instruct` (4-bit NF4)
- LoRA: `artifacts\models\qwen_lora_csrc` (M4.3 产出)

## 四组总表

| 组 | 配置 | EID 命中率 | 格式合规 | 幻觉数字率 | 答案长度 | 延迟/query |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | base + 无 RAG + 弱 prompt | 0.000 | 0.000 | 0.180 | 156 字 | 8.92s |
| G1 | base + RAG + 弱 prompt | 0.000 | 0.000 | 0.100 | 239 字 | 15.46s |
| G2 | base + RAG + 强 prompt | 0.000 | 0.000 | 0.080 | 167 字 | 11.18s |
| G3 | base + **LoRA** + RAG + 强 prompt | 0.280 | 0.760 | 0.020 | 166 字 | 17.41s |
