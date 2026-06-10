# M4.4 G0-G3 四组生成对比评估

- 评测子集: 30 条 (stratified from `data\eval\gold_130.jsonl`)
- 基座: `Qwen/Qwen2.5-0.5B-Instruct` (4-bit NF4)
- LoRA: `artifacts\models\qwen_lora_csrc` (M4.3 产出)

## 四组总表

| 组 | 配置 | EID 命中率 | 格式合规 | 幻觉数字率 | 答案长度 | 延迟/query |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | base + 无 RAG + 弱 prompt | 0.000 | 0.000 | 0.333 | 167 字 | 11.04s |
| G1 | base + RAG + 弱 prompt | 0.000 | 0.000 | 0.033 | 215 字 | 15.16s |
| G2 | base + RAG + 强 prompt | 0.000 | 0.000 | 0.067 | 149 字 | 11.70s |
| G3 | base + **LoRA** + RAG + 强 prompt | 0.200 | 0.767 | 0.067 | 174 字 | 20.33s |
