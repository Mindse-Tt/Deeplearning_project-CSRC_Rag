# M1 — QLoRA 本机可行性 Smoke Test 报告

- **Agent**: 执行阶段 (QLoRA smoke test)
- **日期**: 2026-04-22
- **目标**: 验证 RTX 2060 SUPER 8GB 能否跑 Qwen QLoRA fp16 且显存峰值 < 7GB 不 OOM
- **结论**: 🟢 **本机主训可行**（Qwen-1.5B 4bit QLoRA fp16 峰值仅 2.87 GB）

---

## 1. 环境信息

| 项 | 值 |
|---|---|
| OS | Windows 10/11 |
| Python | 3.12.10 |
| GPU | NVIDIA GeForce RTX 2060 SUPER |
| Compute Capability | 7.5 (Turing，支持 fp16，不支持原生 bf16) |
| 驱动 | 580.88 |
| CUDA runtime | 12.1（通过 torch wheel 携带；宿主驱动 CUDA 13.0 向下兼容） |
| torch | 2.5.1+cu121（原 2.10.0+cpu 已卸载） |
| bitsandbytes | 0.43.3（win_amd64 官方 wheel，开箱即用） |
| transformers | 4.44.2 |
| peft | 0.11.1 |
| accelerate | 0.33.0 |
| trl | 0.9.6 |
| HF_ENDPOINT | `https://hf-mirror.com` |

### 安装过程关键步骤

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "peft==0.11.*" "transformers==4.44.*" "accelerate==0.33.*" "trl==0.9.*"
pip install "bitsandbytes==0.43.*"
```

- **注意 1**：用户原环境里 torch 是 `2.10.0+cpu` 无 CUDA，必须重装。
- **注意 2**：`bitsandbytes 0.43.3` Windows 原生 wheel 直接能用，无需 `bitsandbytes-windows` fallback。
- **注意 3**：`transformers 5.5` 被降级到 `4.44.2`（peft 0.11 与 trl 0.9 兼容线）。

---

## 2. 数据

- 临时构造 50 条极简 QA（脚本：`scripts/smoke_build_qa.py`）
- 输出：`data/train_smoke/smoke_qa.jsonl`
- 格式示例：

```json
{"instruction":"列出一个内幕交易的真实案例","input":"","output":"根据 [EventID=401] 记录：..."}
```

仅用于链路验证，不用于真实训练。

---

## 3. Qwen-0.5B 结果 ✅

脚本：`scripts/smoke_train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --output_dir artifacts/smoke_qwen05_lora --max_steps 20 --batch 1 --grad_accum 4`

| 指标 | 值 |
|---|---|
| 状态 | ok |
| 可训练参数 | 2,162,688（0.44%） |
| 峰值显存 (allocated) | **1.571 GB** |
| 峰值显存 (reserved)  | 2.131 GB |
| 训练耗时 | 96.8 秒 / 20 步 |
| 样本速度 | 0.83 samples/s |

Loss 曲线（5 步打点）：

| Step | Loss |
|---|---|
| 5  | 3.7934 |
| 10 | 3.3656 |
| 15 | 2.9263 |
| 20 | 2.5075 |

✅ Loss 单调下降，反向传播链路健康；LoRA adapter ~8.3 MB 保存到 `artifacts/smoke_qwen05_lora/`。

---

## 4. Qwen-1.5B 结果 ✅

脚本：`scripts/smoke_train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --output_dir artifacts/smoke_qwen15_lora --max_steps 20 --batch 1 --grad_accum 8`

| 指标 | 值 |
|---|---|
| 状态 | ok |
| 可训练参数 | 4,358,144（0.28%） |
| 峰值显存 (allocated) | **2.868 GB** |
| 峰值显存 (reserved)  | 3.559 GB |
| 训练耗时 | 227.4 秒 / 20 步（grad_accum=8 → 160 micro-steps） |
| 样本速度 | 0.70 samples/s |

Loss 曲线：

| Step | Loss |
|---|---|
| 5  | 3.4474 |
| 10 | 3.0053 |
| 15 | 2.6450 |
| 20 | 2.3873 |

✅ Loss 下降良好；LoRA adapter ~17 MB 保存到 `artifacts/smoke_qwen15_lora/`。

### 显存预算解读

- 2.87 GB 是 **我们的 python 进程** `torch.cuda.max_memory_allocated()`，不含桌面/浏览器占用。
- 7 GB 的阈值还有 **~4 GB** 冗余，足以把 `per_device_batch_size` 升到 2～4 或把 `max_seq_length` 升到 1024～2048。
- `nvidia-smi` 在训练时看到 7.48 GB 是因为系统桌面本身占 5.35 GB，训练进程只吃掉 ~2.1 GB 增量（与 `torch.cuda.max_memory_allocated` 一致）。

---

## 5. 关键技术细节（为后续正式训练参考）

| 配置项 | smoke 采用 | 正式训练建议 |
|---|---|---|
| `load_in_4bit` | ✓ | ✓ |
| `bnb_4bit_compute_dtype` | **float16**（不能 bf16，Turing 不支持） | **float16** |
| `bnb_4bit_quant_type` | nf4 | nf4 |
| `bnb_4bit_use_double_quant` | ✓ | ✓ |
| `TrainingArguments.fp16` | `True` | `True`（`bf16=False`） |
| `optim` | `paged_adamw_8bit` | 同上 |
| `gradient_checkpointing` | `True` | 同上 |
| `per_device_train_batch_size` | 1 | 可升 2，视 `max_seq_length` 而定 |
| `gradient_accumulation_steps` | 4 / 8 | 4～8 |
| `target_modules` | `q/k/v/o_proj` | 正式可加 `gate/up/down_proj`（占显存略升） |
| `max_seq_length` | 384 | 先 1024，OOM 再降 |

⚠️ **必须 override `configs/qlora_config.json` 的 `bf16: true` 为 `false` + `fp16: true`**（该配置原为 A100/H100 场景，Turing 不支持 bf16）。smoke 未改该文件，符合"禁止改正式配置"约定，正式训练前由 团队成员 或 执行阶段 显式设置。

---

## 6. 结论

### 🟢 本机主训可行

- Qwen-0.5B smoke：峰值 1.57 GB，通过 ✅
- Qwen-1.5B smoke：峰值 2.87 GB，通过 ✅（距 7 GB 红线尚有 >4 GB 余量）
- Loss 曲线健康，前向/反向/优化器链路无异常
- bitsandbytes Windows wheel 原生可用，无需特殊 fallback

### 推荐方案

1. **正式训练主路径**：本机 Qwen-2.5-1.5B-Instruct 4bit QLoRA fp16，主训数据 5000 条
2. **可用扩展空间**：将 `per_device_train_batch_size` 升到 2、`max_seq_length` 升到 1024、`target_modules` 扩展到 7 个 projection，预计峰值仍在 6 GB 以内
3. **Colab T4 不再是强依赖**：仍可作为加速备份（T4 算力 ~2x 2060 SUPER），但不是可行性前提
4. **性能预期**：1.5B smoke 里 ~0.7 samples/s ≈ 60 samples/min。5000 样本 × 3 epoch = 15000 样本，估算纯训练时间 4～5 小时；加保存/评估 6 小时左右
5. **下一步**：团队成员 或正式 Execution 阶段生成 `data/train/rag_qa_train.jsonl` 真实语料后，即可在 `configs/qlora_config.json` 上改 `bf16→fp16` 启动正式训练

### 风险提示

- 本机 5.35 GB 桌面基线较高（多浏览器/企业微信等）；正式训练时建议关闭非必要 GUI 应用，避免挤占显存
- 单卡 RTX 2060S 训练 1.5B × 5000 条 × 3 epoch 约 5～6 小时，期间最好挂后台（`start /MIN` 或 `nohup` 等价）
- peft 0.11 + transformers 4.44 + trl 0.9 是兼容组合，升级 transformers 5.x 会导致 `prepare_model_for_kbit_training` API 变动，正式训练前锁版本

---

## 7. 产出文件清单

```
artifacts/smoke_qwen05_lora/
├── adapter_config.json
├── adapter_model.safetensors  (~8.3 MB)
├── tokenizer.* / vocab.*
└── metrics.json

artifacts/smoke_qwen15_lora/
├── adapter_config.json
├── adapter_model.safetensors  (~17 MB)
├── tokenizer.* / vocab.*
└── metrics.json

data/train_smoke/smoke_qa.jsonl  (50 条，仅 smoke 用)

scripts/smoke_build_qa.py
scripts/smoke_train_qlora.py
```

均未 commit，遵循任务约束。
