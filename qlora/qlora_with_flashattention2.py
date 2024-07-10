# import the following for QLoRA fine-tuning
# A100GPUを使うこと、T4では使用できない
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, preprare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


# モデルとトークナイザの設定
# add_eos_tokenはトークナイザが入力テキストの最後にEOSトークンを自動的に追加する
model_name = "使用するモデル"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token # padding tokenをunknown tokenを設定
tokenizer.padding_side = "left" # FlashAttentionではleftを使う

# QLoRAの設定.4bit or 8bitを使う
compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsandBytesConfig(
	load_in_4bit=True, # bnb_in_8bit=True
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=compute_dtype,
	bnb_4bit_use_double_quant=True
)

# QLoRAを設定してモデルを読み込む
model = AutoModelForCausalLM.from_pretrained(
	model_name,
	quantization_config=bnb_config,
	device_map={"": 0},
	use_flash_attention_2=True, # ここがFlashAttention2の設定部分
)

# 4bitまたは8bit量子化モデルをファインチューニングするための準備
model = prepare_model_for_kbit_training(model)

# LoRAの設定
peft_config = LoraConfig(
	lora_alpha=16,
	lora_droput=0.1,
	r=16,
	bias='none',
	task_type="CAUSAL_LM",
	target_modules=["q_proj", "v_proj"]
)

# SFTのパラメータ設定
training_arguments = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=16, # 大きくすると勾配消失や爆発の原因になる
        per_device_eval_batch_size=16,
        log_level="debug",
        save_steps=20,
        logging_steps=20,
        learning_rate=4e-4,
        eval_steps=20,
        fp16=True, # bf16もあるA100を使うときに設定
        max_steps=100,
        warmup_steps=10, # warmup_ratio=0.03
        lr_scheduler_type="linear", # cosineもある
        # gradient_accumulation_steps=1,
        # max_grad_norm=0.3,
        # weight_decay=0.001,
        # optim="paged_adamw_32bit", # 途中計算を32bitで保存しておいて計算だったかな
        # group_by_length=True, #シーケンス長が同じサンプルをバッチ処理することでメモリ使用量を削減
)

# trainerの設定
trainer = SFTTrainer(
	model=model,
	train_dataset=dataset["train"],
	eval_dataset=dataset["test"],
	peft_config=peft_config,
	dataset_text_field="text",
	max_seq_length = 1024, # Noneで最大の長さを指定する
	tokenizer=tokenizer,
	args=training_arguments,
	packing=True
)

trainer.train()