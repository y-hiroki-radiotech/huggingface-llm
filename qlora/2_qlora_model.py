from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


# まずは4bitまたは8bitでモデルをダウンロードする
# 量子化部分
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation=attn_implementation
)

# メモリ効率を高めるための設定
"""    量子化モデルを効率的に訓練するための準備を行う関数
    
    主な処理:
    1. 勾配チェックポイントの有効化（メモリ節約）
    2. 埋め込み層の8ビット量子化
    3. バイアスの32ビット精度への変換
    4. 特定層の勾配計算の有効化
    5. 入出力を32ビット精度に設定
    
    結果: メモリ効率が高く、訓練に適した状態のモデルを返す
"""
model = prepare_model_for_kbit_training(model)

# LoRAの設定
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRAの部分をモデルに適用させる
model = get_peft_model(model, config)
