# DoRAはpeft_configでuse_dora=Trueとする
# モデルの量子化設定はしてはいけない

# Mistralの場合、Llamaだとtarget_moduleは異なるかも
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        use_dora=True, # ここがDoraの設定部分
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)