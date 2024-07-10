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