# wandb install
!pip install wandb -qq
import wandb
from google.colab import userdata

WANDB_KEY = userdata.get('WANDB_API_KEY')
wandb.login(key=WANDB_KEY)

# training argumentsの作成
from transformers import TrainingArguments

batch_size = 32
training_dir = "train_dir"

training_args = TrainingArguments(
    output_dir=f"training_dir_{model_ckpt}",
    overwrite_output_dir=True,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    report_to="wandb", # wandbを使う場合の設定はここの2行が必須
    logging_steps=1,
    do_train=True,
    do_eval=True,
    fp16=True
)

# callbacksの設定
from transformers.integrations import WandbCallback
# use sklearn to build compute metrics
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1}

class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.epoch % self.freq == 0:
            predictions = self.trainer.predict(self.sample_dataset)
            predictions = compute_metrics(predictions)
            predictions_df = pd.DataFrame([predictions])
            # add predictions to a wandb.Table
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"distilbert_sample_predictions": records_table})

# trainerの設定とcallbackの設定
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics_evaluate,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=distilbert_tokenizer
)

progress_callback = WandbPredictionProgressCallback(trainer, distilbert_tokenizer, encoded_dataset["validation"])
trainer.add_callback(progress_callback)

# wandb.init()の設定
# TrainingArguments から重要な設定を抽出
config = {
    "num_train_epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "per_device_train_batch_size": training_args.per_device_train_batch_size,
    "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
    "weight_decay": training_args.weight_decay,
    "fp16": training_args.fp16,
    "output_dir": training_args.output_dir,
    "evaluation_strategy": training_args.evaluation_strategy,
    "run_name": training_args.run_name
}

# wandb.init() を呼び出し、config を設定
try:
    wandb.init(
        project="fake-news-detection",
    name="tinybert",
    tags=["baseline", "tinybert"],
    group="tinybert",
    config=config)
    trainer.train()
finally:
    wandb.finish()