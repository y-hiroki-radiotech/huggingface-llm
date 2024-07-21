from transformers import AutoTokenizer
import torch
from typing import Dict, List, Union, Callable, Optional
from transformers import AutoTokenizer
from datasets import Dataset


# 必要なモデルに応じてformatを追加していく
def create_prompt_format(model_type: str) -> Callable[[str], str]:
    """モデルタイプに基づいてプロンプト形式を生成する関数を返す"""
    if model_type.lower() == "llama2":
        return lambda q: f"[INST] <<SYS>>\nYou are a helpful assistant. Answer the following question.\n<</SYS>>\n\nQuestion: {q}\nAnswer: [/INST]"
    elif model_type.lower() == "bert":
        return lambda q: f"[CLS] Question: {q} [SEP] Answer:"
    else:
        return lambda q: q


def preprocess(examples, prompt_format=None):
	# 質問と回答のペアを結合
	questions = examples["question"]
	answers = examples["answer"]

	# プロンプト形式の適用
    if prompt_format is None:
        print("Promtp_formatを設定してください")
    prompts = [prompt_format(q) for q in questions]

    # tokenize
    # ここは必要に応じて修正する
    tokenized_questions = tokenizer(
        prompts,
        truncation=False,
        max_length=MAX_LENGTH,
        padding="max_length"
        )

    tokenized_answers = tokenizer(
        answers,
        truncation=False,
        max_length=MAX_LENGTH
        )

    # 各特徴量の初期化
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    tokenized_examples["labels"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        
        # オーバーフローしたトークンを処理
        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        
        # 特殊トークンの位置を見つける（モデルタイプに応じて調整）
        if model_type.lower() == "llama2":
            answer_start = input_ids.index(tokenizer.encode("[/INST]")[-1]) + 1
        elif model_type.lower() == "bert":
            answer_start = input_ids.index(tokenizer.sep_token_id) + 1
        else:
            answer_start = 0
        
        # ラベルの作成
        labels = [-100] * len(input_ids)
        answer_ids = tokenized_answers["input_ids"][sample_index]
        labels[answer_start:answer_start + len(answer_ids)] = answer_ids
        
        tokenized_examples["labels"].append(labels)
    
    return tokenized_examples


def process_dataset(dataset: Dataset, model_name: str, model_type: str) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, model_type),
        batched=True,
        remove_columns=dataset.column_names,
    )


if __name__ == "__main__":
    # Llama 2の場合
    processed_dataset_llama = process_dataset(dataset, "meta-llama/Llama-2-7b-hf", "llama2")

    # BERTの場合
    processed_dataset_bert = process_dataset(dataset, "bert-base-uncased", "bert")

    print(processed_dataset_llama[0])
    print(processed_dataset_bert[0])