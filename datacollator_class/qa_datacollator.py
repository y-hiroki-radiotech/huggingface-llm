import torch
from dataclasses import dataclass
from typing import Dict, List
from transformers import PreTrainedTokenizer

@dataclass
class QADataCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int = 512
    max_question_length: int = 128
    max_answer_length: int = 384

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        questions = [f['question'] for f in features]
        answers = [f['answer'] for f in features]

        # 質問と回答を別々にトークン化
        question_encodings = self.tokenizer(
            questions,
            truncation=True,
            max_length=self.max_question_length,
            padding='max_length'
        )
        answer_encodings = self.tokenizer(
            answers,
            truncation=True,
            max_length=self.max_answer_length,
            padding='max_length'
        )

        # 質問と回答を結合
        input_ids = [q + a for q, a in zip(question_encodings['input_ids'], answer_encodings['input_ids'])]
        attention_mask = [q + a for q, a in zip(question_encodings['attention_mask'], answer_encodings['attention_mask'])]
        
        # トークンタイプIDの生成
        token_type_ids = [[0] * len(q) + [1] * len(a) for q, a in zip(question_encodings['input_ids'], answer_encodings['input_ids'])]

        # PyTorchテンソルに変換
        batch = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids)
        }

        return batch

if __name__ == "__main__":
    # 使用例
    tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
    collator = QADataCollator(tokenizer)

    # データフレームからサンプルを抽出したと仮定
    samples = [
        {'question': 'What is the capital of France?', 'answer': 'The capital of France is Paris.'},
        {'question': 'Who wrote "Romeo and Juliet"?', 'answer': 'William Shakespeare wrote "Romeo and Juliet".'}
    ]

    batch = collator(samples)
    print(batch)