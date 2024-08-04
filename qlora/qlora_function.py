import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, logging
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def load_8bit_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    return model

def load_quantized_model(model_name):
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=qconf, device_map="auto",
        trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model


if __name__ == "__main__":
	model_name = "NousResearch/Llama-2-7b-hf"
	model = load_quantized_model(model_name)