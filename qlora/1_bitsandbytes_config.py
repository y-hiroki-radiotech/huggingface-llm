major_version, minor_version = torch.cuda.get_device_capability()

if major_version >= 8:
    !pip install flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation="flash_attention_2"
    print("Your GPU is compatible with FlashAttention and bfloat16.")
else:
    torch_dtype = torch.float16
    attn_implementation="eager"
    print("Your GPU is not compatible with FlashAttention and bfloat16.")
    

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True
)