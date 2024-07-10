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