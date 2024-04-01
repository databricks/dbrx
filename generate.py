from typing import Dict

import torch
import random
import contextlib
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_REPO_NAME = 'databricks/dbrx-instruct'
TOKEN = True  # Necessary to verify that the user has accepted the license for the model
TRUST_REMOTE_CODE = True  # Needs to be True until the model is upstreamed to transformers
SEED = 42

# Generation parameters
MESSAGES = [
    {'role': 'user', 'content': 'What is Machine Learning?'},
]
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.01
MAX_NEW_TOKENS = 100

try:
    import flash_attn
    _flash_attention_installed = True
    assert torch.cuda.is_available()
    attn_implementation = 'flash_attention_2'
    autocast_context = torch.autocast('cuda', torch.bfloat16)
except ImportError:
    _flash_attention_installed = False
    attn_implementation = 'eager'
    autocast_context = contextlib.nullcontext()

tokenizer = AutoTokenizer.from_pretrained(
    HF_REPO_NAME,
    trust_remote_code=TRUST_REMOTE_CODE,
    token=True,
)
model = AutoModelForCausalLM.from_pretrained(
    HF_REPO_NAME,
    trust_remote_code=TRUST_REMOTE_CODE,
    token=True,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    device_map='auto',
)


# Seed randomness
random.seed(SEED)
torch.manual_seed(SEED)
print(f'\nGenerate seed:\n{SEED}')

generate_kwargs = {
    'max_new_tokens': MAX_NEW_TOKENS,
    'temperature': TEMPERATURE,
    'top_p': TOP_P,
    'top_k': TOP_K,
    'repetition_penalty': REPETITION_PENALTY,
    'use_cache': True,
    'do_sample': True,
    'eos_token_id': tokenizer.eos_token_id,
    'pad_token_id': tokenizer.pad_token_id,
}
print(f'\nGenerate kwargs:\n{generate_kwargs}')

# Generate function with correct context managers
def _generate(encoded_inp: Dict[str, torch.Tensor]):
    """
    Generates responses using the loaded model and tokenizer, with the specified generation parameters.

    Args:
        encoded_inp: A dictionary containing the encoded input tensor.

    Returns:
        The generated output from the model.
    """
    with torch.inference_mode():
        with autocast_context:
            return model.generate(
                input_ids=encoded_inp,
                **generate_kwargs,
            )

print(f'\nTokenizing prompts...')
tokenized_chat = tokenizer.apply_chat_template(
    MESSAGES,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors='pt')
tokenized_chat = tokenized_chat.to(model.device)

# Run HF generate
print('Generating responses...\n')
encoded_gen = _generate(tokenized_chat)
decoded_gen = tokenizer.batch_decode(encoded_gen)

# Print generations
delimiter = '#' * 100
prompt_output = ''
for message in MESSAGES:
    prompt_output += (f"{message['role']}: {message['content']}\n")
output = decoded_gen[0].split('<|im_start|> assistant\n')[-1]
output = f"assistant: {output}"
print('\033[92m' + prompt_output + '\033[0m' + output)

print(delimiter)
