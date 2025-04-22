import torch
from torch import Tensor

from self_rewarding_lm_pytorch import (
    SelfRewardingTrainer,
    create_mock_dataset
)

from x_transformers import TransformerWrapper, Decoder

transformer = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 1,
        heads = 8
    )
)

sft_dataset = create_mock_dataset(100, lambda: (torch.randint(0, 256, (256,)), torch.tensor(1)))
prompt_dataset = create_mock_dataset(100, lambda: 'mock prompt')

def decode_tokens(tokens: Tensor) -> str:
    decode_token = lambda token: str(chr(max(32, token)))
    return ''.join(list(map(decode_token, tokens)))

def encode_str(seq_str: str) -> Tensor:
    return Tensor(list(map(ord, seq_str)))

trainer = SelfRewardingTrainer(
    transformer,
    finetune_configs = dict(
        train_sft_dataset = sft_dataset,
        self_reward_prompt_dataset = prompt_dataset,
        dpo_num_train_steps = 1000
    ),
    tokenizer_decode = decode_tokens,
    tokenizer_encode = encode_str,
    accelerate_kwargs = dict(
        cpu = True
    )
)

trainer(overwrite_checkpoints = True)

# checkpoints after each finetuning stage will be saved to ./checkpoints