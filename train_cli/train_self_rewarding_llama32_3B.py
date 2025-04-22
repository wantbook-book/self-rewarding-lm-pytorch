import torch
from torch import Tensor
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from self_rewarding_lm_pytorch import SelfRewardingTrainer
from datasets import load_dataset, DatasetDict  # Assuming you're using HuggingFace's `datasets` library
from torch.utils.data import Dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
import debugpy
debugpy.listen(('localhost', 5678))
debugpy.wait_for_client()

class PromptDataset(Dataset):
    def __init__(self, dataset: DatasetDict, tokenizer: LlamaTokenizer, max_length: int = 512):
        """
        Custom Dataset to extract only 'problem' field.
        
        Args:
            dataset: The original dataset.
            tokenizer: The tokenizer to use for tokenizing the 'problem'.
            max_length: Maximum sequence length for tokenization.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.process_data()

    def process_data(self):
        processed_data = []
        
        for item in self.dataset['train']:
            problem = item['problem']
            
            # Tokenize the 'problem' field
            # encoding = self.tokenizer(problem, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            
            # Extract input_ids (the tokenized form of 'problem') from the encoding
            # input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
            processed_data.append(problem)
        
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomSftDataset(Dataset):
    def __init__(self, dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int = 1024):
        """
        Custom dataset class to handle tokenization of problem and solution fields, and creating the mask.
        
        Args:
            dataset: The DatasetDict containing the data.
            tokenizer: The tokenizer to use (LlamaTokenizer or other).
            max_length: Maximum sequence length after tokenization.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.process_data()

    def process_data(self):
        processed_data = []
        
        for item in self.dataset['train']:
            problem = item['problem']
            solution = item['solution']
            
            # Tokenize problem and solution together
            problem_tokens = self.tokenizer(problem, truncation=True, padding=False, max_length=self.max_length // 2)
            solution_tokens = self.tokenizer(solution, truncation=True, padding=False, max_length=self.max_length // 2)
            
            # Concatenate problem and solution tokens
            input_ids = problem_tokens['input_ids'] + solution_tokens['input_ids']
            
            # Create the mask
            mask = [False] * len(problem_tokens['input_ids']) + [True] * len(solution_tokens['input_ids'])
            
            # Padding to the max_length if necessary
            if len(input_ids) < self.max_length:
                input_ids += [self.tokenizer.eos_token_id] * (self.max_length - len(input_ids))
                mask += [False] * (self.max_length - len(mask))

            # Ensure that the length of input_ids and mask are equal
            assert len(input_ids) == len(mask) == self.max_length
            processed_data.append((torch.tensor(input_ids), torch.tensor(mask)))
        
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Load the pre-trained Llama model and tokenizer from Hugging Face
# model_name = 'llama3.2-3B'  # Use the correct model name here
model_name = "/c22940/fwk/model/meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your custom dataset
sft_dataset_path = '/c22940/fwk2/code/self-rewarding-lm-pytorch/data/train_with_idx.jsonl'  # Adjust this path as needed
# Assuming the dataset is in a format that HuggingFace `datasets` can load, e.g., CSV, JSON, or Parquet
# 处理成seq tokens和mask 000011111
dataset = load_dataset('json', data_files=sft_dataset_path)  # Adjust the format and split as needed
dataset['train'] = dataset['train'].shuffle(seed=42).select([i for i in range(16)])
sft_dataset = CustomSftDataset(dataset, tokenizer)
# Ensure the dataset format aligns with the structure you expect for your trainer

# Prepare the prompt dataset (example mock data used for simplicity)
# 处理成只包含prompt
dataset = load_dataset('json', data_files=sft_dataset_path)
dataset['train'] = dataset['train'].shuffle(seed=42).select([i for i in range(16)])
prompt_dataset = PromptDataset(dataset, tokenizer)

# Modify the tokenizer functions
def decode_tokens(tokens: Tensor) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True)

def encode_str(seq_str: str) -> Tensor:
    return tokenizer(seq_str, return_tensors='pt')['input_ids']

# Instantiate the trainer
trainer = SelfRewardingTrainer(
    model=model,
    finetune_configs=dict(
        train_sft_dataset=sft_dataset,
        self_reward_prompt_dataset=prompt_dataset,
        dpo_num_train_steps=1000,
        sft_config={
            'trainer_kwargs': {
                'batch_size': 1,
                'grad_accum_steps': 16
            }
        },
        self_reward_config={
            'reward_generator_kwargs': {
                'batch_size': 1,
            },
            'trainer_kwargs': {
                'batch_size': 1
            }
        }

    ),
    tokenizer_decode=decode_tokens,
    tokenizer_encode=encode_str,
    accelerate_kwargs=dict(
        cpu=False,
    )
)

# Start the training process
trainer(overwrite_checkpoints=True)

# Checkpoints will be saved to ./checkpoints after each finetuning stage
