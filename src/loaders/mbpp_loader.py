# src/data/mbpp_loader.py
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader

def get_mbpp_dataloader(
    data_file: str,  # Path to the MBPP JSONL file.
    model_name_or_path: str,
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = False,
    start_id = 0,
    end_id = None
):
    """
    Loads the MBPP dataset from a JSONL file, tokenizes the 'text' field,
    and returns a DataLoader along with the tokenizer.
    
    The MBPP JSONL file is expected to contain fields such as:
      - "text" (the prompt),
      - "code" (expected solution code),
      - "task_id",
      - "test_setup_code",
      - "test_list",
      - "challenge_test_list"
    
    Args:
        data_file (str): Path to the MBPP .jsonl file.
        model_name_or_path (str): Pretrained model checkpoint name (for the tokenizer).
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum token length for tokenizing the prompt.
        shuffle (bool): Whether to shuffle the dataset.
        
    Returns:
        dataloader: A PyTorch DataLoader containing the tokenized prompts and original fields.
        tokenizer: The tokenizer used for processing.
    """
    # Load the dataset from the JSONL file.
    dataset = load_dataset("json", data_files=data_file, split="train")
    if end_id is not None:
        dataset = dataset.select(range(start_id, end_id))
    else:
        dataset = dataset.select(range(start_id, len(dataset)))
    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Tokenize the "text" field.
    def tokenize_fn(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length  # Account for additional tokens
        )
        return tokenized

    # Map tokenization over the dataset.
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    # Set format to torch for the tokenized fields and keep all other columns.
    tokenized_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask"],
        output_all_columns=False
    )
    
    # Create a DataLoader with a default collator.
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=default_data_collator
    )
    
    return dataloader, tokenizer

# Append the following main function at the end of src/data/mbpp_loader.py

# if __name__ == "__main__":
#     import tempfile
#     import json

#     # Create sample MBPP data
#     sample_data = [
#         {
#             "text": "Write a function to add two numbers.",
#             "code": "def add(a, b): return a + b",
#             "task_id": 1,
#             "test_setup_code": "",
#             "test_list": ["assert add(1, 2) == 3"],
#             "challenge_test_list": []
#         },
#         {
#             "text": "Write a function to multiply two numbers.",
#             "code": "def multiply(a, b): return a * b",
#             "task_id": 2,
#             "test_setup_code": "",
#             "test_list": ["assert multiply(2, 3) == 6"],
#             "challenge_test_list": []
#         }
#     ]

#     # Create a temporary JSONL file and write the sample data
#     with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_file:
#         for entry in sample_data:
#             tmp_file.write(json.dumps(entry) + "\n")
#         temp_filename = tmp_file.name

#     print(f"Temporary JSONL file created at: {temp_filename}")

#     # Use a lightweight model for tokenization (e.g., "bert-base-uncased")
#     dataloader, tokenizer = get_mbpp_dataloader(
#         data_file=temp_filename,
#         model_name_or_path="bert-base-uncased",
#         batch_size=1,
#         max_length=32,
#         shuffle=False
#     )

#     # Retrieve one batch and print its contents to verify fields.

#     for batch in dataloader:
#         print("Batch keys:", list(batch.keys()))
#         print("input_ids shape:", batch["input_ids"].shape)
#         print("attention_mask shape:", batch["attention_mask"].shape)
#         # Extra fields should be preserved (they are not tensorized by default_data_collator)
#         if "text" in batch:
#             print("text:", batch["text"])
#         if "code" in batch:
#             print("code:", batch["code"])
#         if "task_id" in batch:
#             print("task_id:", batch["task_id"])
#         if "test_list" in batch:
#             print("test_list:", batch["test_list"])
#         break

#     # Optionally, remove the temporary file
#     import os
#     os.remove(temp_filename)
