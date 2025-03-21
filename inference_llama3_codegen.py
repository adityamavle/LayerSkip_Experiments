# scripts/main.py
import csv
import torch
from src.loaders.mbpp_loader import get_mbpp_dataloader
from src.models.model import get_mbpp_model

def run_inference(model, dataloader, tokenizer, device, max_gen_length: int = 256):
    """
    Runs inference on the MBPP dataset using the provided model and dataloader.
    
    For each sample, the function decodes:
      - The original prompt (from "text"),
      - The generated output,
      - The expected solution code (from "code"),
      - The task ID, and
      - The test cases (from "test_list").
      
    Args:
        model: The loaded code generation model.
        dataloader: DataLoader containing tokenized MBPP samples and additional fields.
        tokenizer: Tokenizer used for decoding.
        device: Device on which inference is run.
        max_gen_length (int): Maximum length for generated output.
        
    Returns:
        results (list of dict): Each dict contains task_id, prompt, generated code, expected code, and test_list.
    """
    results = []
    for batch in dataloader:
        # Move tokenized tensors to the device.
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Generate model outputs.
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                max_length=max_gen_length,
                temperature=0.7,
                top_p = 0.9,
            )
        
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            # Retrieve original fields from the batch.
            prompt = batch["text"][i] if "text" in batch else tokenizer.decode(input_ids[i], skip_special_tokens=True)
            expected_code = batch["code"][i] if "code" in batch else ""
            task_id = batch["task_id"][i] if "task_id" in batch else ""
            test_list = batch["test_list"][i] if "test_list" in batch else []
            
            # Decode generated output.
            generated = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            
            results.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated": generated,
                "expected_code": expected_code,
                "test_list": test_list
            })
    return results

def save_results_to_csv(results, output_file: str):
    """
    Saves the inference results to a CSV file.
    
    Args:
        results (list of dict): List containing inference results.
        output_file (str): Path to the CSV output file.
    """
    if not results:
        print("No results to save.")
        return
    
    # Convert list fields (like test_list) to strings.
    for r in results:
        if isinstance(r.get("test_list", None), list):
            r["test_list"] = "; ".join(r["test_list"])
    
    keys = results[0].keys()
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

if __name__ == "__main__":
    # Set up the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define parameters for model and dataset.
    # model_name = "Salesforce/codegen-350M-mono"  # Example code generation model.
    data_file = "src/data/mbpp.jsonl"                # Path to your MBPP JSONL file.
    batch_size = 4
    max_length = 512     # Maximum token length for prompt.
    max_gen_length = 512  # Maximum length for generated output.
    
    # Load the MBPP dataset and tokenizer.
    dataloader, tokenizer = get_mbpp_dataloader(
        data_file=data_file,
        model_name_or_path= "notebooks/model_weights",
        batch_size=batch_size,
        max_length=max_length,
        start_id= 0,
        end_id= 100,
        shuffle=False
    )
    
    # Load the code generation model and ensure it is on the device.
    model_dir = "notebooks/model_weights"  # Path to your folder with model files.
    model = get_mbpp_model(model_dir, device)
    model.resize_token_embeddings(len(tokenizer))
    # Run inference.
    results = run_inference(model, dataloader, tokenizer, device, max_gen_length=max_gen_length)
    
    # Save the inference results to a CSV file.
    output_file = "results1.csv"
    save_results_to_csv(results, output_file)
    print(f"Inference complete. Results saved to {output_file}")
