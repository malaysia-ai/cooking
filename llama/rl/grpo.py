import re
import click
import torch
import fasttext
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

filename = hf_hub_download(
    repo_id="mesolitica/fasttext-language-detection-bahasa-en", 
    filename="fasttext.ftz"
)
bahasa_en = fasttext.load_model(filename)

filename = hf_hub_download(
    repo_id="mesolitica/fasttext-language-detection-ms-id", 
    filename="fasttext.ftz"
)
ms_id = fasttext.load_model(filename)

def not_chinese(completions, **kwargs):
    return [float(not bool(chinese_pattern.search(completion[0]["content"]))) for completion in completions]

def is_malay(completions, **kwargs):
    results = []
    for completion in completions:
        t = completion[0]["content"].replace('\n', ' ')
        label_bahasa_en = bahasa_en.predict(t)[0][0] == '__label__bahasa'
        label_ms_id = ms_id.predict(t)[0][0] == '__label__standard-malay'
        results.append(float(label_bahasa_en and label_ms_id))
    return results
    
def length_reward_func(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(completion[0]["content"].split()) / 4096) for completion in completions]

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def correct_reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

@click.command()
@click.option('--model_name', default='malaysia-ai/Malaysian-Llama-3.2-1B-Instruct')
@click.option('--system', default='You are going to enter reasoning mode, first you try to think step-by-step in malay after that give the final answer.')
@click.option('--lora_rank', default=64)
@click.option('--lora_alpha', default=128)
def main(model_name, system, lora_rank, lora_alpha):
    dataset = load_dataset("malaysia-ai/Malaysia-QA", split="train")
    dataset = dataset.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': x['question']}
        ],
        'ground_truth': x['answer']
    })
    training_args = GRPOConfig(
        output_dir=f'output-{model_name}'.replace('/', '-'),
        run_name=f'GRPO-{model_name}'.replace('/', '-'),
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_steps = 50,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=4096,
        num_train_epochs=5,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.8,
        vllm_max_model_len=16384,
        vllm_enforce_eager=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        deepspeed='ds_config_zero3.json'
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None
    )
    if lora_rank > 0:
        """
        Need to verify does merge load from peft properly merge shared tied weight
        """
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
            task_type="CAUSAL_LM",
            bias="none",
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)

    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            not_chinese,
            is_malay,
            length_reward_func,
            format_reward_func,
            correct_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()