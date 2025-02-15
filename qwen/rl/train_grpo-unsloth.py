from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import re
import click
import torch
import fasttext
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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
    return [float(len(completion[0]["content"].split()) / 1024) for completion in completions]


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def correct_reward_func(completions, ground_truth,**kwargs):
    # Extract the full answer text from <answer>...</answer>
    matches = [re.search(r"<answer>(.*?)</answer>", completion[0]['content']) for completion in completions]
    contents = [match.group(1).lower().strip() if match else "" for match in matches]
    ground_truth = [x.split() for x in ground_truth]

    return [1.0 if c in gt else 0.0 for c, gt in zip(contents, ground_truth)]


@click.command()
@click.option('--model_name', default='malaysia-ai/Malaysian-Qwen2.5-1.5B-Instruct')
def main(model_name):
    dataset = load_dataset("malaysia-ai/Malaysia-QA", split="Tatabahasa_QA")
    
    dataset = dataset.map(lambda x: {
        'prompt': [
            {
                "role": "system",
                "content": """You are going to enter reasoning mode. First, you try to think step-by-step in Malay. After that, give the final answer.
    
                Respond in the following format:
                <thinking>
                ...
                </thinking>
                <answer>
                ...
                </answer>"""
            },
            {
                'role': 'user',
                'content': x['question']
            }
        ],
        'ground_truth': ' '.join([x['answer'].lower().strip(),next(
            (line.split(". ")[1].lower().strip() for line in x['question'].split("\n") if line.startswith(f"{x['answer']}. ")),
            None  # Default value if not found
        )])
    
    })


    training_args = GRPOConfig(
        output_dir=f'output-{model_name}'.replace('/', '-'),
        run_name=f'GRPO-{model_name}'.replace('/', '-'),
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=1024,
        num_train_epochs=100,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        vllm_device='cuda:0',
        vllm_gpu_memory_utilization=0.9,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False
    )
    print(training_args)

    model,tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length= 1024,
        fast_inference = True, # Enable vLLM fast inference
        device_map = 'cuda:0',

    )

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        lora_dropout=0,
    )

    print(model)
    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=tokenizer,
        reward_funcs=[
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