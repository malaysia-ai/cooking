#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.

import torch
from torch import nn
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset
from datasets import Audio

import transformers
import random
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    is_torch_tpu_available,
    set_seed,
    Qwen2AudioForConditionalGeneration,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import json
import os
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from peft import LoraConfig, get_peft_model
from cut_cross_entropy import linear_cross_entropy

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

class UInt32(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint32)

_encodings['uint32'] = UInt32

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    rank: int = field(
        default=256,
        metadata={
            "help": "lora rank"
        },
    )

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine.")}, )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}, )
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set.")}, )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set.")}, )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)

class Model(Qwen2AudioForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, attention_mask, position_ids, input_features = None, feature_attention_mask = None, labels = None, **kwargs):
        super_out = super().forward(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            position_ids = position_ids,
            input_features = input_features, 
            feature_attention_mask = feature_attention_mask,
        )
        if labels is not None:
            logits = super_out.logits
            vocab_size = logits.shape[-1]
            logits = logits.float()
            labels = labels.to(logits.device)
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()
            logits = logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(logits.device)
            loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction='mean')
            return {'loss': loss}

def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning)
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level
        # at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    audio_token = "<|AUDIO|>"
    audio_bos_token = "<|audio_bos|>"
    audio_eos_token = "<|audio_eos|>"
    audio_token_id = processor.tokenizer._convert_token_to_id_with_added_voc('<|AUDIO|>')
    pad_token_id = processor.tokenizer.pad_token_id

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = LocalDataset(local=local)
            self.audio = Audio(sampling_rate=16000)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            try:
                f = data['audio']
                if len(f):
                    f = f.replace('output-audio/', 'filter-audio/')
                    if not os.path.exists(f):
                        return None
                    audio = self.audio.decode_example(
                        self.audio.encode_example(f))['array']

                    inputs_audio = processor.feature_extractor([audio], return_attention_mask=True, padding="max_length", return_tensors = 'pt')
                    audio_lengths = inputs_audio["attention_mask"].sum(-1).tolist()

                    sample = data['text']
                    num_audio_tokens = sample.count(audio_token)
                    replace_str = []
                    while audio_token in sample:
                        audio_length = audio_lengths.pop(0)
                        input_length = (audio_length - 1) // 2 + 1
                        num_audio_tokens = (input_length - 2) // 2 + 1

                        expanded_audio_token = audio_token * num_audio_tokens

                        audio_token_start_idx = sample.find(audio_token)
                        audio_token_end_idx = audio_token_start_idx + len(audio_token)

                        has_bos = (
                            sample[audio_token_start_idx - len(audio_bos_token) : audio_token_start_idx]
                            == audio_bos_token
                        )
                        has_eos = (
                            sample[audio_token_end_idx : audio_token_end_idx + len(audio_eos_token)]
                            == audio_eos_token
                        )

                        if not has_bos and not has_eos:
                            expanded_audio_token = audio_bos_token + expanded_audio_token + audio_eos_token

                        replace_str.append(expanded_audio_token)
                        sample = sample.replace(audio_token, "<placeholder>", 1)

                    while "<placeholder>" in sample:
                        sample = sample.replace("<placeholder>", replace_str.pop(0), 1)

                    inputs = {
                        'input_ids': sample,
                        'input_features': inputs_audio['input_features'],
                        'feature_attention_mask': inputs_audio['attention_mask'],
                    }
                    return inputs
                else:
                    data.pop('text', None)
                    data.pop('audio')
                    data['labels'] = data["input_ids"].copy()
                    masking = data.pop('attention_mask')

                    data.pop('token_type_ids', None)
                    for k in data.keys():
                        data[k] = torch.tensor(data[k].astype(np.int64))

                    masks = []
                    for m in masking:
                        masks.append(torch.tril(torch.ones(m, m)))
                    attention_mask = block_diagonal_concat_inverted(*masks)
                    data['attention_mask'] = attention_mask
                    return data

            except Exception as e:
                print(e)
                return None

        def __len__(self):
            return len(self.dataset)

    def collator(batch):
        batch = [b for b in batch if b is not None] 
        input_ids, attention_mask, position_ids, labels = [], [], [], []
        input_features, feature_attention_mask = [], []

        for b in batch:
            if 'input_features' in b:
                input_ids_ = processor.tokenizer(
                    [b['input_ids']], 
                    return_tensors = 'pt', 
                    truncation = True, 
                    max_length = sequence_length,
                    padding = 'max_length',
                )
                attention_mask_ = input_ids_['attention_mask']
                labels_ = input_ids_['input_ids'].clone()
                labels_[labels_ == audio_token_id] = -100
                labels_[labels_ == pad_token_id] = -100
                input_ids.append(input_ids_['input_ids'])
                labels.append(labels_)
                cache_position = torch.arange(0, input_ids_['input_ids'].shape[1])
                causal_mask = torch.full(
                    (sequence_length, sequence_length), fill_value=min_dtype, dtype=torch_dtype
                )
                causal_mask *= torch.arange(sequence_length) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :]
                causal_mask = causal_mask.clone()
                mask_length = attention_mask_.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask_[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
                attention_mask.append(causal_mask)
                position_ids.append(cache_position[None])
                input_features.append(b['input_features'])
                feature_attention_mask.append(b['feature_attention_mask'])
            else:
                input_ids.append(b['input_ids'][None])
                attention_mask.append(b['attention_mask'][None])
                position_ids.append(b['position_ids'][None])
                labels.append(b['labels'][None])

        input_ids = {
            'input_ids': torch.concat(input_ids, 0),
            'attention_mask': torch.concat(attention_mask, 0),
            'position_ids': torch.concat(position_ids, 0),
            'labels': torch.concat(labels, 0),
        }
        if len(input_features):
            input_ids['input_features'] = torch.concat(input_features, 0)
            input_ids['feature_attention_mask'] = torch.concat(feature_attention_mask, 0)

        return input_ids

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0])
    print(collator([dataset[0], dataset[1]]))

    model = Model.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype = torch_dtype,
    )

    selected = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]

    target_modules = []
    for id, (name, param) in enumerate(model.named_modules()):
        if 'language_model.model' in name and any([s in name for s in selected]):
            target_modules.append(name)
    if 'lm_head' in selected:
        target_modules.append('language_model.lm_head')

    peft_config = LoraConfig(
        lora_alpha=model_args.rank * 2,
        lora_dropout=0.0,
        r=model_args.rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

