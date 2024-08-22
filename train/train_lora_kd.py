import os
import json
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig, TaskType


class SelfKdTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

        def kd_loss(logits, teacher_logits, T):
            loss_distil = (
                F.kl_div(
                    F.log_softmax(
                        logits.view(-1, logits.size(-1)) / T,
                        dim=-1,
                    ),
                    F.softmax(
                        teacher_logits.view(-1, teacher_logits.size(-1)) / T,
                        dim=-1,
                    ),
                    reduction="sum",
                )
                * (T) ** 2
            )
            loss_distil /= inputs["input_ids"].size(0)
            return loss_distil

        with torch.no_grad():
            with model.disable_adapter():
                with model.base_model.disable_cqil():
                    t_outputs = model(**inputs)
                    t_logits = t_outputs["logits"] if isinstance(t_outputs, dict) else t_outputs[1]
                    # Zero3 automatically cut off redundant token embed, but the teacher model is ignored.
                    if logits.shape != t_logits.shape:
                        t_logits = t_logits[:, :, :logits.shape[-1]]

        loss = kd_loss(logits, t_logits, 1.0)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

@dataclass
class SelfTrainingArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    device_map: Optional[str] = field(
        default=None,
    )
    lora_rank: Optional[int] = field(
        default=32
    )


def main():
    parser = HfArgumentParser((SelfTrainingArguments, TrainingArguments))
    self_training_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # Dataset and Model Preparation
    dataset = load_from_disk(self_training_args.dataset_name)["train"]

    config_kwargs = {
        "cache_dir": "cache",
        "revision": None,
        "use_auth_token": None,
        "trust_remote_code": True
    }
    config = AutoConfig.from_pretrained(self_training_args.model_name_or_path, **config_kwargs)

    torch_dtype = (
        self_training_args.torch_dtype
        if self_training_args.torch_dtype in ["auto", None]
        else getattr(torch, self_training_args.torch_dtype)
    )

    if self_training_args.device_map is not None and device_map != "auto":
        device_map = json.load(open(self_training_args.custom_device_map, 'r'))
    else:
        device_map = self_training_args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        self_training_args.model_name_or_path,
        config=config,
        cache_dir="cache",
        use_auth_token=None,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    target_modules = []
    if config.model_type in ["llama", "qwen2"]:
        for i in range(0, len(model.model.layers)):
            target_modules += [
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj",
                f"model.layers.{i}.mlp.gate_proj",
                f"model.layers.{i}.mlp.up_proj",
                f"model.layers.{i}.mlp.down_proj",
            ]
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=self_training_args.lora_rank, 
        lora_alpha=self_training_args.lora_rank, 
        lora_dropout=0., 
        target_modules=target_modules
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    print(model)

    # Training
    trainer = SelfKdTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
