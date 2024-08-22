# LoRA Finetune for Concurrent Architecture

This directory contains scripts for fine-tuning and evaluation. After the paper was accepted, we discovered that using knowledge distillation to fine-tune the modified architecture led to better performance. Therefore, this repository provides scripts specifically for fine-tuning with knowledge distillation.

To maintain compatibility with existing training and evaluation frameworks, we do not directly modify the model architecture during training and evalution of the CQIL model. Instead, we simulate concurrent computation at the layer level by modifying the inputs and outputs of the original model layers. For more details, please refer to the `LlamaDecoderLayer` and `LlamaModel` classes in `modeling_llama.py`. You can adjust the layers that share inputs by editing the `cqil_layers` property in the model's `config.json`.

To avoid excessive memory usage from loading two large language models on GPUs simultaneously, we implement a strategy where the model is forwarded twice: once with and once without LoRA and CQIL layer modifications. This approach significantly reduces memory consumption while ensuring that fine-tuning process works efficiently.

We also proide a tokenized toy dataset for fine-tuning LLaMA1, extracted from RedPajama with the same data mixture ratio as used in LLaMA1's pretraining. Since knowledge distillation is employed, you can substitute this dataset with any other tokenized dataset, such C4. The toy dataset can be downloaded from [Huggingface](). Place it under `train/datasets`

## Example of Fine-tuning LLaMA-7B

We've simplified the fine-tuning process as much as possible. All configurations can be set in `train.sh`, while the DeepSpeed settings are managed in `ds_config.json`.

To start fine-tuning, first copy the checkpoint files to the model directory (or create a symbolic link if disk space is limited):

```bash
cp ../models/LLaMA-7B/*.safetensors models/LLaMA-7B/
cp ../models/LLaMA-7B/*.bin models/LLaMA-7B/
cp ../models/LLaMA-7B/*.index.json models/LLaMA-7B/
```

Make sure you have previously downloaded the checkpoint files and placed them in `../models/LLaMA-7B`

Then, simply run the following command:
```bash
./train.sh
```

After fine-tuning, you can monitor the results in tensorboard. If everything works as expected, the loss curve should appear as follows:

TODO: img of loss

## Evaluation

We use lm-harness framework for evaluating the fine-tuned models, along with Huggingface's accelerate framework to speed up inference. For example, you can measure the MMLU score (5 shots) by running the following command:
```bash
accelerate launch -m lm_eval --model hf --model_args pretrained=models/LLaMA-7B,peft=output/LLaMA-7B-L13-L30-D0-KD --tasks mmlu --num_fewshot 5 --batch_size 8
```
