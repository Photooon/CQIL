# CQIL: Concurrent Computation of Quasi-Independent Layers

## Overview

This is the official repository for CQIL, a technique accelerating LLM inference. Here we offer the convenient acceleration class `CQILWrapper` along with comprehensive training and evaluation code for you.

## Supported Model

- [x] LLaMA1-7B P2 D0
- [x] LLaMA1-7B P2 D1
- [x] LLaMA1-7B P4 D1
- [x] Qwen1.5-14B P2 D0

## Measuing Speedup of CQIL

To measure the speedup achieved by CQIL, follow these steps:
1. Download the LoRA weights and model weights from Huggingface. Details can be found in the `lora_weights/README.md` and `models/README.md` files.
2. Run the folowing command to benchmark the speedup of CQIL:

```bash
python benchmark.py
```

The script will automatically evaluate the output's correctness and calculate the speedup ratio.

## Model Performance

We provide the model performance on several tasks to demonstrate the effect of CQIL. The results are evaluate with LM-Harness framework.

We evaluate the performance of our models across several tasks using the LM-Harness framework. The latency reduction is measured with benchmark.py using A100s with NVLinks. The table below summarizes the results for LLaMA1-7B:

| Model    | Configuration | SciQ | Winogrande | ARC-E | MMLU (5 shots) | Latency Reduction |
| -------- | ------------- | ---- | ---------- | ----- | -------------- | ----------------- |
| LLaMA1-7B | Original     | 93.0 | 70.0 | 72.9 | 35.1 | 0% |
| LLaMA1-7B | P2 D0 S12 E30 | 91.6 | 69.1 | 70.1 | 33.3 | 26.2% |
| LLaMA1-7B | P2 D1 S12 E30 | 91.7 | 68.4 | 70.0 | 32.5 | / |
| LLaMA1-7B | P4 D1 S14 E30 | 90.0 | 68.2 | 65.5 | 33.1 | 37.4% |

Note after acceptance, we discovered that fine-tuning with knowledge distillation leads to better performance. So released models are all fine-tuned with knowledge distillation. However, it's still confusing that with knowledge distillation fine-tuning, bypassing technique would slightly decrease the performance. Therefore, we suggest to use `bypassing distance=0`.

## Fine-tune Model

For more information on fine-tuning the cqil-model, please refer to `train/README.md`.

## Citation

If you find this repository helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{
    zou2024c,
    title={CQIL: Inference Latency Optimization with Concurrent Computation of Quasi-Independent Layers},
    author={Longwei Zou, Qingyang Wang, Han Zhao, Jiangang Kong, Yi Yang, Yangdong Deng},
    booktitle={Annual Meeting of the Association for Computational Linguistics},
    year={2024},
}
```