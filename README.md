# CQIL: Concurrent Computation of Quasi-Independent Layers

## Overview

This is the official repository for CQIL, a technique accelerating LLM inference. Here we offer the convenient acceleration class `CQILWrapper` along with comprehensive training and evaluation code for you.

## Supported Model

- [ ] LLaMA1-7B P2 D0
- [ ] LLaMA1-7B P2 D1
- [ ] LLaMA1-7B P4 D1
- [x] Qwen1.5-14B

## Model Performance

We provide the model performance on several tasks to demonstrate the effect of CQIL. The results are evaluate with LM-Harness framework.

We evaluate the performance of our models across several tasks using the LM-Harness framework. The table below summarizes the results for LLaMA1-7B:

| Model    | Configuration | SciQ | Winogrande | ARC-C | MMLU (5 shots) |
| -------- | ------------- | ---- | ---------- | ----- | -------------- |
| LLaMA1-7B | Original     | 93.0 | 70.0 | 41.3 | 35.1 |
| LLaMA1-7B | P2 D0 S13 E30 | | | | |
| LLaMA1-7B | P2 D1 S13 E30 | | | | |
| LLaMA1-7B | P4 D1 S15 E30 | | | | |

## Measuing Speedup of CQIL

To measure the speedup achieved by CQIL, follow these steps:
1. Download the LoRA weights and model weights from Huggingface. Details can be found in the `lora_weights/README.md` and `models/README.md` files.
2. Run the folowing command to benchmark the speedup of CQIL:

```bash
python benchmark.py
```

The script will automatically evaluate the output's correctness and calculate the speedup ratio.

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