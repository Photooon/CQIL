# CQIL: Concurrent Computation of Quasi-Independent Layers

## Brief Introduction

This is the repository for CQIL, accelerating LLM inference. At present, we provide the concurrent inference model and the test benchmark for Qwen1.5-14B. We are working to support more models and will continue to refine the code base.

## Supported Model

- [x] Qwen1.5-14B

## TODO

- [ ] Release Qwen1.5-72B LoRA Weight
- [ ] Release LLaMA LoRA Weight
- [ ] Support LoRA Finetune

## Citation

Please cite our paper if you find the repo helpful for you:

```bibtex
@inproceedings{
    zou2024c,
    title={CQIL: Inference Latency Optimization with Concurrent Computation of Quasi-Independent Layers},
    author={Longwei Zou, Qingyang Wang, Han Zhao, Jiangang Kong, Yi Yang, Yangdong Deng},
    booktitle={Annual Meeting of the Association for Computational Linguistics},
    year={2024},
}
```