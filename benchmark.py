import time
import torch
import torch.nn as nn

from cqil_model import CQILWrapper
from copy import deepcopy
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Benchmark(nn.Module):
    def __init__(self, model, start_l, end_l, device_count):
        super().__init__()
        self.model = model
        self.layer_ids = []

        i = 0
        while i < len(model.model.layers):
            if i < start_l or i >= end_l:
                self.layer_ids.append([i])
                i += 1
            else:
                self.layer_ids.append(list(range(i, i+device_count)))
                i += device_count

    def forward(self, input_ids):
        x = self.model.model.embed_tokens(input_ids)
        for arr in self.layer_ids:
            del_x = None
            for i, l in enumerate(arr):
                layer_out = self.model.model.layers[l](x)[0]
                layer_out = layer_out - x.to(layer_out.device)
                if i == 0:
                    del_x = layer_out
                else:
                    del_x = del_x.to(layer_out.device)
                    del_x += layer_out
            x = x.to(del_x.device)
            x += del_x
        x = self.model.model.norm(x)
        x = self.model.lm_head(x)

        return (x, )

if __name__ == "__main__":
    model_path = "models/Qwen1.5-14B"
    lora_path = "lora_weights/Qwen1.5-14B-S17-E37-P2"
    start_l, end_l, device_count = 17, 37, 2

    # Load LoRA merged models
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(model, lora_path)
    print("Warning: Sometimes merge_and_unload lora costs unreasonable long time ... ")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Wrap with Benchmark and CQIL
    base_model = Benchmark(model, start_l, end_l, device_count)
    cqil_model = CQILWrapper(deepcopy(model), start_l, end_l, device_count)

    warmup = 3          # ignore information in warmup iterations
    iteration = 10
    save_ratios = []    # Cost
    diffs = []

    base_model.eval()
    base_model = base_model.to(0)
    cqil_model = cqil_model.to(0)

    input_ids = tokenizer(
        "接下来我要测试一段非常长的文本：",
        return_tensors="pt",
        padding=True
    )["input_ids"].cuda()

    with torch.no_grad():
        for i in range(iteration):
            t1 = time.time()
            sim_out = base_model(input_ids)[0]
            t2 = time.time()
            out = cqil_model(input_ids)[0]
            t3 = time.time()

            sim_out = sim_out.cpu()
            out = out.cpu()
            if i < warmup:
                continue

            save_ratios.append(1 - (t3 - t2) / (t2 - t1))
            diffs.append(torch.mean(torch.abs((sim_out - out) / sim_out)).item())
            sim_out_char = tokenizer.decode([torch.argmax(sim_out[0][-1]).item()], skip_special_tokens=True)
            out_char = tokenizer.decode([torch.argmax(out[0][-1]).item()], skip_special_tokens=True)
            if sim_out_char != out_char:
                print(f"Warning: Difference Output from two models, {sim_out_char} and {out_char}")
    
    cqil_model.close()
    print(f"Average Absolute Difference Ratio: {sum(diffs) / len(diffs) * 100: .2f}%")
    print(f"Average Time Saving Ratio: {sum(save_ratios) / len(save_ratios) * 100: .2f}%")