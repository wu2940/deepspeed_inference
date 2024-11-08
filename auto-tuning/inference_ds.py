from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch
from huggingface_hub import login
import time 
from transformers import TextStreamer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# deepspeed --autotuning run --num_gpus=1 /media/ailab/20T/dailin/fine-tune/excal_deepspeed/inference/auto-tuning/inference_ds.py --deepspeed /media/ailab/20T/dailin/fine-tune/excal_deepspeed/inference/auto-tuning/ds_config.json

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
# model_name = "internlm/internlm2-wqx-20b"
# model_name = "01-ai/Yi-1.5-34B-Chat"
# model_name = "meta-llama/Meta-Llama-3-70B"

config = AutoConfig.from_pretrained(model_name)


# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

ds_config = {
        "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "nvme_path": "/media/ailab/inno_disk",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
            "nvme_path": "/media/ailab/inno_disk",
            "buffer_count": 3,
            "buffer_size": 11e8,
            # "max_in_cpu": 11e7
        },
        },
        "train_micro_batch_size_per_gpu": 1,
        "fp16": {
          	"enabled": True
        },
        "autotuning": {
          	"enabled": True,
          	"fast": False
        }
}


dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


if model_name == "internlm/internlm2-wqx-20b" :
    target_modules=["wqkv", "wo"]
else :
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]


# # 将LoRA应用于模型
# model = get_peft_model(model)



# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 準備輸入
text = "請你幫我介紹淡江大學"
inputs = tokenizer(text, return_tensors="pt").to(device=local_rank)


text_streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
t1 = time.time()
outputs = ds_engine.module.generate(**inputs, 
                         streamer=text_streamer, # our streamer object!
                         do_sample=True,
                         max_new_tokens=20)


t2 = time.time()

print("time=", t2 - t1)