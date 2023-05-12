# -*- coding:utf-8 -*-
from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer
import deepspeed
import argparse
from torch.utils.data import RandomSampler, DataLoader
import json
import torch
from torch.utils.data import Dataset
import os
from peft import LoraConfig, get_peft_model


class Seq2SeqDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []

        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(prompt_text)

                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]
                src_tokens = ["<s>"] + src_tokens

                tgt_tokens = tokenizer.tokenize(sample["answer"])
                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]

                tokens = prompt_tokens + src_tokens + tgt_tokens + ["</s>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                context_length = len(prompt_tokens) + len(src_tokens)
                labels = [-100] * context_length + input_ids[context_length:]

                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

                self.all_data.append(
                    {"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    return {"input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list)}


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/uspto_train.json', type=str, help='')
    parser.add_argument('--model_dir', default="Chinese-LLaMA-7B/", type=str,
                        help='')
    parser.add_argument('--num_train_epochs', default=20, type=int, help='')
    parser.add_argument('--train_batch_size', default=4, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_lora_llama_2task/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--max_src_len', type=int, default=350, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--prompt_text', type=str, default="", help='')
    return parser.parse_args()


def main():
    args = set_args()
    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
    model = LlamaForCausalLM.from_pretrained(args.model_dir)

    config = LoraConfig(r=args.lora_r,
                        lora_alpha=32,
                        target_modules=["q_proj", "v_proj", "k_proj"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    model = get_peft_model(model, config)
    model = model.half().cuda()

    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": args.log_steps
            }

    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=True,
                                  num_workers=0)

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()
    global_step = 0
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if global_step % args.log_steps == 0:
                print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        model_engine.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 5557 finetuning_llama.py
