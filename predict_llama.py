# -*- coding:utf-8 -*-

import torch
import json
from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer
from peft import PeftModel
from tqdm import tqdm
import time
import argparse
import re
from rdkit.Chem import AllChem
from rdkit import Chem
import os


def convert_cano(smi):
    try:
        mol = AllChem.MolFromSmiles(smi)
        smiles = Chem.MolToSmiles(mol)
    except:
        smiles = '####'
    return smiles


def find_chem(text):
    pattern = r"[A-Za-z0-9()\[\]#@=\-\+\/\\]+\d*[A-Za-z0-9()\[\]#@=\-\+\/\\]*\d*"
    chemicals = re.findall(pattern, text)
    return chemicals


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/uspto_test.json', type=str, help='')
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--ori_model_dir', default="Chinese-LLaMA-7B/", type=str, help='Chinese-Alpaca-7B')
    parser.add_argument('--model_dir', default="output_dir_lora_llama_2task/", type=str, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--max_src_len', type=int, default=350, help='')
    parser.add_argument('--prompt_text', type=str, default="", help='')
    return parser.parse_args()


def main():
    args = set_args()
    acc_dict = {}
    file_list = os.listdir(args.model_dir)
    for file in file_list:
        model_dir = os.path.join(args.model_dir, file)

        model = LlamaForCausalLM.from_pretrained(args.ori_model_dir)
        tokenizer = LlamaTokenizer.from_pretrained(args.ori_model_dir)
        model.eval()
        model = PeftModel.from_pretrained(model, model_dir, torch_dtype=torch.half)
        model.half().to("cuda:{}".format(args.device))
        model.eval()
        save_data = []
        acc_total = 0.0
        total = 0.0
        max_tgt_len = args.max_len - args.max_src_len - 3
        s_time = time.time()
        with open(args.test_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(tqdm(fh, desc="iter")):
                sample = json.loads(line.strip())
                with torch.no_grad():
                    src_tokens = tokenizer.tokenize(sample["text"])

                    prompt_tokens = tokenizer.tokenize(args.prompt_text)

                    if len(src_tokens) > args.max_src_len - len(prompt_tokens):
                        src_tokens = src_tokens[:args.max_src_len - len(prompt_tokens)]

                    tokens = ["<s>"] + prompt_tokens + src_tokens
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    # input_ids = tokenizer.encode("帮我写个快排算法")

                    input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "min_length": 5,
                        "max_new_tokens": max_tgt_len,
                        "temperature": 0.95,
                        "do_sample": False,
                        "num_return_sequences": 1,
                        "num_beams": 4,
                    }
                    response = model.generate(**generation_kwargs)
                    res = []
                    for i_r in range(generation_kwargs["num_return_sequences"]):
                        outputs = response.tolist()[i_r][input_ids.shape[1]:]
                        r = tokenizer.decode(outputs).replace("</s>", "")
                        res.append(r)

                    tgt_chemical = convert_cano(".".join(find_chem(sample["answer"])))
                    chemicals = [convert_cano(".".join(find_chem(r))) for r in res]
                    if tgt_chemical in chemicals:
                        acc = 1
                        acc_total += 1
                    else:
                        acc = 0
                    total += 1
                    save_data.append({"text": sample["text"], "acc": acc, "gen_answer": res, "gen_chemicals": chemicals,
                                      "tgt_chemical": tgt_chemical})
        e_time = time.time()
        print("总耗时：{}s".format(e_time - s_time))
        print(acc_total / total)
        fin = open(os.path.join(model_dir, "task1_answer_top1.json"), "w", encoding="utf-8")
        json.dump(save_data, fin, ensure_ascii=False, indent=4)
        fin.close()
        acc_dict[file] = acc_total / total
    print(acc_dict)


if __name__ == '__main__':
    main()
