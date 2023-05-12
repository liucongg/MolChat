## MolChat项目
- 本项目主要在[ChatGLM模型](https://github.com/THUDM/ChatGLM-6B) 、[中文LLaMA模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 上进行Lora微调，验证模型在化学领域的能力。
- 共进行三组对比实验，ChatGLM-6B、Chinese-LLaMA-7B和Chinese-Alpaca-7B。
- Chinese-LLaMA-7B和Chinese-Alpaca-7B模型参数获取与转换，请参考[手动模型合并与转换](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2) ，需要将模型转换成HuggingFace格式。
- ChatGLM-6B模型训练推理请用本项目中代码，并将[百度网盘](https://pan.baidu.com/s/1-UrZWnqw6Ciyo5K2NLraDg) 中的文件，替换官方文件，提取码：jh0l
- 目前仅包含：分子逆合成任务

## Instruction数据构造
训练集：见[uspto_train.json](https://huggingface.co/datasets/Congliu/USPTO-50k-Instruction/tree/main) ，样例如下：
```jsonl
{"text": "目标产物分子CCCCc1ccc(C#Cc2ccc(CN(Cc3ccc(C(=O)OC)cc3)C(=O)CCC3CCCC3)cc2)cc1的合成所需的化合物是什么？", "answer": "实现该目标产物的合成需要分子CCCCc1ccc(C#Cc2ccc(CNCc3ccc(C(=O)OC)cc3)cc2)cc1和O=C(Cl)CCC1CCCC1。"}
{"text": "如何制备出合成目标产物分子CCCCCCCCNC1CC(C)(C)NC(C)(C)C1所需的分子？", "answer": "制备目标产物需要要有分子CC1(C)CC(=O)CC(C)(C)N1和CCCCCCCCN。"}
{"text": "制备目标产物分子NC(=O)[C@H]1[C@@H]2C=C[C@@H](C2)[C@H]1Nc1c(Cl)cnc2[nH]c(Nc3ccccc3)nc12所需的前体分子是什么？", "answer": "目标产物的合成需要分子NC(=O)[C@H]1[C@@H]2C=C[C@@H](C2)[C@H]1Nc1c(Cl)cnc(N)c1N和S=C=Nc1ccccc1。"}
```
测试集：见[uspto_test.json](https://huggingface.co/datasets/Congliu/USPTO-50k-Instruction/tree/main) ，样例如下：
```jsonl
{"text": "如何合成目标产物分子COc1cc(OC)cc(-c2c(C)nnc(C)c2-c2c(F)cc(F)cc2F)c1所需的分子？", "answer": "要得到目标产物，需要使用分子COc1cc(OC)cc(-c2c(C)nnc(Cl)c2-c2c(F)cc(F)cc2F)c1.O=C([O-])[O-]进行合成。"}
{"text": "如何合成目标产物分子COc1ccc([N+](=O)[O-])c(CCOC(=O)Cl)c1所需的分子？", "answer": "要得到目标产物，需要使用分子COc1ccc([N+](=O)[O-])c(CCO)c1.O=C(Cl)Cl进行合成。"}
{"text": "如何合成目标产物分子COc1ccc(-c2ccccc2)cc1CN1C[C@@H]2CCCN2[C@H](C(c2ccccc2)c2ccccc2)C1所需的分子？", "answer": "要得到目标产物，需要使用分子COc1ccc(Br)cc1CN1C[C@@H]2CCCN2[C@H](C(c2ccccc2)c2ccccc2)C1.OB(O)c1ccccc1进行合成。"}
```

## 模型微调方法-Lora
Lora方法，即在大型语言模型上对指定参数（权重矩阵）并行增加额外的低秩矩阵，并在模型训练过程中，仅训练额外增加的并行低秩矩阵的参数。

当“秩值”远小于原始参数维度时，新增的低秩矩阵参数量也就很小。在下游任务tuning时，仅须训练很小的参数，但能获取较好的表现结果。

- 论文：[paper](https://arxiv.org/abs/2106.09685)
- 官方代码：[Github](https://github.com/microsoft/LoRA)
- HuggingFace封装的peft库：[Github](https://github.com/huggingface/peft)

ChatGLM模型微调命令：
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 5555 finetuning_glm.py --model_dir ChatGLM-6B/ --num_train_epochs 20 --train_batch_size 5 --output_dir output_dir_lora_glm_2task/ --max_len 428 --max_src_len 256
```
Chinese-LLaMA-7B模型微调命令：
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 5555 finetuning_llama.py --model_dir Chinese-LLaMA-7B/ --num_train_epochs 20 --train_batch_size 4 --output_dir output_dir_lora_llama_2task/ --max_len 512 --max_src_len 350
```
Chinese-Alpaca-7B模型微调命令：
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 5555 finetuning_llama.py --model_dir Chinese-Alpaca-7B/ --num_train_epochs 20 --train_batch_size 4 --output_dir output_dir_lora_alpaca_2task/ --max_len 512 --max_src_len 350
```
## 结果
| 模型 |  ChatGLM |  Chinese-LLaMA-7B | Chinese-Alpaca-7B |
| ------- | ------ | ------  | ------ |
| acc | 0.22 | 0.21 | 0.26 |

模型测试效果样例：
```
样例1：
输入：如何合成目标产物分子Fc1cc(Br)cc(F)c1OCc1ccccc1所需的分子？
输出：分子BrCc1ccccc1和Oc1c(F)cc(Br)cc1F是必不可少的,以合成该目标产物。
样例2：
输入：如何合成目标产物分子O=C(O)CNc1nc2ncc(-c3cn(Cc4cc(Cl)c(Cl)c(Cl)c4)nn3)nc2s1所需的分子？
输出：分子COC(=O)CNc1nc2ncc(-c3cn(Cc4cc(Cl)c(Cl)c(Cl)c4)nn3)nc2s1是必不可少的,以合成该目标产物。
样例3：
输入：如何合成目标产物分子O=c1c2ccccc2oc2c(O)cccc12所需的分子？
输出：实现该目标产物的合成需要分子COc1cccc2c(=O)c3ccccc3oc12。
```



