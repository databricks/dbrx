# DBRX Base

* DBRX Base is a mixture-of-experts (MoE) large language model trained from scratch by Databricks.
* We are releasing both DBRX Base, a pretrained base model, and DBRX Instruct, a fine-tuned version for few-turn interactions, under [an open license](https://www.databricks.com/legal/open-model-license).
* This is the repository for DBRX Base. DBRX Instruct can be found [here](https://huggingface.co/databricks/dbrx-instruct).
* For full details on the DBRX models, please read our [technical blog post](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).


## Model Overview
DBRX is a [transformer-based](https://www.isattentionallyouneed.com/) decoder-only large language model (LLM) that was trained using next-token prediction. 
It uses a *fine-grained* mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input. 
It was pre-trained on 12T tokens of text and code data. 
Compared to other open MoE models like Mixtral-8x7B and Grok-1, DBRX is fine-grained, meaning it uses a larger number of smaller experts. DBRX has 16 experts and chooses 4, while Mixtral-8x7B and Grok-1 have 8 experts and choose 2. 
This provides 65x more possible combinations of experts and we found that this improves model quality. 
DBRX uses rotary position encodings (RoPE), gated linear units (GLU), and grouped query attention (GQA). 
It uses the GPT-4 tokenizer as provided in the [tiktoken](https://github.com/openai/tiktoken) repository. 
We made these choices based on exhaustive evaluation and scaling experiments.

DBRX was pretrained on 12T tokens of carefully curated data and a maximum context length of 32K tokens. 
We estimate that this data is at least 2x better token-for-token than the data we used to pretrain the MPT family of models. 
This new dataset was developed using the full suite of Databricks tools, including Apache Spark™ and Databricks notebooks for data processing, and Unity Catalog for data management and governance. 
We used curriculum learning for pretraining, changing the data mix during training in ways we found to substantially improve model quality.

* **Inputs:** DBRX only accepts text-based inputs and accepts a context length of up to 32768 tokens.
* **Outputs:** DBRX only produces text-based outputs.  
* **Model Architecture:** More detailed information about DBRX Instruct and DBRX Base can be found in our [technical blog post](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).
* **License:** [Databricks Open Model License](https://www.databricks.com/legal/open-model-license)
* **Acceptable Use Policy:** [Databricks Open Model Acceptable Use Policy](https://www.databricks.com/legal/acceptable-use-policy-open-model)
* **Version:** 1.0
* **Owner:** Databricks, Inc.


## Usage
These are several general ways to use the DBRX models: 
* DBRX Base and DBRX Instruct are available for download on Hugging Face (see our Quickstart guide below). This is the HF repository for DBRX Base; DBRX Instruct can be found [here](https://huggingface.co/databricks/dbrx-instruct). 
* The DBRX model repository can be found on GitHub [here](https://github.com/databricks/dbrx). 
* DBRX Base and DBRX Instruct are available with [Databricks Foundation Model API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) via both *Pay-per-token* and *Provisioned Throughput* endpoints. These are enterprise-ready deployments.
* For more information on how to fine-tune using LLM-Foundry, please take a look at our LLM pretraining and fine-tuning [documentation](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md).


## Quickstart Guide
**NOTE This is DBRX Base, and has not been instruction finetuned. It has not been trained for interactive chat and is only a completion model.**
If you are looking for the finetuned model, please use [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct).

Getting started with DBRX models is easy with the `transformers` library. The model requires ~264GB of RAM and the following packages:

```bash
pip install transformers tiktoken
```

If you'd like to speed up download time, you can use the `hf_transfer` package as described by Hugging Face [here](https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads).
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### Run the model on a CPU:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-base", device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True)

input_text = "Databricks was founded in "
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Run the model on multiple GPUs:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-base", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

input_text = "Databricks was founded in "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```
If your GPU system supports [FlashAttention2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2), you can add `attn_implementation=”flash_attention_2”` as a keyword to `AutoModelForCausalLM.from_pretrained()` to achieve faster inference.


## Limitations and Ethical Considerations
### Training Dataset Limitations
The DBRX models were trained on 12T tokens of text, with a knowledge cutoff date of December 2023.

The training mix used for DBRX contains both natural-language and code examples. The vast majority of our training data is in the English language. We did not test DBRX for non-English proficiency. Therefore, DBRX should be considered a generalist model for text-based use in the English language.

DBRX does not have multimodal capabilities.

### Associated Risks and Recommendations 
All foundation models are novel technologies that carry various risks, and may output information that is inaccurate, incomplete, biased, or offensive. 
Users should exercise judgment and evaluate such output for accuracy and appropriateness for their desired use case before using or sharing it. 
Databricks recommends [using retrieval augmented generation (RAG)](https://www.databricks.com/glossary/retrieval-augmented-generation-rag) in scenarios where accuracy and fidelity are important. 
We also recommend that anyone using or fine-tuning either DBRX Base or DBRX Instruct perform additional testing around safety in the context of their particular application and domain. 


## Intended Uses
### Intended Use Cases
The DBRX models are open, general-purpose LLMs intended and licensed for both commercial and research applications. 
They can be further fine-tuned for various domain-specific natural language and coding tasks. 
DBRX Base can be used as an off-the-shelf model for text completion for general English-language and coding tasks. 

Please review the Associated Risks section above, as well as the [Databricks Open Model License](https://www.databricks.com/legal/open-model-license) and [Databricks Open Model Acceptable Use Policy](https://www.databricks.com/legal/acceptable-use-policy-open-model) for further information about permissible uses of DBRX Base and its derivatives. 

### Out-of-Scope Use Cases
DBRX models are not intended to be used out-of-the-box in non-English languages and do not support native code execution, or other forms of function-calling. 
DBRX models should not be used in any manner that violates applicable laws or regulations or in any other way that is prohibited by the [Databricks Open Model License](https://www.databricks.com/legal/open-model-license) and [Databricks Open Model Acceptable Use Policy](https://www.databricks.com/legal/acceptable-use-policy-open-model). 


## Training Stack
MoE models are complicated to train, and the training of DBRX Base and DBRX Instruct was heavily supported by Databricks’ infrastructure for data processing and large-scale LLM training (e.g., [Composer](https://github.com/mosaicml/composer), [Streaming](https://github.com/mosaicml/streaming), [Megablocks](https://github.com/stanford-futuredata/megablocks), and [LLM Foundry](https://github.com/mosaicml/llm-foundry)). 

Composer is our core library for large-scale training. 
It provides an optimized training loop, easy [checkpointing](https://docs.mosaicml.com/projects/composer/en/latest/trainer/checkpointing.html) and [logging](https://docs.mosaicml.com/projects/composer/en/latest/trainer/logging.html#wood-logging), 
[FSDP](https://pytorch.org/docs/stable/fsdp.html)-based [model sharding](https://docs.mosaicml.com/projects/composer/en/latest/notes/distributed_training.html#fullyshardeddataparallel-fsdp), 
convenient [abstractions](https://docs.mosaicml.com/projects/composer/en/latest/trainer/time.html), extreme customizability via [callbacks](https://docs.mosaicml.com/projects/composer/en/latest/trainer/callbacks.html), and more.

Streaming enables fast, low cost, and scalable training on large datasets from cloud storage. It handles a variety of challenges around deterministic resumption as node counts change, avoiding redundant downloads across devices, high-quality shuffling at scale, sample-level random access, and speed.

Megablocks is a lightweight library for MoE training. Crucially, it supports “dropless MoE,” which avoids inefficient padding and is intended to provide deterministic outputs for a given sequence no matter what other sequences are in the batch.

LLM Foundry ties all of these libraries together to create a simple LLM pretraining, fine-tuning, and inference experience.

DBRX was trained using proprietary optimized versions of the above open source libraries, along with our [LLM training platform](https://www.databricks.com/product/machine-learning/mosaic-ai-training). 


## Evaluation
We find that DBRX outperforms established open-source and open-weight base models on the [Databricks Model Gauntlet](https://www.databricks.com/blog/llm-evaluation-for-icl), the [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and HumanEval. 
The Databricks Model Gauntlet measures performance on more than 30 tasks across six categories: world knowledge, common sense reasoning, language understanding, reading comprehension, symbolic problem solving, and programming. 
The Hugging Face Open LLM Leaderboard measures the average of ARC-Challenge, HellaSwag, MMLU, TruthfulQA, Winogrande and GSM8k. 
HumanEval measures coding ability.

Full evaluation details can be found in our [technical blog post](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm). 


## Acknowledgements
The DBRX models were made possible thanks in large part to the open-source community, especially:
* The [MegaBlocks](https://arxiv.org/abs/2211.15841) library, which established a foundation for our MoE implementation.
* [PyTorch FSDP](https://arxiv.org/abs/2304.11277), which we built on for distributed training.

