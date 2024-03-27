# DBRX

DBRX is a large language model trained by Databricks, and made available under an open license. This repository contains the minimal code and examples to run inference, as well as a collection of resources and links for using DBRX.

* [Founder's Blog](https://www.databricks.com/blog/announcing-dbrx-new-standard-efficient-open-source-customizable-llms), [DBRX Technical Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
* HuggingFace: https://huggingface.co/collections/databricks/
* LLM Foundry: https://github.com/mosaicml/llm-foundry

A reference model code can be found in this repository at [modeling_dbrx.py](model/modeling_dbrx.py). 

**Note:** this model code is supplied for references purposes only, please see the [HuggingFace](https://huggingface.co/collections/databricks/) repository for the official supported version.

## Model details

DBRX is a Mixture-of-Experts (MoE) model with 132B total parameters and 36B live parameters. We use 16 experts, of which 4 are active during training or inference. DBRX was pre-trained for 12T tokens of text. DBRX has a context length of 32K tokens.

The following models are open-sourced:

| Model                                                            | Description                               |
|------------------------------------------------------------------|-------------------------------------------|
| [DBRX Base](https://huggingface.co/databricks/dbrx-base)         | Pre-trained base model                    |
| [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct) | Finetuned model for instruction following |

The model was trained using optimized versions of our open source libraries [Composer](https://www.github.com/mosaicml/composer), [LLM Foundry](https://www.github.com/mosaicml/llm-foundry), [MegaBlocks](https://github.com/databricks/megablocks) and [Streaming](https://github.com/mosaicml/streaming).

For the instruct model, we used the ChatML format. Please see the [DBRX Instruct model card](./MODEL_CARD_dbrx_instruct.md) for more information on this.


## Quick start

To download the weights and tokenizer, please first visit the DBRX HuggingFace page and accept the license. Note: access to the Base model requires manual approval. 

We recommend having at least 320GB of memory to run the model.

Then, run:

```
pip install -r requirements.txt # Or requirements-gpu.txt to use flash attention on GPU(s)
huggingface-cli login           # Add your Hugging Face token in order to access the model
python generate.py              # See generate.py to change the prompt and other settings
```

For more advanced usage, please see LLM Foundry ([chat script](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/hf_chat.py), [batch generation script](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/hf_generate.py))

If you have any package installation issues, we recommend using our Docker image: [`mosaicml/llm-foundry:2.2.1_cu121_flash2-latest`](https://github.com/mosaicml/llm-foundry?tab=readme-ov-file#mosaicml-docker-images)

## Inference

Both TensorRT-LLM and vLLM can be used to run optimized inference with DBRX. We have tested both libraries on NVIDIA A100 and H100 systems. To run inference with 16-bit precision, a minimum of 4 x 80GB multi-GPU system is required.

### TensorRT-LLM

DBRX support has also been added to TensorRT-LLM library: [Pending PR](https://github.com/NVIDIA/TensorRT-LLM/pull/1363)

After merging, instructions to build and run DBRX TensorRT engines will be found at: [README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/dbrx/README.md)

### vLLM
DBRX support has been added to vLLM: [Pending PR](https://github.com/vllm-project/vllm/pull/3660)

After merging, instructions to run vLLM engine will be found at: [Docs](https://docs.vllm.ai/en/latest/)

## Finetune

An example script to finetune DBRX can be found in our open source library [LLM Foundry](https://www.github.com/mosaicml/llm-foundry)


## Model card

The model cards can be found at:
* [DBRX Base](MODEL_CARD_dbrx_base.md)
* [DBRX Instruct](MODEL_CARD_dbrx_instruct.md)

## Integrations

DBRX is available on the Databricks platform through:
* [Mosaic AI Model Serving](https://docs.databricks.com/machine-learning/foundation-models/supported-models.html#dbrx-instruct)
* [Mosaic AI Playground](https://docs.databricks.com/en/large-language-models/ai-playground.html)

The same tools used to train high quality MoE models such as DBRX are available for Databricks customers. Please reach out to us at https://www.databricks.com/company/contact if you are interested in pre-training, finetuning, or deploying your own DBRX models!

## Issues
For issues with model output, or community discussion, please use the Hugging Face community forum ([instruct](https://huggingface.co/databricks/dbrx-instruct), [base](https://huggingface.co/databricks/dbrx-base))

For issues with LLM Foundry, or any of the underlying training libraries, please open an issue on the relevant GitHub repository.

## License

Our model weights and code are licensed for both researchers and commercial entities. The [Databricks Open Source License](https://www.databricks.com/legal/open-model-license) can be found at [LICENSE](LICENSE), and our Acceptable Use Policy can be found [here](https://www.databricks.com/legal/acceptable-use-policy-open-model).
