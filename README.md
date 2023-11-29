## Gradient Computation for SqueezeLLM
[SqueezeLLM](https://arxiv.org/pdf/2306.07629.pdf) utilizes the Fisher Information matrix as a sensitivity metric. This repository, which builds on top of Huggingface's transformer library, is designed to calculate the Fisher sensitivity score (gradient square). This score can be employed in the quantization pipeline of our official  [SqueezeLLM library](https://github.com/SqueezeAILab/SqueezeLLM).

### Prerequisite
You will need to have your own Huggingface-compatible LLaMA checkpoint saved at `[MODEL_PATH]`.

Run the following command for setup:
```
conda create -n sqllm-grad python=3.9 -y
conda activate sqllm-grad
pip install -e .
pip install -r requirements.txt
```

### Command
Run the following command:
```
CUDA_VISIBLE_DEVICES=0 python run.py --output_dir [OUTPUT_PATH] --model_name [MODEL_PATH]   # single GPU
CUDA_VISIBLE_DEVICES=0,1 python run.py --output_dir [OUTPUT_PATH] --model_name [MODEL_PATH]   # multi GPU
```

This command performs the following steps

1. Loads the model from `[MODEL_PATH]`.
2. Computes the gradient square using a subset of the C4 training dataset as a calibration set. You can define and use your own calibration dataset.
3. Outputs the gradient square at `[OUTPUT_PATH]`. The output format will be identical to the loaded Huggingface model checkpoint, with the only difference being that the weight values are replaced by the gradient square.

If the model size exceeds the capacity of a single GPU, our framework provides an option to distribute the model across multiple GPUs. 
This is automated by configuring multiple CUDA visible devices. 
To be specific, the model is partitioned into multiple chunks of consecutive layers, and each segment is assigned to an individual GPU device.

You can also use the `--num_examples` argument to change the number of calibration examples. This defaults to 100.

### Troubleshoot
If you are getting the following error, please open `[MODEL_PATH]/tokenizer_config.json` and fix `"tokenizer_class": "LlamaTokenizer"` to `"tokenizer_class": "LLaMATokenizer"`.
We are currently working on a proper fix for this.
```
ValueError: Tokenizer class LlamaTokenizer does not exist or is not currently imported.
```
