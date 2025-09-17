# VerseBench: Benchmark code for Universe-1

<div align="center">
  <a href="https://huggingface.co/zuoweizwzw/Verse-Bench-Models"><img src="https://img.shields.io/static/v1?label=Verse-Bench-Models&message=HuggingFace&color=yellow"></a>
  <a href="https://huggingface.co/datasets/dorni/Verse-Bench"><img src="https://img.shields.io/static/v1?label=Verse-Bench&message=HuggingFace&color=yellow"></a>
  <a href="https://dorniwang.github.io/UniVerse-1"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a> &ensp;
</div>

## Introduction

The benchmark evaluation code for Universe-1.

## Benchmark Datasets Download

```bash
hf download dorni/Verse-Bench --local-dir ./verse_bench
```

## Model Download

| Models           | 🤗 Hugging Face                                                          |
|------------------|--------------------------------------------------------------------------|
| VerseBenchModels | [VerseBenchModels](https://huggingface.co/zuoweizwzw/Verse-Bench-Models) |

download the pretrained evaluation models into ./models

## Model Usage

### 🔧 Dependencies and Installation

```bash
pip install -r requirements.txt
```

We tested the code on Python 3.10, PyTorch 2.6.0, and CUDA 12.4 on Ubuntu 22.04 LTS.

### 🚀 Inference Scripts

```bash
export MODELS_PATH=models
python calculate_metrics.py
```

Optional arguments:

```bash
  --input_dir # Your data for evaluation, the file names must match the names in our Verse-Bench datasets. Each test case must contains both .mp4 and .wav.
                        #(default: ./mini_testset for quick test.)
  --models_path # Path to the models. (default: ./models)

```