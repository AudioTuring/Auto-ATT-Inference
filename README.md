# Auto-ATT 🔊🤖 
*Automatically Evaluating the Human-likeness of TTS Systems via Audio-LLM-Based Score Regression*


> **Auto-ATT** is a model that lora finetuned on [Qwen2-Audio-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) and offers a plug‑and‑play pipeline to **grade “Audio Turing Tests” (ATTs)** at scale, producing objective scores that correlate with human judgements––all without manual listening.


## About Audio Turing Test (ATT)

ATT is an evaluation framework with a standardized human evaluation protocol and an accompanying dataset, aiming to resolve the lack of unified protocols in TTS evaluation and the difficulty in comparing multiple TTS systems. To further support the training and iteration of TTS systems, we utilized additional private evaluation data to train [Auto-ATT](https://huggingface.co/AudioTuring/Auto-ATT) model based on Qwen2-Audio-7B, enabling a model-as-a-judge approach for rapid evaluation of TTS systems on the ATT dataset. The datasets and Auto-ATT model can be cound in [ATT Collection](https://huggingface.co/collections/AudioTuring/audio-turing-test-6826e24d2197bf91fae6d7f5).


## Installation

We recommand using virtual env:
```
conda create -n att python=3.12
```
Then
```bash
conda activate att
```

For torch GPU Version:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  
```
Others:
```bash
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge libsndfile -y
pip install transformers>=4.36.0  
pip install pandas
pip install pydub  
pip install requests
pip install librosa  
pip install tqdm
pip install numpy
pip install openpyxl
pip install loguru
```

## Usage

1. Download [Auto-ATT](https://huggingface.co/AudioTuring/Auto-ATT) weights.
2. We provide [inference code](https://github.com/AudioTuring/Auto-ATT-Inference/blob/main/inference.py) for evaluation.

```bash
python3 inference.py -h
usage: inference.py [-h] [--audio_path AUDIO_PATH] [--audio_dir AUDIO_DIR] --model_path MODEL_PATH [--output_file OUTPUT_FILE] [--force_recompute]

Compute HLS score for an audio file or a directory of audio files.

options:
  -h, --help            show this help message and exit
  --audio_path AUDIO_PATH
                        Path to a single audio file (.mp3 or .wav).
  --audio_dir AUDIO_DIR
                        Path to a directory containing audio files.
  --model_path MODEL_PATH
                        Path to the pre-trained Qwen2Audio model.
  --output_file OUTPUT_FILE
                        Path to save the output JSONL file.
  --force_recompute     Force recompute scores for all files, even if already processed.
```


## Datasets & Benchmarks
See [ATT Collection](https://huggingface.co/collections/AudioTuring/audio-turing-test-6826e24d2197bf91fae6d7f5).



## Citation

```
@software{Auto-ATT,
  author = {Wang, Xihuai and Zhao, Ziyi and Ren, Siyu and Zhang, Shao and Li, Song and Li, Xiaoyu and Wang, Ziwen and Qiu, Lin and Wan, Guanglu and Cao, Xuezhi and Cai, Xunliang and Zhang, Weinan},
  title = {Audio Turing Test: Benchmarking the Human-likeness and Naturalness of Large Language Model-based Text-to-Speech Systems in Chinese},
  year = {2025},
  url = {https://github.com/AudioTuring/Auto-ATT-Inference},
  publisher = {huggingface},
}
```
