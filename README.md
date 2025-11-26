# 🗿 GIANT
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-MultiPathQA-yellow.svg)](https://huggingface.co/datasets/tbuckley/MultiPathQA)
[![arXiv](https://img.shields.io/badge/arXiv-2511.19652-b31b1b.svg)](https://arxiv.org/abs/2511.19652)

**Abstract:** Despite being widely used to support clinical care, general-purpose large multimodal models (LMMs) have generally shown poor or inconclusive performance in medical image interpretation, particularly in pathology, where gigapixel images are used. However, prior studies have used either low-resolution thumbnails or random patches, which likely underestimated model performance. Here, we ask whether LMMs can be adapted to reason coherently and accurately in the evaluation of such images. In this study, we introduce **G**igapixel **I**mage **A**gent for **N**avigating **T**issue (**GIANT**), the first framework that allows LMMs to iteratively navigate whole-slide images (WSIs) like a pathologist. Accompanying GIANT, we release **MultiPathQA**, a new benchmark, which comprises 934 WSI-level questions, encompassing five clinically-relevant tasks ranging from cancer diagnosis to open-ended reasoning. MultiPathQA also includes 128 questions, authored by two professional pathologists, requiring direct slide interpretation. Using MultiPathQA, we show that our simple agentic system substantially outperforms conventional patch- and thumbnail-based baselines, approaching or surpassing the performance of specialized models trained on millions of images. For example, on pathologist-authored questions, GPT-5 with GIANT achieves 62.5% accuracy, outperforming specialist pathology models such as TITAN (43.8%) and SlideChat (37.5%). Our findings reveal the strengths and limitations of current foundation models and ground future development of LMMs for expert reasoning in pathology.

---

## GIANT
Coming soon ⌛

## MultiPathQA

This repository contains a lightweight Python utility for downloading whole-slide images (WSIs) used in the **MultiPathQA** dataset.  
It supports automated downloading from **GTEx**, **TCGA**, and **PANDA**, and organizes the images into per-dataset folders.

## 🚀 Installation

```bash
git clone https://github.com/KianWeihrauch/GIANT.git
cd GIANT

# (Optional) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```


## 📥 Downloading Slides

Run the downloader script:

```bash
python pull_dataset.py \
    --csv /path/to/MultiPathQA.csv \
    --out-root /path/to/save/images \
    --dataset "*"
```

### Arguments

| Argument       | Description |
|----------------|-------------|
| `--csv`        | Path to the MultiPathQA CSV file |
| `--out-root`   | Root directory where images will be downloaded |
| `--dataset`    | Which dataset to download (`tcga`, `gtex`, `panda`, or `"*"` for all) |
| `--chunk-size` | Optional chunk size for downloads (default: 8MB) |

### Example: download only TCGA slides

```bash
python pull_dataset.py --csv MultiPathQA.csv --out-root slides --dataset tcga
```

### Example: download everything

```bash
python pull_dataset.py --csv MultiPathQA.csv --out-root slides --dataset "*"
```

Images will be saved in:

```
out-root/
    tcga/
    gtex/
    panda/
```


## 🧩 Notes

- GTEx and TCGA slides are downloaded via HTTP from public APIs.
- PANDA slides require the **Kaggle API** and a valid `~/.kaggle/kaggle.json`.
- Existing files are **skipped automatically** to allow resuming.

---
