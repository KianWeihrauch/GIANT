# 🗿 GIANT
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-MultiPathQA-yellow.svg)](https://huggingface.co/datasets/tbuckley/MultiPathQA)
[![arXiv](https://img.shields.io/badge/arXiv-2511.19652-b31b1b.svg)](https://arxiv.org/abs/2511.19652)

**Abstract:** Despite being widely used to support clinical care, general-purpose large multimodal models (LMMs) have generally shown poor or inconclusive performance in medical image interpretation, particularly in pathology, where gigapixel images are used. However, prior studies have used either low-resolution thumbnails or random patches, which likely underestimated model performance. Here, we ask whether LMMs can be adapted to reason coherently and accurately in the evaluation of such images. In this study, we introduce **G**igapixel **I**mage **A**gent for **N**avigating **T**issue (**GIANT**), the first framework that allows LMMs to iteratively navigate whole-slide images (WSIs) like a pathologist. Accompanying GIANT, we release **MultiPathQA**, a new benchmark, which comprises 934 WSI-level questions, encompassing five clinically-relevant tasks ranging from cancer diagnosis to open-ended reasoning. MultiPathQA also includes 128 questions, authored by two professional pathologists, requiring direct slide interpretation. Using MultiPathQA, we show that our simple agentic system substantially outperforms conventional patch- and thumbnail-based baselines, approaching or surpassing the performance of specialized models trained on millions of images. For example, on pathologist-authored questions, GPT-5 with GIANT achieves 62.5% accuracy, outperforming specialist pathology models such as TITAN (43.8%) and SlideChat (37.5%). Our findings reveal the strengths and limitations of current foundation models and ground future development of LMMs for expert reasoning in pathology.

---

## 🚀 Installation

```bash
git clone https://github.com/KianWeihrauch/GIANT.git
cd GIANT

# (Optional) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 📥 Downloading MultiPathQA Slides

Download whole-slide images from GTEx, TCGA, and PANDA:

```bash
python pull_dataset.py \
    --csv MultiPathQA.csv \
    --out-root /path/to/save/slides \
    --dataset "*"
```

**Arguments:**
- `--csv`: Path to the MultiPathQA CSV file
- `--out-root`: Root directory where images will be downloaded
- `--dataset`: Which dataset to download (`tcga`, `gtex`, `panda`, or `"*"` for all)
- `--chunk-size`: Optional chunk size for downloads (default: 8MB)

**Notes:**
- GTEx and TCGA slides are downloaded via HTTP from public APIs
- PANDA slides require the Kaggle API and a valid `~/.kaggle/kaggle.json`
- Existing files are skipped automatically to allow resuming

## 🧩 Generating Patches for Baselines

Clone [CLAM](https://github.com/mahmoodlab/CLAM) and generate patches:

```bash
# Example: PANDA dataset
python create_patches_fp.py \
    --source /path/to/panda_slides \
    --save_dir /path/to/panda_patches \
    --patch_size 224 \
    --step_size 224 \
    --seg --patch --no_auto_skip \
    --preset clam_presets/panda.csv
```

Presets for different datasets are in the `clam_presets/` folder.

## 🌐 Running the Web App

Launch the interactive web interface to analyze pathology slides:

```bash
python3 app.py
```

Then open http://127.0.0.1:3010 in your browser. Click on a slide to start analyzing it with the AI agent.

**Configuration:**
- Add your OpenAI API key to `config.ini`
- Place `.svs` or other supported slide files in the project directory
- The app uses `gpt-5` by default (configurable in `app.py`)

## 🧪 Running Benchmarks

The benchmark script runs experiments using configuration files:

```bash
python scripts/run_benchmark.py --config scripts/configs/full_benchmark.yaml
```

**Available configurations:**
- `full_benchmark.yaml` - Complete benchmark (agent, patch, thumbnail modes)
- `scaling_experiment.yaml` - Agent scaling across iteration counts
- `thumbnail_resolution_sweep.yaml` - Thumbnail mode at different resolutions
- `random_patch_experiment.yaml` - Random patch agent mode

**Configuration structure:**

```yaml
models: ["claude-sonnet-4-5-20250929"]

benchmarks:
  - name: "gtex"
    dataset: "benchmarks/gtex_organ_type_test.csv"
    filedir: "/path/to/gtex_slides"
    patch_dir: "/path/to/gtex_patches/patches"

modes:
  agent:
    enabled: true
    iterations: [20]
    max_images_in_context: [null]
    enable_note_tool: false
    image_resolutions: [500]
  
  patch:
    enabled: true
    num_patches: 30
  
  thumbnail:
    enabled: true

timeout: 900.0
max_retries: 3
max_workers: 10
limit: null
output_dir: "results/full_benchmark"
log_dir: "logs/full_benchmark"
resume: true
```

Update `filedir` and `patch_dir` paths in the config files to match your local setup.

## 📚 Citation

If you use GIANT or MultiPathQA, please cite our [paper](https://arxiv.org/abs/2511.19652):

```bibtex
@misc{buckley2025giant,
      title={Navigating Gigapixel Pathology Images with Large Multimodal Models}, 
      author={Thomas A. Buckley and Kian R. Weihrauch and Katherine Latham and Andrew Z. Zhou and Padmini A. Manrai and Arjun K. Manrai},
      year={2025},
      eprint={2511.19652},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19652}, 
}
```
