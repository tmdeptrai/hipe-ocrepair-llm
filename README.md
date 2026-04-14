# hipe-ocrepair-llm

# 1. Set up & Install dependencies

For this project, I'll be using `uv` which is a pieton package manager but better than conda, it uses oxidized iron (rust) to speed up packages download lmao.

Install `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync (download) with my dependencies in `pyproject.toml`:

```bash
uv sync
```

Note: This is gonna download `pytorch`, it will take a while and **A LOT OF DISK SPACE**, if you have `pytorch` installed elsewhere, use it instead xD.

Optional: Afterwards, you'll see the folder `/.venv/`, you can activate this virtual env with 
```bash
source .venv/bin/activate
which python #should say ...hipe-ocrepair-llm/.venv/bin/python
```
Or if you dislike it, then you'll need to add `uv run` in front of every commands below. Example: `python src/grab_samples.py` --> `uv run python src/grab_samples.py`

# 2. Download dataset: HIPE-OCRepair-2026-data

Download the dataset first:
```bash
chmod +x ./scripts/*
./download_dataset.sh
```

# 3. Grab samples from dataset

This script will let you see the differences between curated ground truth text and ocr text.

```bash
python src/grab_samples.py
```

# 4. BART Baseline for OCR Post-Correction

This script downloads the BART model named `pykale/bart-base-ocr` to correct OCR errors, see model card [here](http://huggingface.co/pykale/bart-base-ocr)

```bash
python src/bart_base_ocr.py
```



Rsync from local machine to l3icalculmaster:
```bash
rsync -avz ~/Desktop/hipe-ocrepair-llm mtran01@l3icalculmaster:/Utilisateurs/mtran01/ --exclude={'.venv/*','.venv'}
```