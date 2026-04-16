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


# Nothing to see down here xD

Rsync from local machine to l3icalculmaster:
```bash
rsync -avz ~/Desktop/hipe-ocrepair-llm mtran01@l3icalculmaster:/Utilisateurs/mtran01/ --exclude={'.venv/*','.venv'}
```

l3icalculmaster to local machine:
```bash
rsync -avz mtran01@l3icalculmaster:/Utilisateurs/mtran01/hipe-ocrepair-llm/ ~/Desktop/hipe-ocrepair-llm/ --exclude={'.venv/*','.venv','model','model/*','scripts/data_aggregation.py'}
```


Google sheet link for experiment tracking: [Here](https://docs.google.com/spreadsheets/d/1RFEZXg5q-4pkao0pQvJn3HQheoMqu051YjwCMMazEto/edit?usp=sharing)


## Remarquable findings:
```json
{"document_id":"BEHI-1875-09-25-a-p0012_par7","chunk_idx":0,"dataset":"impresso-snippets","ocr_text":"It has fillen under our notice that a well-known democratic friend, who\nhas laboured earnestly and long in behalf of human freedom, has fallen by\nthe wayside. He has a large family, and cannot, without assistance, free\nhimself from th. diffliulties by whieh he is beset. As a matter of duty, a\nfew friends have already rallied round him, but it is beyond their power to\ndo all that is required; they are, therefore, compelled to make an appeal to\nfriends. We are assured that this is a thoroughly genuine and deserving\ncase. All remittances and communications to be addressed to the secretary,\nMt. J. B. Leno, 50, Bookseller's-row, Strand, W.C.","ground_truth":"It has fallen under our notice that a well-known democratic friend, who\nhas laboured earnestly and long in behalf of human freedom, has fallen by\nthe wayside. He has a large family, and cannot, without assistance, free\nhimself from the difficulties by which he is beset. As a matter of duty, a\nfew friends have already rallied round him, but it is beyond their power to\ndo all that is required; they are, therefore, compelled to make an appeal to\nfriends. We are assured that this is a thoroughly genuine and deserving\ncase. All remittances and communications to be addressed to the secretary,\nMr. J. B. Leno, 50, Bookseller's-row, Strand, W.C.","model_output":"It has come under our notice that a well-known democratic friend, who\nhas laboured earnestly and long in behalf of human freedom, has fallen by\nthe wayside. He has a large family, and cannot, without assistance, free\nhimself from the diffliulties by which he is beset. As a matter of duty, a\nfew friends have already rallied round him, but it is beyond their power to\ndo all that is required; they are, therefore, compelled to make an appeal to\nfriends. We are assured that this is a thoroughly genuine and deserving\ncase. All remittances and communications to be addressed to the secretary,\nMt. J. B. Leno, 50, Bookseller's‐row, Strand, W.C."}
```

OCR: It has fillen under our notice

Ground Truth: It has fallen under our notice

Model Output: It has come under our notice

My explanation: The phrase "come under our notice" is more common in English literature than "fallen under our notice." The model decided to rewrite our sentence to make it sound more natural.

```json
{"document_id":"ALST-1835-05-05-a-p0003_par21","chunk_idx":0,"dataset":"impresso-snippets","ocr_text":"THE LATE Mn. HERER.—In addilion to the\nnotice lately given in the English newspapers about the\nfamous bibliophile, It. Lieber, Esq., you will see in the\ns' Dagblad van Gravenhage, of this morning that the state-\nment does by no means appear exaggerated here, as that\ngentleman has been known, during his residence in this\nplace, to spend his time at the Royal Library, daily, from\nseven in the morning till seven in the evening, comparing\nand compiling ancient and rare works, without taking any\nmeals in the course of the twelve hours.—Letter from the\nHague.","ground_truth":"THE LATE MR. HEBER.—In addition to the\nnotice lately given in the English newspapers about the\nfamous bibliophile, R. Heber, Esq., you will see in the\ns' Dagblad van Gravenhage, of this morning that the state¬\nment does by no means appear exaggerated here, as that\ngentleman has been known, during his residence in this\nplace, to spend his time at the Royal Library, daily, from\nseven in the morning till seven in the evening, comparing\nand compiling ancient and rare works, without taking any\nmeals in the course of the twelve hours.—Letter from the\nHague.","model_output":"THE LATE MRS. HERER.—In addilion to the\nnotice lately given in the English newspapers about the\nfamous bibliophile, H. Lieber, Esq., you will see in the\ns' Dagblad van Gravenhage, of this morning that the state¬\nment does by no means appear exaggerated here, as that\ngentleman has been known, during his residence in this\nplace, to spend his time at the Royal Library, daily, from\nseven in the morning till seven in the evening, comparing\nand compiling ancient and rare works, without taking any\nmeals in the course of the twelve hours.—Letter from the\nHague."}

```

OCR: THE LATE Mn. HERER

Ground Truth: THE LATE MR. HEBER

Model Output: THE LATE MRS. HERER

Explanation: `HERER` is not a common English name in its vocabulary. The model relied on statistical guesswork. It saw THE LATE [Title] [Unknown Word], and it guessed MRS. instead of `MR..` It completely ignored the fact that `Mn.` is only one character away from `MR..`


