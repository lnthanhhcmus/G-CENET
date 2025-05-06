# G-CENET: Temporal Knowledge Graph Reasoning via Generation and Contrastive Learning

This repository contains the official implementation of the paper:

**"Temporal Knowledge Graph Reasoning Based on Generation Mode and Historical Contrastive Learning (G-CENET)"**.

G-CENET is a novel dual-path model for temporal knowledge graph link prediction that integrates a generation mode with contrastive learning. It introduces a Dynamic Entity Weighting Module (DEWM) and a latent temporal generative module to improve performance across both frequent and rare entities.

## ⚙️ Environment Setup

We recommend Python 3.8+ and PyTorch 1.10 or above.

Install required packages:
```bash
pip install -r requirements.txt
```

---

## 🧪 Data Preprocessing

Each dataset needs to be preprocessed to construct historical snapshots. Run:
```bash
cd data/YAGO
python get_history_graph.py
```
Repeat this for other datasets (ICEWS14, ICEWS18, WIKI) by navigating into each dataset folder.

This script creates intermediate files required for computing historical frequency, contrastive signals, and DEWM vectors.

---

## 🚀 Training and Evaluation

Below are example commands to train and evaluate G-CENET on each dataset.
You must run subject and object prediction separately, and then average results for final scores.

### ICEWS14
```bash
python main.py -d ICEWS14 --max-epochs 500 --valid-epochs 5 --alpha 0.5 --beta 0.4 --gamma 1.0 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir ./save/ICEWS14_subj --eva_dir ./save/ICEWS14_subj --time-stamp 1 --entity subject

python main.py -d ICEWS14 --max-epochs 500 --valid-epochs 5 --alpha 0.5 --beta 0.4 --gamma 1.0 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir ./save/ICEWS14_obj --eva_dir ./save/ICEWS14_obj --time-stamp 1 --entity object
```

### ICEWS18
```bash
python main.py -d ICEWS18 --max-epochs 500 --valid-epochs 5 --alpha 0.5 --beta 0.4 --gamma 0.9 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir ./save/ICEWS18_subj --eva_dir ./save/ICEWS18_subj --time-stamp 24 --entity subject

python main.py -d ICEWS18 --max-epochs 500 --valid-epochs 5 --alpha 0.5 --beta 0.4 --gamma 0.9 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir ./save/ICEWS18_obj --eva_dir ./save/ICEWS18_obj --time-stamp 24 --entity object
```

### WIKI
```bash
python main.py -d WIKI --max-epochs 500 --valid-epochs 5 --alpha 0.2 --beta 0.4 --gamma 0.9 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir ./save/WIKI_subj --eva_dir ./save/WIKI_subj --time-stamp 1 --entity subject

python main.py -d WIKI --max-epochs 500 --valid-epochs 5 --alpha 0.2 --beta 0.4 --gamma 0.9 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir ./save/WIKI_obj --eva_dir ./save/WIKI_obj --time-stamp 1 --entity object
```

### YAGO
```bash
python main.py -d YAGO --max-epochs 500 --valid-epochs 5 --alpha 0.1 --beta 0.4 --gamma 1.0 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir ./save/YAGO_subj --eva_dir ./save/YAGO_subj --time-stamp 1 --entity subject

python main.py -d YAGO --max-epochs 500 --valid-epochs 5 --alpha 0.1 --beta 0.4 --gamma 1.0 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir ./save/YAGO_obj --eva_dir ./save/YAGO_obj --time-stamp 1 --entity object
```

---

## 📊 Hyperparameter Details
- `--alpha`: weight for contrastive + structural loss \( \mathcal{L}_{	ext{CE}} \)
- `--beta`: prediction fusion weight between generative and contrastive paths
- `--gamma`: weight for generative loss \( \mathcal{L}_{g} \)
- `--lambdax`: weight for contrastive frequency signal
- `--time-stamp`: number of time steps in history window
- `--entity`: whether predicting subject or object

Default vector dimension is 200, batch size is 1024, and learning rate is 0.001.

---

## 📈 Output & Evaluation
- Results are stored under the `--save_dir` path.
- Evaluation is automatically triggered if `--eva_dir` is provided.
- Outputs include Hits@1/3/10, MRR, and ranked candidate lists.

---

## 📦 Reproducibility Notes
- All default settings match Table 3 from the paper.
- The code has been tested on a single NVIDIA GPU (RTX 3090 / A100).
- Random seeds are fixed internally for reproducibility.

For further validation, refer to **Section 4.7** for hyperparameter sensitivity curves.

---



