# CT Corona Classifier

A PyTorch deep-learning project that classifies chest CT scans into three categories:

| Label | Meaning |
|-------|---------|
| **nCT** | Normal (no findings) |
| **NiCT** | Non-informative CT |
| **pCT** | COVID-19 positive |

The model used is a custom **TinyVGG** CNN trained from scratch.

---

## Project Structure

```
CT_Corona_Classifier/
├── prepare_dataset.py    # Download & copy dataset from Kaggle
├── data_exploratory.py   # EDA: image samples, class distribution
├── train.py              # Model definition, training loop, evaluation
└── requirements.txt      # Python dependencies
```

> **Note:** The `data/` folder is excluded from the repository (too large).
> Run `prepare_dataset.py` to download it automatically.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/CT_Corona_Classifier.git
cd CT_Corona_Classifier
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** if the CUDA version in `requirements.txt` doesn't match your driver,
> install PyTorch manually from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Configure Kaggle credentials
Place your `kaggle.json` file in `~/.kaggle/` (Linux/macOS) or `C:\Users\<user>\.kaggle\` (Windows).

### 5. Download the dataset
```bash
python prepare_dataset.py
```

---

## Usage

### Explore the data
```bash
python data_exploratory.py
```
Displays a random CT scan sample and a class-distribution pie chart.

### Train the model
```bash
python train.py
```

Key hyperparameters (editable at the top of `train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Mini-batch size |
| `NUM_EPOCHS` | 15 | Training epochs |
| `LEARNING_RATE` | 0.001 | Adam LR |
| `HIDDEN_UNITS` | 10 | Conv channels |
| `USE_CLASS_WEIGHTS` | False | Weighted cross-entropy |

Training saves:
- `tinyvgg_best.pth` — best checkpoint (by val macro F1)
- `training_curves.png` — loss / accuracy / F1 curves
- `confusion_matrix.png` — confusion matrix on the test set

---

## Dataset

**Preprocessed CT Scans for COVID-19**
- Source: [Kaggle – azaemon](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19)
- Split: 70% train / 15% validation / 15% test (stratified)

---

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | — |
| Macro F1 | — |

*(Fill in after training)*

---

## License

This project is released under the [MIT License](LICENSE).
