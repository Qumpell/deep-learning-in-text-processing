Jasne! Poniżej masz kompletny przykład pliku **README.md** do Twojego projektu, przygotowany tak, żeby każdy mógł go łatwo uruchomić lokalnie:

---

# Emotion Classification in Text

## Project Overview

This project performs **emotion classification** on short English sentences. Three different approaches are implemented and compared:

1. **TF-IDF + Logistic Regression** – baseline classical NLP method.
2. **LSTM** – recurrent neural network with learned embeddings.
3. **DistilBERT** – pretrained Transformer-based model fine-tuned on the dataset.

The goal is to classify sentences into six emotions: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.

---

## Requirements

* Python 3.8+
* PyTorch 2.x
* HuggingFace Transformers
* Datasets library
* scikit-learn
* matplotlib, seaborn
* tqdm
* NumPy

Install dependencies via pip:

```bash
pip install -r requirements.txt
```
or

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn tqdm numpy
```


---

## Dataset

The project uses the **Emotion dataset** from HuggingFace:

* URL: [https://huggingface.co/datasets/emotion](https://huggingface.co/datasets/emotion)
* ~20,000 samples
* Split: 80% train, 10% validation, 10% test

---


