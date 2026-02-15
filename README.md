# Sports vs Politics — Text Classification

**Course:** Natural Language Understanding (NLU)  
**Assignment:** 1, Problem 4  
**Roll Number:** M25CSA007  

---

## Overview

This project builds a binary text classifier to distinguish **Sports** from **Politics** news articles. Three machine learning models are compared using different feature representations.

## Dataset

We use the [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) dataset (Zhang et al., 2015):
- **Class 1 (World/Politics):** 30,000 training / 1,900 test  
- **Class 2 (Sports):** 30,000 training / 1,900 test  

> **Note:** The CSV files are large (~30MB) and not included in this repo. Download from:  
> https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset  
> Place `train.csv` and `test.csv` in the root directory.

## Models & Results

| Model | Feature | Accuracy |
|:---|:---|:---|
| Multinomial Naive Bayes | Bag of Words | **97.45%** |
| Linear SVM | TF-IDF | **97.55%** |
| Logistic Regression | TF-IDF | **97.61%** |

![Accuracy Comparison](model_comparison.png)

## Files

| File | Description |
|:---|:---|
| `M25CSA007_prob4.py` | Main classification script |
| `model_comparison.png` | Bar chart comparing model accuracies |
| `docs/index.html` | GitHub Pages site |

> **Note:** The detailed report `M25CSA007_prob4_report.md` has been removed from this repository as per requirements, but is available upon request or in the local submission package.

## How to Run

```bash
# Install dependencies
pip install scikit-learn numpy matplotlib

# Download AG News dataset (train.csv, test.csv) and place in root

# Run classifier
python M25CSA007_prob4.py
```

## Output

The script prints accuracy and detailed classification reports (precision, recall, F1) for each model, and saves `model_comparison.png`.

## GitHub Pages

The project page is hosted at:  
**[https://duckquack123.github.io/M25CSA007_NLU_Assignment1_Q4/](https://duckquack123.github.io/M25CSA007_NLU_Assignment1_Q4/)**

---
*M25CSA007 — IIT Jodhpur, Semester 2*
