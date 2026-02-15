import csv
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_20newsgroups
# data = fetch_20newsgroups(subset='all', categories=['talk.politics.misc','rec.sport.hockey'])
# xtr = data.data
# ytr = data.target
# kept getting timeout error so switched to AG News csv

# tr_f = "bbc_data/bbc/politics"
# te_f = "bbc_data/bbc/sport"
# scores were all 1.0 so it was overfitting or too easy

# for fname in os.listdir("bbc_data/bbc/politics"):
#     with open("bbc_data/bbc/politics/" + fname) as f:
#         xtr.append(f.read())
#         ytr.append(0)

tr_f = "train.csv"
te_f = "test.csv"

if not os.path.exists(tr_f):           
    print("train.csv not found")
    sys.exit()

# xtr, ytr = [], []
# with open(tr_f) as f:   # forgot encoding='utf-8' here, got unicode error
#     for line in f:
#         parts = line.split(",")   # this didnt work because some text had commas in it
#         c = int(parts[0])

# import pandas as pd
# df = pd.read_csv(tr_f, header=None, names=['class','title','desc'])
# df = df[df['class'].isin([1,2])]   # filter politics and sports
# this worked but wanted to keep it simple with csv module

def load_ag_news(path):
    x, y = [], []
    with open(path, 'r', encoding='utf-8') as f:  
        r = csv.reader(f)
        for row in r:
            if len(row) < 3: continue
            c = int(row[0])
            # combine title and description
            # txt = row[1]
            txt = row[1] + " " + row[2]  
            
            if c == 1:
                x.append(txt)
                y.append(0)
            elif c == 2: 
                x.append(txt)
                y.append(1)
            # elif c == 3:
            #     x.append(txt)
            #     y.append(2)   # business
    return np.array(x), np.array(y)

xtr, ytr = load_ag_news(tr_f)
xte, yte = load_ag_news(te_f)

print("Train:", len(xtr))
print("Test:", len(xte))

tn = ['Politics', 'Sports']

# vec = CountVectorizer()
# xtr_vec = vec.fit_transform(xtr)
# xte_vec = vec.transform(xte)
# clf = MultinomialNB()
# clf.fit(xtr_vec, ytr)
# p1 = clf.predict(xte_vec)   

print("\nNB (BoW):")
m1 = make_pipeline(CountVectorizer(), MultinomialNB())  
m1.fit(xtr, ytr)
p1 = m1.predict(xte)
a1 = accuracy_score(yte, p1)
print(f"Acc: {a1:.4f}")
print(classification_report(yte, p1, target_names=tn))

print("\nSVM (TFIDF):")
m2 = make_pipeline(TfidfVectorizer(), LinearSVC(dual='auto'))
m2.fit(xtr, ytr)
p2 = m2.predict(xte)
a2 = accuracy_score(yte, p2)
print(f"Acc: {a2:.4f}")
print(classification_report(yte, p2, target_names=tn))

print("\nLR (TFIDF):")
# m3 = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=100)) #convergence problem  i odnt know why
m3 = make_pipeline(TfidfVectorizer(), LogisticRegression())
m3.fit(xtr, ytr)
p3 = m3.predict(xte)
a3 = accuracy_score(yte, p3)
print(f"Acc: {a3:.4f}")
print(classification_report(yte, p3, target_names=tn))

accs = [a1, a2, a3]
names = ['NB', 'SVM', 'LR']
plt.figure(figsize=(6, 4))
plt.bar(names, accs, color=['blue', 'green', 'red'])
plt.ylim(0.9, 1.0)
plt.title('Accuracy Comparison')
plt.savefig('model_comparison.png')
