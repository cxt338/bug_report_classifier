import pandas as pd
import numpy as np
import re
import math

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

import os
import subprocess
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'pytorch'
path = f'bug_report_classifier/data/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

datafile = 'Title+Body.csv'

REPEAT = 10

out_csv_name = f'bug_report_classifier/results/{project}_LR.csv'

data = pd.read_csv(datafile).fillna('')
text_col = 'text'

original_data = data.copy()

data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

lr_params = {
    'C': [0.01, 0.1, 1, 10]
}

nb_params = {
    'var_smoothing': np.logspace(-12, 0, 13)
}

nb_f1_scores = []
lr_f1_scores = []

for repeated_time in range(REPEAT):
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)

    nb_clf = GaussianNB()
    nb_grid = GridSearchCV(
        nb_clf,
        nb_params,
        cv=5,
        scoring='f1_macro'
    )
    nb_grid.fit(X_train.toarray(), y_train)

    nb_best_clf = nb_grid.best_estimator_
    nb_best_clf.fit(X_train.toarray(), y_train)

    nb_y_pred = nb_best_clf.predict(X_test.toarray())

    f1 = f1_score(y_test, nb_y_pred, average='macro')
    nb_f1_scores.append(f1)

    lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_grid = GridSearchCV(
        lr_clf,
        lr_params,
        cv=5,
        scoring='f1_macro'
    )
    lr_grid.fit(X_train, y_train)

    lr_best_clf = lr_grid.best_estimator_
    lr_best_clf.fit(X_train, y_train)

    lr_y_pred = lr_best_clf.predict(X_test)

    f1 = f1_score(y_test, lr_y_pred, average='macro')
    lr_f1_scores.append(f1)

from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(nb_f1_scores, lr_f1_scores)

print("=== T-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"NB Mean F1: {np.mean(nb_f1_scores):.4f}")
print(f"LR Mean F1: {np.mean(lr_f1_scores):.4f}")
diff = np.mean(lr_f1_scores) - np.mean(nb_f1_scores)
print(f"Mean Model F1 Difference: {diff:.4f}")