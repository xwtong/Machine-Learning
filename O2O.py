# -*- coding: utf-8  -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn import svm
train=pd.read_csv('./data/train.csv',sep='\t')
test=pd.read_csv('./data/test_new.csv')

import re
def extractChinese(s):
    pattern="[\u4e00-\u9fa5]+"#中文正则表达式
    regex = re.compile(pattern) #生成正则对象
    results = regex.findall(s) #匹配
    return "". join(results)
# 预处理数据
label = train['label']
train_data = []
for i in range(len(train['comment'])):
    train_data.append(' '.join(extractChinese(train['comment'][i])))
test_data = []
for i in range(len(test['comment'])):
    test_data.append(' '.join(extractChinese(test['comment'][i])))

tfidf = TFIDF(min_df=1,  # 最小支持长度
              max_features=150000,# 取特征数量
              strip_accents = 'unicode',
              analyzer = 'word',
              token_pattern = r'\w{1,}',
              ngram_range = (1, 3),
              use_idf = 1,
              smooth_idf = 1,
              sublinear_tf = 1,
              stop_words = None,)

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)
# print(data_all)
tfidf.fit(data_all)

data_all = tfidf.transform(data_all)

# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print(train_x.shape)
print(test_x.shape)
print('TF-IDF处理结束.')

from sklearn.model_selection import train_test_split
from classification_utilities import display_cm
import joblib
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(train_x,label,
                                                    test_size=0.3,
                                                    random_state=42)
print('支持向量机....')
clf = svm.SVC(C=100,probability = True)
clf.fit(X_train,y_train)
print('混淆矩阵')
cv_conf = confusion_matrix(y_test, clf.predict(X_test))
labels = ['0','1']
display_cm(cv_conf, labels, display_metrics=True, hide_zeros=False)


clf1 = svm.SVC(C=100,probability = True)
clf1.fit(train_x,label)
svm_pre=clf1.predict(test_x)
svm = pd.DataFrame(data=svm_pre, columns=['comment'])
svm['id'] = test.id
svm = svm[['id', 'comment']]
svm.to_csv('svm11.csv',index=False)
print ("结束！")
