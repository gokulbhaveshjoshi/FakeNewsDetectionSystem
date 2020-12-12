# -*- coding: utf-8 -*-
"""
Fake News Detection System
The SVM model

@author: Gokul Bhavesh Joshi
"""

#import libraries
from getEmbeddings import getEmbeddings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os

#Plotting confusion matrix
def plot_cmat(yte, ypred):
    fig, ax = plt.subplots(figsize=(5, 5))
    skplt.plot_confusion_matrix( yte, ypred, ax=ax)
    plt.show()
    
#Read the data
if not os.path.isfile('./xtr.npy') or \
    not os.path.isfile('./xte.npy') or \
    not os.path.isfile('./ytr.npy') or \
    not os.path.isfile('./yte.npy'):
      
    xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
    np.save('./xtr', xtr)
    np.save('./xte', xte)
    np.save('./ytr', ytr)
    np.save('./yte', yte)
    
xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

# Use the built-in SVM for classification
clf = SVC()
clf.fit(xtr, ytr)
y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy ="+ format((m-n)/m*100, '.2f')+ "%")

#Draw the confusion matrix
plot_cmat(yte, y_pred)
