#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:52:02 2021

@author: alexandersu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt

#1a.
digits=load_digits(as_frame=True)
target=pd.DataFrame(digits.target)
pca=PCA(n_components=2)
answer=pca.fit_transform(digits.data)
answer=pd.DataFrame(answer)
finalDF=pd.concat([answer,target],axis=1)
finalDF=finalDF.rename(columns={0: "component_1", 1: "component_2"})
sns.scatterplot(data=finalDF, x='component_1', y='component_2', hue='target', palette='deep')

#1b.
from sklearn.manifold import MDS
mds=MDS(n_components=2)
answer2=mds.fit_transform(digits.data)
answer2=pd.DataFrame(answer2)
finalDF2=pd.concat([answer2,target],axis=1)
finalDF2=finalDF2.rename(columns={0: "component_1", 1: "component_2"})
sns.scatterplot(data=finalDF2, x='component_1', y='component_2', hue='target', palette='deep')

#2a.
data=np.loadtxt("pcaData.csv", delimiter=',')
pca2=PCA(.99)
pca3=PCA(n_components=3)
fit=pca2.fit_transform(data)
fit2=pca3.fit_transform(data)
print(pca2.explained_variance_ratio_)
print(np.cumsum(pca3.explained_variance_ratio_))

#2b.
a=data[:,0]
b=data[:,1]
c=data[:,2]
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(a, b, c, color = "green")
plt.show()

#3a.
iris=load_iris(as_frame=True)
np.cov(iris.data.T)
np.corrcoef(iris.data.T)

#3b
FINALdf=pd.concat([iris.data,iris.target], axis=1)
df9=FINALdf[['sepal width (cm)','petal length (cm)','target']]
np.corrcoef(FINALdf['sepal width (cm)'],FINALdf['petal length (cm)'])
sns.scatterplot(data=df9, x='sepal width (cm)', y='petal length (cm)', hue='target', palette='deep')

#3c
x0=FINALdf[['sepal width (cm)','petal length (cm)']][FINALdf['target']==0]
x1=FINALdf[['sepal width (cm)','petal length (cm)']][FINALdf['target']==1]
x2=FINALdf[['sepal width (cm)','petal length (cm)']][FINALdf['target']==2]
print(np.corrcoef(FINALdf['sepal width (cm)'],FINALdf['petal length (cm)']))
print(np.corrcoef(x0['sepal width (cm)'],x0['petal length (cm)']))
np.corrcoef(x1['sepal width (cm)'],x1['petal length (cm)'])
np.corrcoef(x2['sepal width (cm)'],x2['petal length (cm)'])


