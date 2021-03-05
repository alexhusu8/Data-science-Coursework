#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:24:25 2021

@author: alexandersu
"""
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats as stats
import matplotlib.pyplot as plt

#question 1

df=pd.read_excel("MA-Jan-14-2021-city-data.xlsx")
df.shape
newd=df[df["Total Case Counts"]=="<5"]

df1=df[(df["Total Case Counts"]!="<5") &(df["Percent Positivity"]!="*")]
df2=df1[["Total Case Counts","Percent Positivity"]]

df3=df2.astype(float)
len(df3[(df3["Total Case Counts"]>1000) &(df3["Percent Positivity"]<.1)])

q=df3["Total Case Counts"].mean()
q1=df3["Total Case Counts"].std()
print("{} +- {}".format(q,q1))
q2=df3["Percent Positivity"].mean()
q3=df3["Percent Positivity"].std()
print("{} +- {}".format(q2,q3))

#questions 2 and 3

df_1=pd.read_csv("governor_polls.csv")
df_2=df_1[['candidate_party','state',"pct"]]
df_2['candidate_party'].astype('string')
df_3=df_2[df_2['candidate_party']=='REP']

print(df_3.groupby('state').median().sort_values('pct',ascending=False))

a=df_2['pct'][df_2['state']=='New Hampshire']
b=df_2['pct'][df_2['state']=='Maryland']
c=[a,b]

plt.title('Poll ratings of two governers')
plt.ylabel('Ratings')
d=plt.boxplot(c, notch=True)
plt.xticks([1, 2], ['New Hampshire', 'Maryland'])
#plt.yticks(np.arange(45,62.5, step=2.5))

stats.mood(a,b)