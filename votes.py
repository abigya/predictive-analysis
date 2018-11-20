#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:17:29 2018

@author: abigyadevkota
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


infilename = "/Users/abigyadevkota/Downloads/pictures-train.tsv"
df = pd.read_csv(infilename, sep=r"\s+")

#piecharts for regions and titles
#group regions by their first letter 
#in order to make the piechart look cleaner
region_char = pd.Series(df.region.str.slice(start=0, stop=1))
region_counts = region_char.value_counts()
highlight = [0] * len(region_counts)
highlight[0] = 0.1
labels = list(region_counts.index)
fig1 = plt.figure(1)
ax = region_counts.plot(kind="pie",explode=highlight,labels=labels,autopct='%.2f',ax=fig1.gca(), 
                        title="Region",legend=True)
ax.set_ylabel('')
plt.savefig("region-pie.png",dpi=200)

#group titles by their first letter
#in order to make the piechart look cleaner
etitles_char = pd.Series(df.etitle.str.slice(start=0, stop=1))
etitles_counts = etitles_char.value_counts()
label= list(etitles_counts.index)
fig2 = plt.figure(2)
ax2 = etitles_counts.plot(kind="pie", ax=fig2.gca(),labels=label,autopct='%.2f',title="Titles",legend=True)
ax2.set_ylabel('')
plt.savefig("titles-pie.png",dpi=200)

#replace negative values with nan in last three columns
df.loc[df.votes < 0] = np.nan
df.loc[df.viewed < 0]=np.nan
df.loc[df.n_comments < 0]=np.nan
#fill nans with 0s
df= df[2:].fillna(0)


#convert all date columns to pandas DateTime
df["votedon"]=pd.to_datetime(df["votedon"],errors = 'coerce')
df['takenon'] = pd.to_datetime(df['takenon'],errors = 'coerce')

#convert the number of votes, comments and views to log
df['votes']=np.log(df['votes']+1)
df['n_comments']=np.log(df['n_comments']+1)
df['viewed']=np.log(df['viewed']+1)


#Line graphs of average number of pictures, upvotes, views, 
#and comments by year; 
#log Y axis and provide a legend.
df_graphs = df.loc[:,:]
df_graphs = df_graphs.drop(["takenon"],axis=1)
df_graphs = df_graphs.drop(["author_id"],axis=1)
df_graphs['votedon'] = pd.to_datetime(df_graphs['votedon']).dt.year
df_graphs = df_graphs.sort_values(by='votedon',ascending= True)
df_avg = df_graphs.groupby(['votedon']).mean()
df_avg["pictures"]=(df_graphs.votedon.value_counts())
df_avg = df_avg.reset_index()
df_avg["pictures"]=df_avg["pictures"]/(len(df_avg.votedon.index))

fig3 = plt.figure(3)
x=df_avg["votedon"]
plt.plot(x, df_avg["votes"],c="g",label="votes")
plt.plot(x, df_avg["n_comments"],c="r",label="comments")
plt.plot(x,df_avg["viewed"],c="b",label="views")
plt.plot(x,df_avg["pictures"],c="y",label="pictures")
plt.legend()
plt.grid()
plt.yscale('log')
plt.savefig("linegraphs.png",dpi=200)

#categorical data are etitle and region
df = pd.get_dummies(df,columns = ['etitle','region'],drop_first=True)
df["votedon_int"]=df["votedon"].astype(int)
df["takenon_int"]=df["takenon"].astype(int)


#feature selection 
q = 0.8 
sel = VarianceThreshold(threshold=(q * (1 - q)))
df_new = df.loc[:,"author_id":]
df_new = df_new.drop(["votes"],axis=1)

X_new= sel.fit_transform(df_new)
mask = sel.get_support()
feature_names = df_new.columns[mask]
features = pd.DataFrame(X_new,columns=feature_names)

#model fitting
X_train, X_test, y_train, y_test = tts(features, df["votes"],test_size=0.30, random_state=42)
rfr = RandomForestRegressor()
model = rfr.fit(X_train, y_train)
training_score = rfr.score(X_train, y_train)
testing_score= rfr.score(X_test, y_test)
predicted_votes = rfr.predict(X_test)

#cross validate the model
scores = cross_val_score(rfr, X_train, y_train, cv = 10)
cross_valscore= np.mean(cross_val_score(rfr, X_train, y_train, cv=10))
print("Cross valscore: ")
print(cross_valscore)

#create df with true and predicted votes from the testing model
true_predict = pd.DataFrame(y_test.copy())
true_predict["predicted"]=predicted_votes.copy()
true_predict["difference"] = true_predict["votes"]-true_predict["predicted"]
fig4 = plt.figure(4)
ax4 = true_predict[true_predict["difference"]>0]["difference"].plot.hist(log=True,
                  bins=50,ax=fig4.gca(),title = "Difference between predicted and actual votes")
plt.grid()
plt.savefig("predict-vs-actual-his.png",dpi=200)


#join x and y training sets for histograms
df_train = X_train.join(y_train)

#histogram of views
fig5= plt.figure(5)
ax5 = df_train[df_train["viewed"]>2]["viewed"].plot.hist(log=True,bins=50,ax=fig5.gca(),title="Views")
plt.grid()
plt.savefig("views-hist.png",dpi=200)

#histogram of upvotes
fig6=plt.figure(6)
ax6=df_train[df_train["votes"]>2]["votes"].plot.hist(log=True,bins=50,ax=fig6.gca(),title ="Votes")
plt.grid()
plt.savefig("votes-hist.png",dpi=200)

#histogram of comments
fig7=plt.figure(7)
ax7=df_train[df_train["n_comments"]>2.5]["n_comments"].plot.hist(log=True,bins=50,
            ax=fig7.gca(),title = "Comments")
plt.grid()
plt.savefig("comments-hist.png",dpi=200)

# save the model to disk
filename = 'model.p'
pickle.dump(rfr, open(filename, 'wb'))
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))