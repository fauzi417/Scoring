import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import plot_confusion_matrix
df=pd.read_csv('scoring_dataset.csv')
df.columns

#univariate-categorical
df.BranchID.value_counts(dropna = False)
df.BranchID.value_counts(dropna= False).plot.bar()
(df.BranchID.value_counts(normalize=True, dropna= False).round(4) * 100).plot.bar()
plt.xlabel('BranchID')
plt.ylabel('Percentage (%)')
df.BranchID.unique()

df.PersonalType.value_counts(dropna = False)
df.PersonalType.value_counts(dropna= False).plot.bar()
(df.PersonalType.value_counts(normalize=True, dropna= False).round(4) * 100).plot.bar()
plt.xlabel('PersonalType')
plt.ylabel('Percentage (%)')
df.PersonalType.unique()

df['Occupation '].value_counts(dropna = False)
df['Occupation '].value_counts(dropna= False).plot.bar()
(df['Occupation '].value_counts(normalize=True, dropna= False).round(4) * 100).plot.bar()
plt.xlabel('Occupation')
plt.ylabel('Percentage (%)')
df['Occupation '].unique()

df['brand'].value_counts(dropna = False)
df['brand'].value_counts(dropna= False).plot.bar()
prob=df['brand'].value_counts(normalize=True, dropna= False)
threshold=0.03
mask=prob>threshold
tail_prob=prob.loc[~mask].sum()
prob=prob.loc[mask]
prob['OTHER']=tail_prob
prob.plot.bar()
plt.xlabel('Brand')
plt.ylabel('Percentage (%)')
df['brand'].unique()

df['area_region'].value_counts(dropna = False)
df['area_region'].value_counts(dropna= False).plot.bar()
(df['area_region'].value_counts(normalize=True, dropna= False).round(4) * 100).plot.bar()
plt.xlabel('Area Region')
plt.ylabel('Percentage (%)')
df['area_region'].unique()

#univariate-numerical

df.Tenor_.describe()
df.Tenor_.unique()
df.Tenor_.value_counts(dropna= False)
df.Tenor_.value_counts(normalize=True, dropna= False).round(4) * 100
df.Tenor_.plot.hist()
plt.hist(df.Tenor_,density=True)

df.LTV_.describe()
df.LTV_.unique()
df.LTV_.value_counts(dropna= False)
df.LTV_.value_counts(normalize=True, dropna= False).round(4) * 100
df.LTV_.plot.hist()
plt.hist(df.LTV_,density=True)

df.BICheckingResult.describe()
df.BICheckingResult.unique()
df.BICheckingResult.value_counts(dropna=False)
df.BICheckingResult.value_counts(normalize=True, dropna= False).round(4) * 100
df.BICheckingResult.plot.hist()
plt.hist(df.BICheckingResult,density=True)

df.CustAge.describe()
df.CustAge.unique()
df.CustAge.value_counts(dropna=False)
df.CustAge.value_counts(normalize=True, dropna= False).round(4) * 100
df.CustAge.plot.hist()
plt.hist(df.CustAge,density=True)

#bivariate
#categorical ['BranchID', 'PersonalType', 'Occupation ', 'brand',  'area_region']
#numerical ['Tenor_','LTV_', 'BICheckingResult','CustAge',]
#target ['FLAG_BAD']

ax=sns.catplot(y='area_region', x='CustAge',hue='FLAG_BAD',data=df[df['FLAG_BAD']==1])
plt.show()

sns.barplot(y='FLAG_BAD', x='area_region',data=df)
sns.barplot(y='FLAG_BAD', x='BranchID',data=df)
sns.barplot(y='FLAG_BAD', x='PersonalType',data=df)
sns.barplot(y='FLAG_BAD', x='Occupation ',data=df)
sns.barplot(y='FLAG_BAD', x='brand',data=df)

sns.catplot(x='FLAG_BAD', y='LTV_',data=df)

#multivariate
#heatmap
toSearch = df.loc[:, ['BranchID', 'PersonalType', 'Occupation ', 'brand',  'area_region']]
toSearch = toSearch.agg(LabelEncoder().fit_transform)
toSearch['Tenor_']=df['Tenor_']
toSearch['LTV_']=df['LTV_']
toSearch['BICheckingResult']=df['BICheckingResult']
toSearch['CustAge']=df['CustAge']
sns.heatmap(toSearch.corr(), cmap ="YlGnBu", annot=True)

df['BICheckingResult']=df['BICheckingResult'].fillna(value='-')

#one hot encoding
df2=df.drop('BranchID',axis=1)
df2=df2.drop('PersonalType',axis=1)
df_encoded=pd.get_dummies(df2, columns=[ 'Occupation ', 'brand',  'area_region','BICheckingResult'])
df_encoded.shape
df_encoded.columns

X=df_encoded.drop("FLAG_BAD",axis=1).copy()
X=X.drop('AGREEMENTNO',axis=1)
Y=df_encoded["FLAG_BAD"].copy()

#split train test
X_train, X_test, y_train, y_test= train_test_split(X, Y, random_state=0, stratify=Y)

#LR Full data
lr = LogisticRegression()
lr.fit(X,Y)
train_accuracy=lr.score(X,Y)
train_accuracy*100
plot_confusion_matrix(lr,X,Y)
crossvali=cross_val_score(lr,X,Y,cv=5,scoring='accuracy').mean()
print(crossvali)
coef_table = pd.DataFrame(list(X.columns)).copy()
coef_table.insert(len(coef_table.columns),"Coefs",lr.coef_.transpose())

#LR train test
lr = LogisticRegression()
lr.fit(X_train,y_train)
train_accuracy=lr.score(X_train,y_train)
train_accuracy*100
test_accuracy=lr.score(X_test,y_test)
test_accuracy*100
plot_confusion_matrix(lr,X_train,y_train)
plot_confusion_matrix(lr,X_test,y_test)

#DT
clf_dt=DecisionTreeClassifier(random_state=0)
clf_dt=clf_dt.fit(X_train,y_train)
plot_tree(clf_dt,filled=True,rounded=True,class_names=['good','bad'],feature_names=X.columns)
plot_confusion_matrix(clf_dt,X_train,y_train)
plot_confusion_matrix(clf_dt,X_test,y_test)
clf_dt.score(X_test,y_test)*100



paths=clf_dt.cost_complexity_pruning_path(X_train,y_train)
ccpalphas=paths.ccp_alphas
ccpalphas=ccpalphas[:-1]
clf_dts=[]
for ccpalpha in ccpalphas:
    clf_dt=DecisionTreeClassifier(random_state=0,ccp_alpha=ccpalpha)
    clf_dt.fit(X_train,y_train)
    clf_dts.append(clf_dt)
trainscore=[clf_dt.score(X_train,y_train) for clf_dt in clf_dts]
testscore=[clf_dt.score(X_test,y_test) for clf_dt in clf_dts]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccpalphas, trainscore, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccpalphas, testscore, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()

clf_dt_pruned = DecisionTreeClassifier(random_state=0,
                                       ccp_alpha=0.0001)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)
plot_confusion_matrix(clf_dt_pruned, X_test, y_test)

#XGBoost
clf_xgb=xgb.XGBClassifier(seed=0)
clf_xgb.fit(X_train,y_train)
plot_confusion_matrix(clf_xgb,X_train,y_train)
clf_xgb.score(X_train,y_train)*100
plot_confusion_matrix(clf_xgb,X_test,y_test)
clf_xgb.score(X_test,y_test)*100
crossvali=cross_val_score(clf_xgb,X,Y,cv=5,scoring='accuracy').mean()
print(crossvali)
xgb.plot_tree(clf_xgb,class_names=['good','bad'])

