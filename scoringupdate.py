import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv('C:/Users/PREDATOR/PycharmProjects/Study/MIF/scoring_dataset.csv')
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

sns.catplot(x='FLAG_BAD', y='Tenor_',data=df)

#multivariate
#heatmap
toSearch = df.loc[:, ['BranchID', 'PersonalType', 'Occupation ', 'brand',  'area_region']]
toSearch = toSearch.agg(LabelEncoder().fit_transform)
toSearch['Tenor_']=df['Tenor_']
toSearch['LTV_']=df['LTV_']
toSearch['BICheckingResult']=df['BICheckingResult']
toSearch['CustAge']=df['CustAge']
toSearch['FLAG_BAD']=df['FLAG_BAD']
sns.heatmap(toSearch.corr(), cmap ="YlGnBu", annot=True)

df['BICheckingResult']=df['BICheckingResult'].fillna(value='-')

#one hot encoding
df2=df.drop('BranchID',axis=1)
df2=df2.drop('PersonalType',axis=1)
#df2=df2.drop('Tenor_',axis=1)
df_encoded=pd.get_dummies(df2, columns=[ 'Occupation ', 'brand',  'area_region','BICheckingResult'])
df_encoded.shape
df_encoded.columns

X=df_encoded.drop("FLAG_BAD",axis=1).copy()
X=X.drop('AGREEMENTNO',axis=1)
Y=df_encoded["FLAG_BAD"].copy()

#split train test
X_train, X_test, y_train, y_test= train_test_split(X, Y, random_state=0, stratify=Y,test_size=0.25)

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
lr = LogisticRegression(solver='saga',max_iter=4000)
lr.fit(X_train,y_train)
train_accuracy=lr.score(X_train,y_train)
train_accuracy*100
test_accuracy=lr.score(X_test,y_test)
test_accuracy*100
pred_lr=lr.predict(X_test)
prob_lr=lr.predict_proba(X_test)[:,1]
roc_auc_score(y_test,prob_lr)
fpr_lr,tpr_lr,_lr=roc_curve(y_test,prob_lr)
plt.plot(fpr_lr,tpr_lr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plot_confusion_matrix(lr,X_train,y_train)
plot_confusion_matrix(lr,X_test,y_test)
print(classification_report(y_test,pred_lr))

#KNN
k=50
mean_acc_train=np.zeros((k-1))
mean_acc_test=np.zeros((k-1))
std_acc=np.zeros((k-1))
for i in range(1,k):
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    r2train=neigh.score(X_train,y_train)
    r2test=neigh.score(X_test,y_test)
    pred=neigh.predict(X_test)
    mean_acc_train[i-1]=r2train
    mean_acc_test[i-1]=r2test
    std_acc[i-1]=np.std(pred==y_test)/np.sqrt(pred.shape[0])
k_ke=np.array(*[range(1,50)])
mean_acc_train
mean_acc_test
std_acc
pd.DataFrame({'k_ke':k_ke,'train':mean_acc_train,'test':mean_acc_test,'std':std_acc})

plt.plot(range(2,21),mean_acc_test[1:20],'b')
plt.fill_between(range(2,21),mean_acc_test[1:20] - 1 * std_acc[1:20], mean_acc_test[1:20] + 1 * std_acc[1:20],alpha=0.10)
plt.fill_between(range(2,21),mean_acc_test[1:20] - 3 * std_acc[1:20], mean_acc_test[1:20] + 3 * std_acc[1:20],alpha=0.10)
plt.plot(range(2,k),mean_acc_train[1:],'g')
plt.fill_between(range(2,k),mean_acc_train[1:] - 1 * std_acc[1:], mean_acc_train[1:] + 1 * std_acc[1:],alpha=0.10)
plt.fill_between(range(2,k),mean_acc_train[1:] - 3 * std_acc[1:], mean_acc_train[1:] + 3 * std_acc[1:],alpha=0.10)

neigh=KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train,y_train)
pred=neigh.predict(X_test)
prob=neigh.predict_proba(X_test)[:,1]
roc_auc_score(y_test,prob)
fpr,tpr,_=roc_curve(y_test,prob)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plot_confusion_matrix(neigh,X_train,y_train)
plot_confusion_matrix(neigh,X_test,y_test)
print(classification_report(y_test,pred))

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
prob_dtprn=clf_dt_pruned.predict_proba(X_test)[:,1]
roc_auc_score(y_test,prob_dtprn)
fpr_dtprn,tpr_dtprn,_dtprn=roc_curve(y_test,prob_dtprn)
plt.plot(fpr_dtprn,tpr_dtprn)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#XGBoost
clf_xgb=xgb.XGBClassifier(seed=0)
clf_xgb.fit(X,Y)
plot_confusion_matrix(clf_xgb,X,Y)
clf_xgb.score(X,Y)*100
plot_confusion_matrix(clf_xgb,X_test,y_test)
clf_xgb.score(X_test,y_test)*100
crossvali=cross_val_score(clf_xgb,X,Y,cv=5,scoring='accuracy').mean()
print(crossvali)
xgb.plot_tree(clf_xgb,class_names=['good','bad'])

#adaboost
adabos=AdaBoostClassifier(n_estimators=100,random_state=0)
clf_ada=adabos.fit(X,Y)
plot_confusion_matrix(clf_ada,X,Y)
clf_ada.score(X,Y)*100
crossvali=cross_val_score(clf_ada,X,Y,cv=5,scoring='accuracy').mean()
print(crossvali)

#SVM
clf_svm=svm.SVC(kernel='rbf',probability=True)
clf_svm.fit(X_train,y_train)
yhat=clf_svm.predict(X_test)
plot_confusion_matrix(clf_svm,X_test,y_test)
print(classification_report(y_test,yhat))
pred_svm=clf_svm.predict(X_test)
prob_svm=clf_svm.predict_proba(X_test)[:,1]
roc_auc_score(y_test,prob_svm)
fpr_svm,tpr_svm,_svm=roc_curve(y_test,prob_svm)
plt.plot(fpr_svm,tpr_svm)

#Random Forest
clf_rf=RandomForestClassifier(n_estimators=100,oob_score=True)
clf_rf.fit(X_train,y_train)
clf_rf.score(X_train,y_train)
clf_rf.oob_score_
clf_rf.score(X_test,y_test)
parameters={'max_depth':[1,2,3,4,5],'min_samples_leaf':[1,2,3,4,5],'min_samples_split':[2,3,4,5],'criterion':['gini','entropy']}
grid_obj=GridSearchCV(clf_rf,parameters)
grid_fit=grid_obj.fit(X_train,y_train)
best_rf=grid_fit.best_estimator_
best_rf
best_rf.fit(X_train,y_train)
best_rf.score(X_train,y_train)
best_rf.oob_score_
best_rf.score(X_test,y_test)
