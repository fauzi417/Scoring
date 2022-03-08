import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
df_asli=pd.read_csv('C:/Users/PREDATOR/PycharmProjects/Study/MIF/BACKSCORE_ASCORE.csv')
df_asli=df_asli[df_asli["FLAG_DATA"]!="OOT"]
df_asli=df_asli[["branchid","BICheckingResult","tenor","GRPPersonalType","GRPBrandNew2","LTV_Manual","Age","F2","F25","FLAG_BAD","FLAG_DATA"]]
df_asli=df_asli.rename(columns={"F2":"Occupation ","F25":"area_region","branchid":"BranchID","Age":"CustAge","LTV_Manual":"LTV_","tenor":"Tenor_"
                      ,"GRPBrandNew2":"brand","GRPPersonalType":"PersonalType"})

#Full untuk test
df_fulltest=df_asli
df_fulltest['BICheckingResult']=df_fulltest['BICheckingResult'].fillna(value=0)
df_fulltest2=df_fulltest.drop('BranchID',axis=1)
df_fulltest2=df_fulltest2.drop('PersonalType',axis=1)
df_fulltest2=df_fulltest2.drop('FLAG_DATA',axis=1)
#df_fulltest2=df_fulltest2.drop('Tenor_',axis=1)
df_encoded_fulltest=pd.get_dummies(df_fulltest2, columns=[ 'Occupation ', 'brand',  'area_region','BICheckingResult'])
X_fulltest=df_encoded_fulltest.drop("FLAG_BAD",axis=1).copy()
Y_fulltest=df_encoded_fulltest["FLAG_BAD"].copy()


#data processing
df=pd.read_csv('C:/Users/PREDATOR/PycharmProjects/Study/MIF/scoring_dataset.csv')
df['BICheckingResult']=df['BICheckingResult'].fillna(value=0)
df2=df.drop('BranchID',axis=1)
df2=df2.drop('PersonalType',axis=1)
df2=df2.drop('AGREEMENTNO',axis=1)
#df2=df2.drop('FLAG_DATA',axis=1)
#df2=df2.drop('Tenor_',axis=1)
df_encoded=pd.get_dummies(df2, columns=[ 'Occupation ', 'brand',  'area_region','BICheckingResult'])
X=df_encoded.drop("FLAG_BAD",axis=1).copy()
Y=df_encoded["FLAG_BAD"].copy()
X_train, X_test, y_train, y_test= train_test_split(X, Y, random_state=0, stratify=Y,test_size=0.25)
df_train=X_train
df_train["FLAG_BAD"]=y_train

####
toSearch = df.loc[:, ['BranchID', 'PersonalType', 'Occupation ', 'brand',  'area_region']]
toSearch = toSearch.agg(LabelEncoder().fit_transform)
toSearch['Tenor_']=df['Tenor_']
toSearch['LTV_']=df['LTV_']
toSearch['BICheckingResult']=df['BICheckingResult']
toSearch['CustAge']=df['CustAge']
toSearch['FLAG_BAD']=df['FLAG_BAD']
sns.heatmap(toSearch.corr(), cmap ="YlGnBu", annot=True)
sns.pairplot(toSearch,hue="FLAG_BAD")


#undersampling
minorityclasslen=len(df_train[df_train["FLAG_BAD"]==1])
minorityclass=df_train[df_train["FLAG_BAD"]==1].index
majorityclass=df_train[df_train["FLAG_BAD"]==0].index
randommajoriry=np.random.choice(majorityclass,minorityclasslen,replace=False)
indexundersample=np.concatenate([minorityclass,randommajoriry])
df_undersampling=df_train.loc[indexundersample]
X_train=df_undersampling.drop("FLAG_BAD",axis=1).copy()
y_train=df_undersampling["FLAG_BAD"].copy()


#oversampling blind copy
majorityclasslen=len(df_train[df_train["FLAG_BAD"]==0])
majority=df_train[df_train["FLAG_BAD"]==0]
minorityblindcopy=df_train[df_train["FLAG_BAD"]==1].sample(majorityclasslen,replace=True)
df_blind=pd.concat([majority,minorityblindcopy],axis=0)
X_train=df_blind.drop("FLAG_BAD",axis=1).copy()
y_train=df_blind["FLAG_BAD"].copy()


#oversampling smote
smote=SMOTE(sampling_strategy='minority',k_neighbors=3)
X_train,y_train=smote.fit_resample(X_train,y_train)

#undersampling ensemble


#cek
y_train.value_counts()
df_train["FLAG_BAD"].value_counts()
(df.FLAG_BAD.value_counts(normalize=True).round(4)*100).plot.bar()

#xgboost
clf_xgb=xgb.XGBClassifier(seed=0)
clf_xgb.fit(X_train,y_train)
plot_confusion_matrix(clf_xgb,X_train,y_train)
plot_confusion_matrix(clf_xgb,X_test,y_test)
print(clf_xgb.score(X_test,y_test)*100,clf_xgb.score(X_train,y_train)*100)
pred_XGB_train=clf_xgb.predict(X_train)
print(classification_report(y_train,pred_XGB_train))
pred_XGB_test=clf_xgb.predict(X_test)
print(classification_report(y_test,pred_XGB_test))


#test full data
plot_confusion_matrix(clf_xgb,X_fulltest,Y_fulltest)
clf_xgb.score(X_fulltest,Y_fulltest)*100
pred_XGB_test_fulltest=clf_xgb.predict(X_fulltest)
print(classification_report(Y_fulltest,pred_XGB_test_fulltest))





df=pd.read_csv('C:/Users/PREDATOR/PycharmProjects/Study/MIF/scoring_dataset.csv')
df['BICheckingResult']=df['BICheckingResult'].fillna(value=0)
df2=df
df2=df2.drop('AGREEMENTNO',axis=1)
#df2=df2.drop('BranchID',axis=1)
df2=df2.drop('PersonalType',axis=1)
df2=df2.drop('Tenor_',axis=1)
#df2=df2.drop('BICheckingResult',axis=1)
#df2=df2.drop('brand',axis=1)
#df2=df2.drop('LTV_',axis=1)
#df2=df2.drop('Occupation ',axis=1)
df2=df2.drop('CustAge',axis=1)
df2=df2.drop('area_region',axis=1)
df_encoded=pd.get_dummies(df2, columns=[
    'BranchID',
    #'PersonalType',
    'Occupation ',
    'brand',
    #'area_region',
    'BICheckingResult'
])
X=df_encoded.drop("FLAG_BAD",axis=1).copy()
Y=df_encoded["FLAG_BAD"].copy()
X_train, X_test, y_train, y_test= train_test_split(X, Y, random_state=32, stratify=Y,test_size=0.25)
k=20
precc=np.zeros((k-1))
recc=np.zeros((k-1))
f1cc=np.zeros((k-1))
k_ke=np.array(*[range(1,k)])
for i in range(1,k):
    smote=SMOTE(sampling_strategy='minority',k_neighbors=i,random_state=32)
    X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)
    clf_xgb=xgb.XGBClassifier(seed=32,use_label_encoder=False)
    clf_xgb.fit(X_train_smote,y_train_smote)
    pred=clf_xgb.predict(X_test)
    precc[i-1]=precision_score(y_test,pred)
    recc[i-1]=recall_score(y_test,pred)
    f1cc[i-1]=f1_score(y_test,pred)
summar=pd.DataFrame({'k_ke':k_ke,'precc':precc,'recc':recc,'f1':f1cc})
summar
plot_confusion_matrix(clf_xgb,X_train,y_train)
plot_confusion_matrix(clf_xgb,X_test,y_test)
