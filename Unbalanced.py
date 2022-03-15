import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

df=pd.read_csv('C:/Users/PREDATOR/PycharmProjects/Study/MIF/BACKSCORE_ASCORE.csv')
df=df[df["FLAG_DATA"]!="OOT"]
df=df[["branchid","BICheckingResult","tenor","GRPPersonalType","GRPBrandNew2","LTV_Manual","Age","F2","F25","FLAG_BAD","FLAG_DATA"]]
df=df.rename(columns={"F2":"Occupation ","F25":"area_region","branchid":"BranchID","Age":"CustAge","LTV_Manual":"LTV_","tenor":"Tenor_"
                      ,"GRPBrandNew2":"brand","GRPPersonalType":"PersonalType"})
df['BICheckingResult']=df['BICheckingResult'].fillna(value=0)
(df[df["FLAG_DATA"]=="TRAINING"].FLAG_BAD.value_counts(normalize=True).round(4)*100).plot.bar()
df2=df
#df2=df2.drop('BranchID',axis=1)
df2=df2.drop('PersonalType',axis=1)
#df2=df2.drop('Tenor_',axis=1)
#df2=df2.drop('BICheckingResult',axis=1)
#df2=df2.drop('brand',axis=1)
df2=df2.drop('LTV_',axis=1)
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
Y=df_encoded[["FLAG_BAD","FLAG_DATA"]].copy()
X_train,X_test,y_train,y_test=X[X["FLAG_DATA"]=="TRAINING"],X[X["FLAG_DATA"]=="VALIDATION"],Y[Y["FLAG_DATA"]=="TRAINING"],Y[Y["FLAG_DATA"]=="VALIDATION"]
X_train,X_test,y_train,y_test=X_train.drop("FLAG_DATA",axis=1),X_test.drop("FLAG_DATA",axis=1),y_train.drop("FLAG_DATA",axis=1),y_test.drop("FLAG_DATA",axis=1)
y_train.value_counts()

df_train=X_train.copy()
df_train["FLAG_BAD"]=y_train

#undersampling
minorityclasslen=len(df_train[df_train["FLAG_BAD"]==1])
minorityclass=df_train[df_train["FLAG_BAD"]==1].index
majorityclass=df_train[df_train["FLAG_BAD"]==0].index
randommajoriry=np.random.choice(majorityclass,minorityclasslen,replace=False)
indexundersample=np.concatenate([minorityclass,randommajoriry])
df_undersampling=df_train.loc[indexundersample]
X_train=df_undersampling.drop("FLAG_BAD",axis=1).copy()
y_train=df_undersampling["FLAG_BAD"].copy()
y_train.value_counts()

#undersampling cluster based prototype generation
cc=ClusterCentroids(sampling_strategy='majority')
X_train,y_train=cc.fit_resample(X_train,y_train)
y_train.value_counts()


#oversampling blind copy
majorityclasslen=len(df_train[df_train["FLAG_BAD"]==0])
majority=df_train[df_train["FLAG_BAD"]==0]
minorityblindcopy=df_train[df_train["FLAG_BAD"]==1].sample(majorityclasslen,replace=True)
df_blind=pd.concat([majority,minorityblindcopy],axis=0)
X_train=df_blind.drop("FLAG_BAD",axis=1).copy()
y_train=df_blind["FLAG_BAD"].copy()
y_train.value_counts()


#oversampling smote
smote=SMOTE(sampling_strategy='minority',k_neighbors=1) #k_neighbors nya bisa cari yg optimal pake loop
X_train,y_train=smote.fit_resample(X_train,y_train)
y_train.value_counts()


#balanced on sklearn
lr = LogisticRegression(max_iter=5000,class_weight='balanced')
#lr_balanced=cross_validate(lr,X_train,y_train,cv=StratifiedKFold(n_splits=2),scoring='f1') #scoring bisa liat di https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print(lr_balanced['test_score'].mean())
#lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(classification_report(y_test,pred))



#undersampling ensemble kfold
#train data misal goodcust ada 1000 data, badcust ada 100 data
#ambil 100-100 goodcust, goodcust[0:100], goodcust[101:200],....
#train data 1 = [goodcust[0:100],badcust]
#train data 2 = [goodcust[101:200],badcust],.....
#bikin model buat semua train
#voting buat y_predfinal nya
