import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV


df_asli=pd.read_excel('C:/Users/PREDATOR/PycharmProjects/Study/MIF/MarginalUPD.xlsx')
df_asli.columns
df=df_asli.copy()

df = df.drop(['golivedate','ApplicationID','Agreementno','CreditScoringResult','Creditscore','Radius',
              'CreditScoreSchemeID','Score','Grade','ReasonDescription','DefaultStatus', 'Trend',
              'dukcapilbaru', 'cekdukcapil','cekdukcapilpercentage', 'TotalLamaPernahBekerja',''
              'Autoapprovalresult','AssetCategoryName','ContinueCreditApproval','RepossesDate',
              'employmentsinceyear','MOB'], axis = 1)

df["FLAG_BAD"] = np.where((df["ever30plus"] == 'a. Ever 30Plus') | (df["FlagNegCust"] == '1. NegativeCustomer') , 1, 0)
nullcheck=df.isnull().sum()


df['jenispinjaman'] = df['jenispinjaman'].fillna(value = '-')
df['StartingBalance'] = df['StartingBalance'].fillna(value = 0.0)
df['AvgDebit'] = df['AvgDebit'].fillna(value = 0.0)
df['AvgCredit'] = df['AvgCredit'].fillna(value = 0.0)
df['AvgBalance'] = df['AvgBalance'].fillna(value = 0.0)
df['LastBalanceAmt'] = df['LastBalanceAmt'].fillna(value = 0.0)
df['GrowthBalance'] = df['GrowthBalance'].fillna(value = 0.0)
df['JarakDomisili'] = df['JarakDomisili'].fillna(value = 'Kurang atau Sama Dengan 60 KM')
df.dropna(inplace = True)

nullcheck=df.isnull().sum()

data_fin = df.drop(['PersonalType', 'CompanyCity', 'Status_pekerjaan', 'Jenis_pekerjaan2', 'ProductType','UsedNew',
                    'FlagMerkPopular','GRPNTF','NTFIDR','TotalOTR','CompanyZipCode', 'DPPercentage','ever30plus',
                    'dayoverduemax','AssetCategoryId'],
                   axis = 1)

woe_data = data_fin.copy()

woe_columns = ['Gender','MaritalStatus','HomeCompanyStatus','ResidenceCompanyCity','ResidenceCompanyZipCode','Pekerjaan',
               'jabatan','Nama_Perusahaan','Pendidikan','HomeFixedLine','KantorFixedLine','Wayofpayment','IsSyariah',
               'GroupProductId','applicationsource','FirstInstallment','AssetCategory','AssetUsage','MainCoverage',
               'GRPOTR','Bidang_usaha','flagImpact','isoveride', 'ProductOffering','GRPDP','SektorEkonomi','Supplier',
               'SupplierType','SupplierCategory','PaymentMethod','JarakDomisili',
               'InstallmentScheme', 'AssetType','MadeIn','isKPM','Jenis_KPM','jenispinjaman','FlagNegCust','FlagRO']

woe_features = woe_data[woe_columns]

woe_encoder = ce.WOEEncoder(cols = woe_columns)
woe_features = woe_encoder.fit_transform(woe_data[woe_columns],woe_data['FLAG_BAD']).add_suffix('_woe')

woe_data = woe_data.join(woe_features)

data_fin[woe_columns] = woe_features


processed_data = data_fin.copy()

col_names = ['Lama_Menempati_Tahun','Lama_Menempati_Bulan','NumOfDependence','Pendapatan',
             'CAMOSObligorExposure','jmltunggakanpokok',
             'jmltunggakanhari','StartingBalance','AvgDebit','AvgCredit','AvgBalance','LastBalanceAmt','GrowthBalance']
features = data_fin[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

processed_data[col_names] = features

X = processed_data.drop("FLAG_BAD", axis = 1)
y = processed_data['FLAG_BAD']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3, random_state=0, stratify = y)

lr_model = LogisticRegression(fit_intercept = True, solver = 'saga', max_iter = 5000)
lr_model.fit(X_train,y_train)

lr_pred = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

train_acc = lr_model.score(X_train, y_train)
test_acc = lr_model.score(X_test, y_test)
roc_value = roc_auc_score(y_test, lr_probs)

print("train accuracy: {:.2f}%".format(train_acc * 100))
print("test accuracy: {:.2f}%".format(test_acc * 100))
print("roc value : {:.2f}%".format(roc_value*100))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK


def objective (params, n_folds =10, X_train=X_train,y_train=y_train):
    clf=LogisticRegression(**params,verbose=0)
    scores=cross_val_score(clf,X_train,y_train,cv=10,scoring='roc_auc')
    best_scores=max(scores)
    loss=1-best_scores
    return{'loss':loss, 'params':params, 'status':STATUS_OK}

space = {
        'warm_start': hp.choice('warm_start', [True, False]),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'tol': hp.uniform('tol', 0.00001, 0.0001),
        'C': hp.uniform('C', 0.05, 3),
        'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
        'max_iter': hp.choice('max_iter', range(100, 1000)),
        'multi_class': 'auto',
        'class_weight': hp.choice('class_weight',[None,'balanced'])
        }

tpe_algor=tpe.suggest
bayes_trials=Trials()
best=fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=10, trials=bayes_trials)
best

lr_model = LogisticRegression(C= best['C'],
 class_weight= None,
 fit_intercept= True,
 max_iter= 554,
 solver= 'newton-cg',
 tol= 3.275136975472081e-05,
 warm_start= True)
lr_model.fit(X_train,y_train)

lr_pred = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

train_acc = lr_model.score(X_train, y_train)
test_acc = lr_model.score(X_test, y_test)
roc_value = roc_auc_score(y_test, lr_probs)

print("train accuracy: {:.2f}%".format(train_acc * 100))
print("test accuracy: {:.2f}%".format(test_acc * 100))
print("roc value : {:.2f}%".format(roc_value*100))
