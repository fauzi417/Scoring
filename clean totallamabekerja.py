import pandas as pd
import re

df_check=pd.read_excel('C:/Users/PREDATOR/PycharmProjects/Study/MIF/MarginalUPD.xlsx')
df_check['TotalLamaPernahBekerja']=df_check['TotalLamaPernahBekerja'].fillna(0)

def lamakerja(i):
    i=str(i)
    i=re.sub("[^0-9]", "", i)
    if (len(i)==1 or len(i)==2 or len(i)==3):
        z=i
    elif (i[0]==i[2] and i[1]==i[3]):
        z=i[0:2]
    elif (i[0]==i[3] and i[1]==i[4] and i[2]==i[5]):
        z=i[0:3]
    elif (i=='2015'):
        z='215'
    elif (i=='2016'):
        z='216'
    elif (i=='1994'):
        z='228'
    elif (i=='21552263233'):
        z='1'
    else:
        z=i[0:3]
    return int(z)

df_check['TotalLamaPernahBekerja']=df_check['TotalLamaPernahBekerja'].apply(lambda x: lamakerja(x))
df_check['TotalLamaPernahBekerja']
