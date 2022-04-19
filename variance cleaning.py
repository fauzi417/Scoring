cat_cols = ['PersonalType','Gender','MaritalStatus', 'ResidenceCompanyProvince', 'HomeCompanyStatus',
            'Pekerjaan','jabatan','Status_pekerjaan','Jenis_pekerjaan2', 
            'Pendidikan','HomeFixedLine','KantorFixedLine','Wayofpayment','IsSyariah','ProductType','UsedNew',
            'GroupProductId','applicationsource','FirstInstallment','AssetCategory','AssetUsage','MainCoverage', 'GRPNTF',
            'GRPOTR','Bidang_usaha','flagImpact','isoveride', 'ProductOffering', 'GRPDP', 'SektorEkonomi',
            'SupplierType', 'SupplierCategory','InstallmentScheme','AssetType','AssetCategoryId','FlagMerkPopular',
            'MadeIn','isKPM','Jenis_KPM', 'jenispinjaman', 'PaymentMethod','FlagRO','JarakDomisili'
           ,'CompanyCity','ResidenceCompanyCity','ResidenceCompanyZipCode','CompanyZipCode','Nama_Perusahaan','Supplier']
hm = df.loc[:, cat_cols]
oe=OrdinalEncoder()
hm[cat_cols]=oe.fit_transform(hm[cat_cols])
df[cat_cols]=oe.fit_transform(hm[cat_cols])
print(hm.columns)
hm.shape




#categorical delete variabel dengan variance tinggi sekali
vt = VarianceThreshold(threshold=500)
vt.fit(hm)
mask=vt.get_support()
hm2=hm.loc[:, mask]
print(hm2.columns)
hm2.shape


#categorical delete variabel dengan variance rendah sekali
vt = VarianceThreshold(threshold=0.01)
vt.fit(hm)
mask=vt.get_support()
hm3=hm.loc[:, ~mask]
print(hm3.columns)
hm3.shape




df=df[df.columns[~df.columns.isin(hm2)]]
df=df[df.columns[~df.columns.isin(hm3)]]
df.shape
