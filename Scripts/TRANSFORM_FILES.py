#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('notebook', '"D:/GITWorkspace/PCA/Scripts/TRANSFORM_FILES.ipynb"')


# ## REPLACE NULL IPPS BY np.NaN

# In[49]:


import pandas as pd
import numpy as np

Path="D:\\StageMai2019\\Project\\data\\In\\"

file_name1 = Path+'FullData.xlsx'
file_name2=Path+'AdditionalData.xlsx'
sheet_name='Sheet1'

def Replace_Null_Ipp(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name)
    df['IPP'] = df['IPP'].astype('str') 
    df['IPP']= df['IPP'].replace(['pasIPP','nan'],np.NaN)
    return df

df1=Replace_Null_Ipp(file_name1,sheet_name)
df2=Replace_Null_Ipp(file_name2,sheet_name)


# ## ENCODE PATIENTS USING IPPS AND MD5 HASH

# In[50]:


import hashlib


def Encode_Patients(df):
    for index, row in df.iterrows():
        if row['IPP'] is not None:
             df.loc[index,'IPP_HASH']=hashlib.md5(str(row['IPP']).encode("utf-8")).hexdigest()
        else:
             df.loc[index,'IPP_HASH']=hashlib.md5(str(row['NAME']+row['SURNAME']+row['DOB']).encode("utf-8")).hexdigest()


    df=df.drop(['PATIENT','REF_NAME','REF_ID770S4V3638','REF_SURNAME','NAME','SURNAME','NO','IPP','Column1'],1)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols] 
    return df


df1=Encode_Patients(df1)
df2=Encode_Patients(df2)

df1.to_excel(Path+'FullData_Hashed.xlsx',encoding='utf-8-sig',index=False)
df2.to_excel(Path+'AdditionalData_Hashed.xlsx',encoding='utf-8-sig',index=False)


# ## TRANSFORM STRUCTURE OF FullData_Hashed.xlsx

# In[52]:


import pandas as pd

file_name='FullData_Hashed.xlsx'

df_A=pd.DataFrame()
df_B=pd.DataFrame()
df_C=pd.DataFrame()

#create lists to hold headers & other variables
HEADERS = []
DATES_EXAM = []
HEIGHTS=[]
WEIGHTS_REF=[]
WEIGHTS=[]
BMIS=[]
ALS=[]
ALS_PARO=[]
ALS_SALI =[]
ALS_DEGL=[] 
ALS_ERCI=[] 
ALS_SGAS=[] 
ALS_AGAS=[] 
ALS_HABI=[] 
ALS_LITD=[]
ALS_MARC=[] 
ALS_ESCA =[]
ALS_DYSPNE=[] 
ALS_ORTHOPNE=[] 
ALS_INSR=[]
DATES_RILUZ=[]
DATES_PREVENT=[]
CVF_ASSIS_THEO=[]
CVL_ASSIS_THEO=[]
All_cols=[]
#group variables

info=['IPP_HASH','SEX','DOB','DIAGPROBA','DATEDIAG','FIRSTSYMPTOM','LIEUDEB','AGE_DEBUT']
clinical_measures_1=['DATEXAM','HEIGHT','WEIGHT_REF','WEIGHT','BMI','ALS','ALS_PARO','ALS_SALI','ALS_DEGL','ALS_ERCI','ALS_SGAS','ALS_AGAS','ALS_HABI','ALS_LITD','ALS_MARC','ALS_ESCA','ALS_ALS_dyspne','ALS_ALS_orthopne','ALS_INSR']
clinical_measures_2=['DATE_PREVENT_PP','CVF_ASSIS_THEO_PP','CVL_ASSIS_THEO_PP']
clinical_measures_riluz_1=['DATDRILU_L1']



#Read CSV File
df = pd.read_excel(file_name, sheet_name='Sheet1')

#create a list of all the columns
columns = list(df)

#split columns list into headers and other variables
for col in columns:
    if col.startswith('DATEXAM'):
        DATES_EXAM.append(col)
    elif  col.startswith('HEIGHT'):
        HEIGHTS.append(col)
    elif col.startswith('WEIGHT_REF') :
        WEIGHTS_REF.append(col)
    elif  col.startswith('WEIGHT') :
        WEIGHTS.append(col)
    elif   col.startswith('BMI'): 
        BMIS.append(col)
    elif  col.startswith('DATDRILU'):
        DATES_RILUZ.append(col)
    elif col.startswith('DATE_PREVENT'):
        DATES_PREVENT.append(col)
    elif col.startswith('CVF_ASSIS_THEO'):
        CVF_ASSIS_THEO.append(col)
    elif col.startswith('CVL_ASSIS_THEO'):
        CVL_ASSIS_THEO.append(col)
    elif  col.startswith('ALS'):
        ##
        if col.__contains__('PARO'):
            ALS_PARO.append(col)
        elif col.__contains__('SALI'):
            ALS_SALI.append(col)
        elif  col.__contains__('DEGL'):
            ALS_DEGL.append(col)
        elif  col.__contains__('ERCI'):
            ALS_ERCI.append(col)
        elif  col.__contains__('SGAS') :
            ALS_SGAS.append(col)
        elif  col.__contains__('AGAS'):
            ALS_AGAS.append(col)
        elif col.__contains__('HABI') :
            ALS_HABI.append(col)
        elif col.__contains__('LITD') :
            ALS_LITD.append(col)
        elif  col.__contains__('MARC') :
            ALS_MARC.append(col)
        elif  col.__contains__('ESCA'):
            ALS_ESCA.append(col)
        elif  col.__contains__('dyspne'):
            ALS_DYSPNE.append(col)
        elif col.__contains__('orthopne'):
            ALS_ORTHOPNE.append(col)
        elif col.__contains__('INSR'):
            ALS_INSR.append(col)
        else :
            ALS.append(col)
    else:
        HEADERS.append(col)

#For headers take into account only info 

HEADERS=list( x for x in info )

#group column variables
All_cols=[]
All_cols.append(DATES_EXAM)
All_cols.append(HEIGHTS)
All_cols.append(WEIGHTS_REF)
All_cols.append(WEIGHTS)
All_cols.append(BMIS)
All_cols.append(ALS)
All_cols.append(ALS_PARO)
All_cols.append(ALS_SALI)
All_cols.append(ALS_DEGL)
All_cols.append(ALS_ERCI)
All_cols.append(ALS_SGAS)
All_cols.append(ALS_AGAS)
All_cols.append(ALS_HABI)
All_cols.append(ALS_LITD)
All_cols.append(ALS_MARC)
All_cols.append(ALS_ESCA)
All_cols.append(ALS_DYSPNE)
All_cols.append(ALS_ORTHOPNE)
All_cols.append(ALS_INSR)
All_cols.append(DATES_RILUZ)
All_cols.append(DATES_PREVENT)
All_cols.append(CVF_ASSIS_THEO)
All_cols.append(CVL_ASSIS_THEO)

#remove empty lists from All_cols list if exist 
All_cols = [x for x in All_cols if x != []]

#Create a final DF with modified columns
for lst in All_cols:
    df_x = pd.melt(df,
                  id_vars=HEADERS,
                  value_vars=lst,
                  var_name=lst[0],
                  value_name=lst[0]+'_VALUE')
         #Concatenate DataFrames 1
    if any(elem in lst for elem in clinical_measures_1):
        df_A= pd.concat([df_A, df_x],axis=1)
    if any(elem in lst for elem in clinical_measures_riluz_1):
        #Concatenate DataFrames 2
        df_B= pd.concat([df_B, df_x],axis=1)
    if any(elem in lst for elem in clinical_measures_2):
        df_C= pd.concat([df_C, df_x],axis=1)

#Delete duplicate columns
df_A= df_A.loc[:, ~df_A.columns.duplicated()]
df_B= df_B.loc[:, ~df_B.columns.duplicated()]
df_C= df_C.loc[:, ~df_C.columns.duplicated()]

#Transform columns of Als items
list_to_replace=['ALS_PARO','ALS_SALI','ALS_DEGL','ALS_ERCI','ALS_SGAS','ALS_AGAS','ALS_HABI','ALS_LITD','ALS_MARC','ALS_ESCA','ALS_ALS_dyspne','ALS_ALS_orthopne','ALS_INSR']
list_replaced_by=['ALS_PARO','ALS_SALI','ALS_DEGL','ALS_ERCI','ALS_SGAS','ALS_AGAS','ALS_HABI','ALS_LITD','ALS_MARC','ALS_ESCA','ALS_DYSPNE','ALS_ORTHOPNE','ALS_INSR']

for t,b in zip(list_to_replace,list_replaced_by): 
    df_A[t] = df_A[t].str.lower()
    df_A.loc[df_A[t].str.contains(b.split('_')[1].lower()), t]=b 


#Write Dataframes to csv
df_A.to_csv(Path+"df_clinical_measures_1.csv",index=False,encoding='utf-8-sig')
df_B.to_csv(Path+"df_clinical_measures_1_Date_Riluzole.csv",index=False,encoding='utf-8-sig')
df_C.to_csv(Path+"df_clinical_measures_2.csv",index=False,encoding='utf-8-sig')


# ## TRANSFORM STRUCTURE OF AdditionalData_Hashed.xlsx

# In[13]:


file_name=Path+'AdditionalData_Hashed.xlsx'

df_D=pd.DataFrame()

#create lists to hold headers & other variables
HEADERS = []
DATES_DECES=[]
All_cols=[]

#group variables
info=['IPP_HASH','SEX','DOB','DIAGPROBA','DATEDIAG','FIRSTSYMPTOM','LIEUDEB','AGE_DEBUT']
clinical_measures_deces=['DATEDCD']


#Read CSV File
df = pd.read_excel(file_name, sheet_name='Sheet1')

#create a list of all the columns
columns = list(df)

#split columns list into headers and other variables
for col in columns:
    if col.startswith('DATEDCD'):
        DATES_DECES.append(col)
    else:
        HEADERS.append(col)

#For headers take into account only info 

HEADERS=list( x for x in info )

#group column variables
All_cols=[]
All_cols.append(DATES_DECES)

#remove empty lists from All_cols list if exist 
All_cols = [x for x in All_cols if x != []]

#Create a final DF with modified columns
for lst in All_cols:
    df_x = pd.melt(df,
                  id_vars=HEADERS,
                  value_vars=lst,
                  var_name=lst[0],
                  value_name=lst[0]+'_VALUE')
         #Concatenate DataFrames
    if any(elem in lst for elem in clinical_measures_deces):
        df_D=pd.concat([df_D,df_x],axis=1)
        
#Delete duplicate columns
df_D=df_D.loc[:, ~df_D.columns.duplicated()]

#Write Dataframes to csv
df_D.to_csv(Path+"df_clinical_measures_1_Date_Deces.csv",index=False,encoding='utf-8-sig')


# In[ ]:




