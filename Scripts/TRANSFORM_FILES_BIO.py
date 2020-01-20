#!/usr/bin/env python
# coding: utf-8

# ### anonymized patients 

# In[ ]:


import pandas as pd
import numpy as np
import hashlib

Path="D:\\StageMai2019\\Project\\data\\In\\"

file_name1 = Path+'Extraction.xlsx'
sheet_name='Sheet1'

def Replace_Null_Ipp(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name)
    df['IPP'] = df['IPP'].astype('str') 
    df['IPP']= df['IPP'].replace(['pasIPP','nan'],np.NaN)
    return df

df1=Replace_Null_Ipp(file_name1,sheet_name)



def Encode_Patients(df):
    for index, row in df.iterrows():
        if row['IPP'] is not None:
             df.loc[index,'IPP_HASH']=hashlib.md5(str(row['IPP']).encode("utf-8")).hexdigest()
        else:
             df.loc[index,'IPP_HASH']=hashlib.md5(str(row['Nom']+row['Prénom']+row['Date de naissance']).encode("utf-8")).hexdigest()


    df=df.drop(['IEP','IPP','Nom','Prénom','N° de travail','Correspondant','Column1','Valeur de résultat du Créatinine_29','Unité_30'],1)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols] 
    return df


df1=Encode_Patients(df1)

df1.to_csv(Path+'Bilogical_Hashed.csv',encoding='utf-8-sig',index=False)


# ### Transform Structure

# In[2]:


lst=['Valeur de résultat du Sodium','Unité',
'Valeur de résultat du Potassium','Unité_1',
'Valeur de résultat du Chlorures','Unité_2',
'Valeur de résultat du Albumine','Unité_3',
'Valeur de résultat du Calcium','Unité_4',
'Valeur de résultat du Urée','Unité_5',
'Valeur de résultat du Glucose','Unité_6',
'Valeur de résultat du Créatine kinase','Unité_7',
'Valeur de résultat du Bêta 2 microglobuline','Unité_8',
'Valeur de résultat du Créatinine','Unité_9',
'Valeur de résultat du LDH','Unité_10',
'Valeur de résultat du ASAT','Unité_11',
'Valeur de résultat du ALAT','Unité_12',
'Valeur de résultat du Phosphatases alc.','Unité_13',
'Valeur de résultat du Gamma-GT','Unité_14',
'Valeur de résultat du Préalbumine','Unité_15',
'Valeur de résultat du Bilirubine conjuguée','Unité_16',
'Valeur de résultat du Bilirubine non conjuguée','Unité_17',
'Valeur de résultat du Bilirubine totale','Unité_18',
'Valeur de résultat du HDL Cholestérol (mmol/L)','Unité_19',
'Valeur de résultat du LDL Cholestérol (mmol/L)','Unité_20',
'Valeur de résultat du Fer','Unité_21',
'Valeur de résultat du Transferrine','Unité_22',
'Valeur de résultat du Capacité tot. fixation Trf','Unité_23',
'Valeur de résultat du Coeff Saturation Trf','Unité_24',
'Valeur de résultat du Récept.Soluble Transferrine','Unité_25',
'Valeur de résultat du Ferritine','Unité_26',
'Valeur de résultat du Triglycérides','Unité_27',
'Valeur de résultat du CRP','Unité_28']

df_x = pd.melt(df1,
                  id_vars=['IPP_HASH','Date de prélèvement','Sexe','Date de naissance'],
                  value_vars=lst,
                  var_name='mesure',
                  value_name='mesure_value')


# In[24]:


df_x['mesure'].replace(regex=True,inplace=True,to_replace=r'Valeur de résultat du',value=r'')
df_x['mesure'].replace(regex=True,inplace=True,to_replace=r'(mmol/L)',value=r'')
df_x['mesure'].replace(regex=True,inplace=True,to_replace=r'\(\)',value=r'')
df_x['mesure']=df_x['mesure'].str.strip()


# In[27]:


df_x=df_x[pd.notna(df_x['mesure_value'])]
df_x.to_csv(Path+"df_biological_measures.csv",index=False,encoding='utf-8-sig')


# In[ ]:




