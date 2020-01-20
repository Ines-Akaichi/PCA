#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import pandas as pd 
from datetime import datetime
import numpy as np
import datetime

conn = psycopg2.connect(user="user_name",password="password",
                              host="10.195.25.10",
                              port="54464",
                              database="db_21807140t_stage_production")

cur = conn.cursor()

df_dates_follow_up=pd.read_sql('''SELECT f.patient_key,birthdate,sex,date_diagnostic,e.date_examination_full,date_death_full,startplace_fs,date_fs
FROM public.fact_clinical_measures_1 f ,public.dim_patient p,public.dim_date_examination e
where f.patient_key =p.patient_key and f.date_examination_key=e.date_examination_key''',conn)

#replace values
df_dates_follow_up.loc[df_dates_follow_up['startplace_fs'].eq('Cervical') | df_dates_follow_up['startplace_fs'].eq('Membre supérieur distal G') | df_dates_follow_up['startplace_fs'].eq('Membre inférieur distal G') |
df_dates_follow_up['startplace_fs'].eq('Membre supérieur distal D') | df_dates_follow_up['startplace_fs'].eq('Membre inférieur distal D') | df_dates_follow_up['startplace_fs'].eq('Membre supérieur proximal D') |df_dates_follow_up ['startplace_fs'].eq('Membre supérieur proximal G') |df_dates_follow_up ['startplace_fs'].eq('Membre supérieur proximal Bilat') | df_dates_follow_up ['startplace_fs'].eq('Membre inférieur distal Bilat') | df_dates_follow_up['startplace_fs'].eq('Membre inférieur proximal D') | df_dates_follow_up['startplace_fs'].eq('Membre inférieur proximal Bilat') | df_dates_follow_up['startplace_fs'].eq('Membre supérieur distal Bilat') | df_dates_follow_up['startplace_fs'].eq('Membre inférieur proximal G'), 'startplace_fs'] = 'Spinal' 
df_dates_follow_up.loc[df_dates_follow_up['startplace_fs'].eq('Bulbaire'),'startplace_fs'] ='Bulbaire'
df_dates_follow_up.loc[df_dates_follow_up['startplace_fs'].eq(''),'startplace_fs'] ='ND'
df_dates_follow_up.loc[df_dates_follow_up['startplace_fs'].eq('Respiratoire'),'startplace_fs'] ='Respiratoire'

#convert dates
df_dates_follow_up['date_examination_full']=pd.to_datetime(df_dates_follow_up['date_examination_full'],format='%d/%m/%Y',errors='coerce')
df_dates_follow_up['date_death_full']=pd.to_datetime(df_dates_follow_up['date_death_full'],format='%d/%m/%Y',errors='coerce')
df_dates_follow_up['birthdate']=pd.to_datetime(df_dates_follow_up['birthdate'],format='%d/%m/%Y',errors='coerce')
df_dates_follow_up['date_fs']=pd.to_datetime(df_dates_follow_up['date_fs'],format='%d/%m/%Y',errors='coerce')
df_dates_follow_up['date_diagnostic']=pd.to_datetime(df_dates_follow_up['date_diagnostic'],format='%d/%m/%Y',errors='coerce')


df_last_follow_up_date=df_dates_follow_up.groupby(['patient_key'])['date_examination_full'].max().reset_index()


# In[2]:


df_f = pd.merge(df_last_follow_up_date, df_dates_follow_up[['patient_key','date_death_full','startplace_fs','date_fs','birthdate','sex','date_diagnostic']], on=['patient_key'])
df_f=df_f.drop_duplicates(subset ="patient_key")
df_f=df_f.reset_index(drop=True)
df_f['diagnostic_delay']=df_f.date_diagnostic.dt.to_period('M') - df_f.date_fs.dt.to_period('M')
df_f['diagnostic_age']=df_f.date_diagnostic.dt.to_period('Y') - df_f.birthdate.dt.to_period('Y')

#diagnostic age groups
bins= [0,55,65,75,np.inf]
labels = ['<55','55-65','65-75','75+'] 
df_f['diagnostic_age_group'] = pd.cut(df_f.diagnostic_age[df_f['diagnostic_age'].notnull()],bins=bins, labels=labels,include_lowest=True)

#diagnostic delay groups

bins= [0,12,24,np.inf]
labels = ['<12','12-24','24+'] 
df_f['diagnostic_delay_group'] = pd.cut(df_f.diagnostic_delay[df_f['diagnostic_delay'].notnull()],bins=bins, labels=labels,include_lowest=True)

df_f['death']=0
df_f.loc[df_f.date_death_full.notnull() , 'death'] = 1


#compute survival Time 


def months(d1, d2):
    return d1.month - d2.month + 12*(d1.year - d2.year)

date_now=datetime.datetime.now()

for index, row in df_f.iterrows():
    if (pd.isnull(row['date_death_full'])):
        duration=months(row['date_examination_full'],row['date_fs'])
        df_f.set_value(index,'Duration',duration)
        #print('Do Nothing')
    else:
        duration=months(row['date_death_full'],row['date_fs'])
        df_f.set_value(index,'Duration',duration)

        
# Filter survival Data       
filtered_df = df_f[(df_f['Duration'].notnull()) & (df_f['Duration']> 0) & (df_f['Duration'] <= 175)  & (df_f['startplace_fs'] != 'ND') & (df_f['startplace_fs'] != 'Respiratoire') & (df_f['diagnostic_age_group'].notnull()) & (df_f['diagnostic_delay_group'].notnull())]


# ### Survival analysis with clinical data 

# In[4]:


from lifelines.utils import datetimes_to_durations
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


kmf = KaplanMeierFitter()
T=filtered_df['Duration']
C=filtered_df['death']


fig = plt.figure(figsize=[8, 9])
#ax=plt.subplot(1,1,1) 
for r in filtered_df['startplace_fs'].unique():   # change column startplace_fs by any group column
    ix=filtered_df['startplace_fs'] == r
    kmf.fit(T.ix[ix],C.ix[ix],label=r)
    ax=kmf.plot()#ax=ax)


# In[ ]:




