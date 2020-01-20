# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:51:58 2020

@author: akaic
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:21:19 2019

@author: akaic
"""
import psycopg2
import pandas as pd 
import jenkspy
import numpy as np
from sklearn.linear_model import LinearRegression
from lifelines.statistics import multivariate_logrank_test
import seaborn as sns
from lifelines import KaplanMeierFitter


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


conn = psycopg2.connect(user="user_name",password="password",
                              host="10.195.25.10",
                              port="54464",
                              database="db_21807140t_stage_production")


#df
bio_data=pd.read_sql('''SELECT f.patient_key,date_fs, f.date_sampling_key,date_sampling_full,measure_key,measure FROM public.Fact_Bilogical_measures f,public.Dim_Date_Sampling d ,public.dim_patient p
where f.patient_key=p.patient_key and f.date_sampling_key = d.date_sampling_key;''',conn)

#patients
patients_data=pd.read_sql('''SELECT f.patient_key,birthdate,sex,date_diagnostic,e.date_examination_full,date_death_full,startplace_fs,date_fs
FROM public.fact_clinical_measures_1 f ,public.dim_patient p,public.dim_date_examination e
where f.patient_key =p.patient_key and f.date_examination_key=e.date_examination_key ;''',conn)

measures=pd.read_sql('''SELECT * FROM public.dim_measure ;''',conn)


# Require : DF
# Ensure: Same DF but with converted measure column
def Convert_Measure(df): 
    df=df.iloc[df[['measure']].convert_objects(convert_numeric=True).dropna().index]
    df['measure']=pd.to_numeric(df["measure"])
    return df

# Require : DF
# Ensure: Same DF but with converted dates columns
def Convert_Dates(df):
    df=df.replace('ND', np.nan)
    data = [pd.to_datetime(df[x],format='%d/%m/%Y',errors='coerce') if df[x].astype(str).str.match(r'[0-9]{2}/[0-9]{2}/[0-9]{4}').any() else df[x] for x in df.columns]
    df = pd.concat(data, axis=1, keys=[s.name for s in data])
    return df

# Require : two dates
# Ensure: number of months between two dates
    
def months(d1, d2):
    return d1.month - d2.month + 12*(d1.year - d2.year)

# Require : DF
# Ensure: DF of the form [patient_key,Duration,death] : Duration is the survival time and death takes 1 or 0
    
def Prepare_Suvival_Data(df):
    #convert
    df=Convert_Dates(df)
    dfmax_date = df.groupby('patient_key')['date_examination_full']
    df['max_exam_date'] =dfmax_date.transform('max')
    
    for index, row in df.iterrows():
        if (pd.isnull(row['date_death_full'])):
            duration=months(row['max_exam_date'],row['date_fs'])
            df.set_value(index,'Duration',duration)
        else:
            duration=months(row['date_death_full'],row['date_fs'])
            df.set_value(index,'Duration',duration)
    df['death']=0
    df.loc[df.date_death_full.notnull(), 'death'] = 1
    df=df.drop_duplicates(subset=['patient_key']).reset_index()
    df=df.drop(axis=1,labels=['index'])
    df=df[df['Duration'].notnull()]
    df=df[df['Duration'] > 0]
    df=df[df['Duration'] <= 70 ]
    
    return df[['patient_key','Duration','death']] 

# Require : DF
# Ensure: DF with added column delta computed as  the number of months between first exam date and the next exam date (delta = 0 is the first exam date )
    
def compute_delta_months(df):
    dfmin_date = df.groupby('patient_key')['date_sampling_full']
    df['min_exam_date'] =dfmin_date.transform('min')
    df['min_exam_date']=pd.to_datetime(df['min_exam_date'],format='%d/%m/%Y',errors='coerce')
    df['delta_months']=(df.date_sampling_full.dt.to_period('M') - df.min_exam_date.dt.to_period('M'))


# Require : DF, measure key, the min date of the chosen period, max date ,number of classes per group oof measure, labels of groups
# Ensure: DF of the format [patient key, slope , measure key , breaks] : slope is the varaiation of every measure for every patient and breaks are the result of the natural jenks method to create groups
    
def filter_data(df,measure_key,min_date,max_date,nb_classes,labels) :
    df=Convert_Dates(df)
    df=Convert_Measure(df)
    df=df.loc[ df['date_sampling_full'] > df['date_fs']]
    compute_delta_months(df)
    df=df[['patient_key','measure_key','delta_months','measure']].sort_values(['patient_key','delta_months'])

    try:
        df_f1=df[df['measure_key']==measure_key]
        df_f2=df_f1[(df_f1['delta_months']>=min_date) & (df_f1['delta_months']<=max_date) ]
        agg=df_f2.groupby(['patient_key','delta_months'])['measure'].mean().reset_index().rename(columns={'measure':'measure'})
        counts=agg.groupby('patient_key')['delta_months'].size().reset_index(name='counts')
        patient_list = counts[counts['counts'] > 1]['patient_key'].tolist()
        data = agg[agg['patient_key'].isin(patient_list)]
        p=reg_model(data,measure_key)
        d=pd.DataFrame(p, columns=['patient_key', 'slope','measure_key'])
        breaks=create_groups(d,nb_classes,labels,measure_key)
        d['breaks']=[breaks for _ in range(len(d))]
        sns_plot=sns.distplot(d['slope'])
        fig=sns_plot.get_figure()
        fig.savefig("D:/StageMai2019/Project/fig"+str(max_date)+"_"+str(measure_key)+".png")
        fig.clf()
    except:
        print('error')
    return d
    
# Require : DF [d, number of classes per group of measure , number of labels ,measure key] and d : A DF contains the measure key, patient key , slope 
# Ensure: DF with added column group
def create_groups(d,nb_classes,labels,measure_key):
    breaks = jenkspy.jenks_breaks(d['slope'], nb_classes)
    d['group']=pd.cut(d['slope'], breaks, labels=labels,include_lowest=True)
    return breaks 


# Require : DF [patient key, delta_months, measure ] and measure_KEY
# Ensure: A DF [patient key, slope ,measure key]
    
def reg_model(df,measure_key):
    P=[] #[[patient_key,slope,measure]]
    dict=df.groupby('patient_key').groups
    for k in dict.keys():  #k is a patient
        list_index=dict[k]
        X=[] #months
        Y=[] #measure values
        for index in list_index:
            month=df.loc[index,'delta_months']
            measure=df.loc[index,'measure']
            X.append(month)
            Y.append(measure)
        #regression model
        regressor = LinearRegression()  
        X_shaped=np.array(X).reshape(-1, 1)
        Y_shaped=np.array(pd.to_numeric(Y)).reshape(-1, 1)
        #print(X_shaped)
        #print(Y_shaped)
        regressor.fit(X_shaped,Y_shaped)
        #predicting values for 0 and 9 months
        #y_pred = regressor.predict(np.array([0,9]).reshape(-1, 1))
        #compute slop between 0 and 9 months 
        slope = regressor.coef_[0][0]
        #Store all of this 
        P.append([k,slope,measure_key]) 
    return P

#Require : 
#DF :bio date
#DF2: survival date
#end date: 6 or 9 or 12
#start date: 0
# Ensure:  df_stats_final [group, median survival, number per group , measure name], df_logrank_final [pvalue, measure name, start_date, end_date]
def Compute_Pvalue(df,df2,end_date,start_date=0): #df=bio data, #df2=survival data
    measures_list=measures['measure_key'].tolist()
    df_stats=pd.DataFrame()
    df_logrank=pd.DataFrame()
    df_stats_final=pd.DataFrame()
    df_logrank_final=pd.DataFrame()
    
    for m in measures_list:
        number =0
        numberp =0
        print(m)
        if (m != 30):
            d=filter_data(df,m,0,end_date,2,['1','2'])
            #d['patient_key']=d['patient_key'].astype(int)
            measure_name=measures[measures['measure_key']==m]['measure_name']
            d['measure_name']=measure_name.values[0]
            if d.empty == False:
                #filename='C:/Users/akaic/bio/data_bio_measure_'+str(measure_name.values[0])+'.csv'
                #Join patients with their survival data
                data=surviv_data.merge(d,how='inner',on='patient_key')
                # Create survival model here 
                groups=['1','2']
                ###
                #Save infos
                NbP=0
                try:
                    for g in groups:
                        group1=data[data['group']==g]
                        T=group1['Duration']
                        E=group1['death']
                        kmf_1=KaplanMeierFitter().fit(T, E, label="Group "+str(g))
                        median1=kmf_1.median_survival_time_
                        nb1=data[data['group']==g].shape[0]
                        df_stats.loc[number,'group']='group_'+str(g)+'_'+str(measure_name.values[0])
                        df_stats.loc[number,'median']=median1
                        df_stats.loc[number,'nombre']=nb1
                        df_stats.loc[number,'measure']=(measure_name.values[0])

                        #print(d[1,'breaks'][0])
                        #df_stats.at[number,'breaks']=str(d[1,'breaks'][0])
                        number=number+1
                        NbP=NbP+nb1
                    if (NbP >=20):
                        print('Yes')
                        p_value=multivariate_logrank_test(data['Duration'], data['group'], data['death']).p_value
                        #df_logrank.loc[numberp,'breaks'].applymap(lambda x: d.iloc[1]['breaks'])
                        df_logrank.loc[numberp,'pvalue']=p_value
                        df_logrank.loc[numberp,'measure']=(measure_name.values[0])
                        df_logrank.loc[numberp,'start_date']=0
                        df_logrank.loc[numberp,'end_date']=end_date
                        
                        df_stats_final=df_stats_final.append(df_stats,ignore_index=True)
                        df_logrank_final=df_logrank_final.append(df_logrank,ignore_index=True)
                except:
                        pass
                    
    return df_stats_final,df_logrank_final
       



#Main
    

#parameters 
end_date=12

#prepare survival data
surviv_data=Prepare_Suvival_Data(patients_data)  
          
#Survival analysis
df_logrank_compare=pd.DataFrame()
df_stats_final,df_logrank_final=Compute_Pvalue(bio_data,surviv_data,end_date) #put parameter end date
df_logrank_compare=df_logrank_compare.append(df_logrank_final[df_logrank_final['pvalue']<0.05])

