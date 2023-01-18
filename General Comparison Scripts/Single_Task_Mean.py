# Single Task Mean

#Import packages
import pandas as pd
import numpy as np
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

#load toxicity dataset
df_LC50 = pd.read_excel('FINAL ECOTOX Internal LC50.xlsx')

global_seed = 50
np.random.seed(global_seed)

# Internal 5 Fold Cross Validation
chemicals = pd.Series(df_LC50['SMILES'].unique())
cv = model_selection.KFold(n_splits=5, shuffle = True, random_state = global_seed)
temp = cv.split(chemicals)

folds_setting_3 = []
for tr, te in temp:
    #get smiles
    smiles_train = chemicals.iloc[tr].tolist()
    smiles_test = chemicals.iloc[te].tolist()
    train = list(np.where(df_LC50['SMILES'].isin(smiles_train))[0])
    test = list(np.where(df_LC50['SMILES'].isin(smiles_test))[0])
    folds_setting_3.append([train, test])

i = 0
resultsdf = pd.DataFrame()
for index in folds_setting_3: 
    print('Fold', i)
    y_tr, y_te = df_LC50.iloc[index[0]], df_LC50.iloc[index[1]]
    for s in y_te['Test organisms (species)'].unique():
        y_tr_s = y_tr.loc[y_tr['Test organisms (species)'] == s]
        y_te_s = y_te.loc[y_te['Test organisms (species)'] == s]
        pred = np.mean(y_tr_s['Value.MeanValue'].tolist())
        p = 0
        for index2, instance in y_te_s.iterrows(): 
            resultsdf = pd.concat([resultsdf, pd.DataFrame([{'Fold': i,'Species': instance['Test organisms (species)'] ,
                                                          "Chemical":instance['SMILES'] , 
                                                          "Duration": instance['Duration.MeanValue'],
                                                          "Actual": instance['Value.MeanValue'],
                                                          "Prediction":pred}])])
            p+=1
    i+=1
resultsdf.to_csv(f'{global_seed}ECOTOX Single Task Mean Internal.csv')

#External Validation 
np.random.seed(global_seed)
df_LC50_test = pd.read_excel('FINAL ECOTOX External LC50.xlsx')

resultsdf = pd.DataFrame()
for s in df_LC50_test['Test organisms (species)'].unique():
    y_tr_s = df_LC50.loc[df_LC50['Test organisms (species)'] == s]
    y_te_s = df_LC50_test.loc[df_LC50_test['Test organisms (species)'] == s]
    pred = np.mean(y_tr_s['Value.MeanValue'].tolist())
    p = 0
    for index2, instance in y_te_s.iterrows(): 
        resultsdf = pd.concat([resultsdf,pd.DataFrame([{'Species': instance['Test organisms (species)'] ,
                                                        "Chemical":instance['SMILES'] , 
                                                        "Duration": instance['Duration.MeanValue'],
                                                        "Actual": instance['Value.MeanValue'],
                                                        "Prediction":pred}])])
        p+=1
resultsdf.to_csv(f'{global_seed}ECOTOX Single Task Mean External.csv')