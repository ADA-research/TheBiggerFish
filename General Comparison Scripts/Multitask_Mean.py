# Multitask Mean 

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

    pred = np.mean(y_tr['Value.MeanValue'].tolist())
    p = 0
    for index2, instance in y_te.iterrows(): 
        resultsdf = pd.concat([resultsdf, pd.DataFrame([{'Fold': i,'Species': instance['Test organisms (species)'] ,
                                                        "Chemical":instance['SMILES'] , 
                                                        "Duration": instance['Duration.MeanValue'],
                                                        "Actual": instance['Value.MeanValue'],
                                                        "Prediction":pred}])])
        p+=1
    i +=1
print('Internal Validation Complete')
resultsdf.to_csv(f'{global_seed}ECOTOX Multitask Mean Internal.csv')

# External Validation
np.random.seed(global_seed)

#load toxicity dataset
df_LC50_test = pd.read_excel('FINAL ECOTOX External LC50.xlsx')

resultsdf = pd.DataFrame()

pred = np.mean(df_LC50['Value.MeanValue'].tolist())
for index2, instance in df_LC50_test.iterrows(): 
    resultsdf = pd.concat([resultsdf, pd.DataFrame([{  'Species': instance['Test organisms (species)'] ,
                                    "Chemical":instance['SMILES'] , 
                                    "Duration": instance['Duration.MeanValue'],
                                    "Actual": instance['Value.MeanValue'],
                                    "Prediction":pred}])])
resultsdf.to_csv(f'{global_seed}ECOTOX Multitask Mean External.csv')
            