# Multitask Random Forest

#Import packages
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

#load toxicity dataset
df_LC50 = pd.read_excel('FINAL ECOTOX Internal LC50.xlsx')

global_seed = 50
np.random.seed(global_seed)

def fetch_fingerprints(df):
    # Calculate Molecular Fingerprints via RDKIT: ECFP Radius 2
    fps = []; delete = []; i = 0
    for s in df['SMILES']:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m,2, nBits =1024)))
        else:
            delete.append(i)
        i +=1
    df = df.reset_index(drop = True)
    df.drop(delete, inplace = True)
    df['Fingerprint'] = fps

    fingerprints = []
    for x in range(len(fps[0])):
        fingerprints.append(pd.Series([fp[x] for fp in fps], name = f'fp{x}'))
    all_columns = [d]
    all_columns.extend(fingerprints)
    df = pd.concat(all_columns, axis = 1)   
    df.drop(columns = ['Fingerprint'], inplace = True)
    return df

df_LC50 = fetch_fingerprints(df_LC50)

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

resultsdf = pd.DataFrame()
y = df_LC50['Value.MeanValue']
drop = ['Value.MeanValue','CAS Number','Superclass']
dummies = ['Test organisms (species)', 'Phylum', 'Class']
x = df_LC50.drop(columns=drop)

i = 0
for index in folds_setting_3: 
    print('Fold', i)
    x_tr, x_te = x.iloc[index[0]], x.iloc[index[1]]
    y_tr, y_te = y.iloc[index[0]], y.iloc[index[1]]

    #ensure the training set dummies to be used: no information leak on new species in the test set
    x_tr = pd.get_dummies(x_tr, columns = dummies)
    x_te = pd.get_dummies(x_te, columns = dummies)
    x_te = x_te.reindex(columns = x_tr.columns, fill_value=0)
    
    #Hyperparameter optimization
    chemicals = pd.Series(x_tr['SMILES'].unique())
    cv = model_selection.KFold(n_splits=3, shuffle = True, random_state = global_seed)
    temp = cv.split(chemicals)
    ho_folds = []
    for tr, te in temp:
        #get smiles
        smiles_train = chemicals.iloc[tr].tolist()
        smiles_test = chemicals.iloc[te].tolist()
        train = np.where(x_tr['SMILES'].isin(smiles_train))[0]
        test = np.where(x_tr['SMILES'].isin(smiles_test))[0]
        ho_folds.append([train, test])
    x_tr.drop(columns = ['SMILES'], inplace = True); x_te.drop(columns = ['SMILES'], inplace = True)
    #Hyperparameter optimisation grid taken and adapted from Autosklearn
    param_grid = { 
                  'n_estimators': stats.randint(100,500),
                  'criterion': ['squared_error', 'absolute_error'],
                  'max_features' :  stats.uniform(0.0001, 1),
                  'min_samples_split':stats.randint(2,20), 
                  'min_samples_leaf':stats.randint(1,20),
                  'bootstrap':['True', 'False'],
                  'min_weight_fraction_leaf':[0],
                  'max_depth':[None],
                  'max_leaf_nodes':[None],
                  'min_impurity_decrease':[0],
                  'random_state': [global_seed]
                }
    rf = RandomForestRegressor(n_jobs = 8)
    search = RandomizedSearchCV(estimator=rf,n_iter = 50, param_distributions=param_grid,scoring ='neg_root_mean_squared_error',cv= ho_folds, random_state = global_seed,n_jobs = 8).fit(x_tr, y_tr)
    rf_params = search.best_params_
    print(search.best_params_, search.best_score_)
    
    rf = RandomForestRegressor(**rf_params, n_jobs = 8)
    rf.fit(x_tr, y_tr)
    print('Model Fitted')
    y_pred = rf.predict(x_te)
    y_te = y_te.tolist()
    print('Predictions Fetched')
    # Append to results
    fold_results = pd.DataFrame()
    q = 0
    for ind, row in x_te.iterrows(): 
        if q % 100 == 0: print(q, 'Test results added')
        fold_results = pd.concat([fold_results,pd.DataFrame([{'Fold': i, 'Species': df_LC50.loc[ind, 'Test organisms (species)'],
                                                            "Chemical":df_LC50.loc[ind, 'SMILES'] , 
                                                            "Duration": df_LC50.loc[ind, 'Duration.MeanValue'],
                                                            "Actual": y_te[q], "Prediction": y_pred[q]}])])
        q+=1
                  
    resultsdf = pd.concat([resultsdf, fold_results])
    i+=1

resultsdf.to_csv(f'{global_seed} ECOTOX Multitask Random Forest Internal.csv')
