#Transformational Machine Learning + Single Task Random Forest Models 
#import packages
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from BaseTMLModels import BaseTMLModels
import time
import warnings
warnings.filterwarnings('ignore')

global_seed = 50
np.random.seed(global_seed)

#load toxicity dataset
df_LC50 = pd.read_excel('FINAL ECOTOX Internal LC50.xlsx')

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
    all_columns = []
    all_columns.extend(fingerprints)
    df = pd.concat(all_columns, axis = 1)   
    df.drop(columns = ['Fingerprint'], inplace = True)
    return df

df_LC50 = fetch_fingerprints(df_LC50)

y = df_LC50['Value.MeanValue']
drop = ['Value.MeanValue','Phylum', 'CAS Number','Class', 'Superclass']
x = df_LC50.drop(columns=drop)

#Internal 5 Fold Cross Validation
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
i=0
#for each of the 5 cross validation folds
for index in folds_setting_3:
    print("Fold",i)
    x_train, x_test = x.iloc[index[0]], x.iloc[index[1]]
    y_train, y_test = y.iloc[index[0]], y.iloc[index[1]]
    
    #Hyperparamter optimization for the single task RF models
    chemicals = pd.Series(x_train['SMILES'].unique())
    cv = model_selection.KFold(n_splits=3, shuffle = True, random_state = global_seed)
    temp = cv.split(chemicals)
        
    folds = []
    for tr, te in temp:
        #get smiles
        smiles_train = chemicals.iloc[tr].tolist()
        smiles_test = chemicals.iloc[te].tolist()
        train = np.where(x_train['SMILES'].isin(smiles_train))[0]
        test = np.where(x_train['SMILES'].isin(smiles_test))[0]
        folds.append([train, test])
        
    x_train.drop(columns = ['SMILES'], inplace = True); x_test.drop(columns = ['SMILES'], inplace = True)
    fixed_species = x_train['Test organisms (species)'].unique().tolist()
    param_grid = { 
              'n_estimators': stats.randint(100,500),
              'criterion': ['squared_error', 'absolute_error'],
              'max_features' :  stats.uniform(0.001,1),
              'min_samples_split':stats.randint(2,20), 
              'min_samples_leaf':stats.randint(1,20),
              'bootstrap':['True', 'False'],
              'min_weight_fraction_leaf':[0],
              'max_depth':[None],
              'max_leaf_nodes':[None],
              'min_impurity_decrease':[0],
              'random_state': [global_seed]
    }
    rf = BaseTMLModels(species = fixed_species)
    start = time.time()
    search = RandomizedSearchCV(estimator=rf,n_iter = 50, param_distributions=param_grid,scoring ='neg_root_mean_squared_error',cv=folds,random_state = global_seed,verbose = 1, n_jobs = 8).fit(x_train, y_train)
    rf_params = search.best_params_
    print('search ended in', time.time()-start)
    print(search.best_params_, search.best_score_)

    #Fit Singular RF models on training set 
    def base_model(species50):
        print(len(species50))
        rf_dict ={}
        for s in species50:
              x_tr = x_train.loc[x_train['Test organisms (species)'] == s]
              y_tr = y_train.loc[x_tr.index.tolist()]
              if x_tr.shape[0] > 2:
                  rf = RandomForestRegressor(**rf_params)
                  temp_x_tr = x_tr.drop(columns= ['Test organisms (species)'])
                  rf.fit(temp_x_tr, y_tr)
                  rf_dict[s] = rf
        return rf_dict
    train_species_split = np.array_split(x_train['Test organisms (species)'].unique(), 30)
    dicts = Parallel(n_jobs = 12, verbose = 50)(delayed(base_model)(s) for s in train_species_split)          
    rf_dict = {k: v for d in dicts for k, v in d.items()}
    print("Base Models Built")
    
    #Build transformational embeddings and establish end models
    #tml2: single task random forest end model
    #tml3: Stacked endmodel (a single task RF with TML representations and a single task RF with chemical descriptors)
    meta_rf = RandomForestRegressor(**rf_params)
    def species_prediction(species50):
        fold_results = pd.DataFrame()
        for s in species50:
            x_tr = x_train.loc[x_train['Test organisms (species)'] == s]
            y_tr = y_train.loc[x_tr.index.tolist()]
            if x_tr.shape[0] > 0:
                x_te = x_test.loc[x_test['Test organisms (species)'] == s]
                y_te = y_test.loc[x_te.index.tolist()]
          
                # build TML embeddings
                rfs = [rf_dict[sp] for sp in rf_dict.keys()]
                # TML representations for training instances
                temp_x_tr = x_tr.drop(columns= ['Test organisms (species)'])
                preds = [rf.predict(temp_x_tr)if rf != None else np.zeros(temp_x_tr.shape[0]) for rf in rfs]
                x_trans = np.stack(preds, axis=-1)
                # TML representations for test instances
                temp_x_te = x_te.drop(columns= ['Test organisms (species)'])
                preds = [rf.predict(temp_x_te)if rf != None else np.zeros(temp_x_te.shape[0]) for rf in rfs]
                x_test_trans = np.stack(preds, axis=-1)
                          
                # Fit TML model on training TML representations
                meta_rf.fit(x_trans, y_tr.array)
                tml2_preds = meta_rf.predict(x_test_trans)
                if s in rf_dict.keys():
                    rf_preds = rf_dict[s].predict(temp_x_te)
                    #Fetch input to stacks
                    train_input = np.stack([rf_dict[s].predict(temp_x_tr),meta_rf.predict(x_trans)], axis=-1)
                    test_input = np.stack([rf_preds,tml2_preds], axis=-1)
                            
                    # stack these predictions with predictions of that singular model
                    ridge = Ridge(random_state = global_seed)
                    #fit ridge on base and tml train predictions 
                    ridge.fit(train_input, y_tr.array)
                    tml3_preds = ridge.predict(test_input)       
                else:
                    tml3_preds = [None for xt in  x_te]
                    rf_preds = [None for xt in  x_te] 
            else:
                x_te = x_test.loc[x_test['Test organisms (species)'] == s]
                y_te = y_test.loc[x_te.index.tolist()]
                if s in rf_dict.keys():  
                    temp_x_te = x_te.drop(columns= ['Test organisms (species)'])
                    rf_preds = rf_dict[s].predict(temp_x_te)
                else: 
                    rf_preds =[None for xt in  x_te]
                    tml3_preds = [None for xt in  x_te]
                    tml2_preds = [None for xt in  x_te]
              
            p = 0
            for index2, instance in x_te.iterrows():
                fold_results = pd.concat([fold_results, pd.DataFrame([{'Fold': i, 'Species': s ,
                                                                       "Chemical":df_LC50.loc[index2, 'SMILES'] , 
                                                                        "Duration": df_LC50.loc[index2, 'Duration.MeanValue'],
                                                                        "Actual": df_LC50.loc[index2, 'Value.MeanValue'],
                                                                        "RF":rf_preds[p], 
                                                                        'TML2':tml2_preds[p],
                                                                        'TML3': tml3_preds[p]}])])
                p+=1
                    
        return fold_results
    test_species_split = np.array_split(x_test['Test organisms (species)'].unique(), 30)
    f_results = Parallel(n_jobs = 42, verbose = 50)(delayed(species_prediction)(s) for s in test_species_split)
    fold_results = pd.concat(f_results)
    resultsdf = pd.concat([resultsdf, fold_results])
    i +=1
             
resultsdf.to_csv(f'{global_seed}ECOTOX TML Internal.csv')
