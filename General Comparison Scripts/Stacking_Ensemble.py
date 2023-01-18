#Stacking Ensemble Method 

#Import Packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import sys
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.utils.fixes import loguniform
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV

global_seed = 50
np.random.seed(50) 

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
    all_columns = [d]
    all_columns.extend(fingerprints)
    df = pd.concat(all_columns, axis = 1)   
    df.drop(columns = ['Fingerprint'], inplace = True)
    return df

df_LC50 = fetch_fingerprints(df_LC50)

#Internal 5 Fold Cross Validation
chemicals = pd.Series(df_LC50['SMILES'].unique())
cv = model_selection.KFold(n_splits=5, shuffle = True, random_state = global_seed)
temp = cv.split(chemicals)

folds_setting_3 = []
for tr, te in temp:
    smiles_train = chemicals.iloc[tr].tolist()
    smiles_test = chemicals.iloc[te].tolist()
    train = list(np.where(df_LC50['SMILES'].isin(smiles_train))[0])
    test = list(np.where(df_LC50['SMILES'].isin(smiles_test))[0])
    folds_setting_3.append([train, test])

class intuniform():
  def __init__(self,lower, upper,log):
    self.log = log
    self.lower = lower
    self.upper = upper
    self.rs = np.random.RandomState(1)

  def rvs(self, random_state =0):
    if self.log:
      r =  self.rs.uniform(np.log(self.lower),np.log(self.upper))
      a = np.exp(r).astype(int)
    else: 
      a = self.rs.randint(self.lower, self.upper+1)
    return a

y = df_LC50['Value.MeanValue']
drop = ['Value.MeanValue', 'CAS Number','Superclass']
dummies = ['Test organisms (species)', 'Phylum', 'Class']
x = df_LC50.drop(columns=drop)


resultsdf = pd.DataFrame()

z = 0
for index in folds_setting_3:
    z+=1
    print('Fold',z)
    x_tr, x_te = x.iloc[index[0]], x.iloc[index[1]]
    x_tr = pd.get_dummies(x_tr, columns = dummies)
    x_te = pd.get_dummies(x_te, columns = dummies)
    x_te = x_te.reindex(columns = x_tr.columns, fill_value=0)
    y_tr, y_te = y.iloc[index[0]], y.iloc[index[1]]
        
    #Hyperparameter Optimization 
    chemicals = pd.Series(x_tr['SMILES'].unique())
    cv = model_selection.KFold(n_splits=3, shuffle = True, random_state = global_seed)
    temp = cv.split(chemicals)
        
    folds_setting_3 = []
    for tr, te in temp:
        smiles_train = chemicals.iloc[tr].tolist()
        smiles_test = chemicals.iloc[te].tolist()
        train = np.where(x_tr['SMILES'].isin(smiles_train))[0]
        test = np.where(x_tr['SMILES'].isin(smiles_test))[0]
        folds_setting_3.append([train, test])
             
    param_grid = { 
          'C': loguniform(0.03125, 32768),
          'epsilon' : loguniform(0.001,1),
          'kernel' : ['rbf'], 
          'degree' : intuniform(2,4, log = False),  
          'gamma':loguniform(0.00004, 8),
          'coef0':stats.uniform(-1,1),
          'shrinking':[True, False],
          'tol' : loguniform(0.00001, 0.1), 
          'max_iter': [10000]
    }
    sv = SVR()
    s = time.time()
    search = RandomizedSearchCV(estimator=sv, n_iter = 50, param_distributions=param_grid,cv=folds_setting_3,verbose = 1,scoring ='neg_root_mean_squared_error', random_state = global_seed, n_jobs = 16).fit(x_tr.drop(columns =['SMILES']), y_tr)
    sv_params = search.best_params_    
    print('SVR',search.best_params_, search.best_score_, 'Time taken', time.time()-s)

    param_grid = {
        'loss' :['squared_error'],
        'learning_rate':loguniform(0.01,1),
        'min_samples_leaf':intuniform(1,200, log = True),
        'max_depth':[None],
        'max_bins':[255],
        'l2_regularization': loguniform(0.00001,0.1),
        'tol' : [1e-7],
        'scoring':['loss'],
        'n_iter_no_change': intuniform(1, 20, log = True),
        'validation_fraction':stats.uniform(0.01,0.4)
    }
    xg = HistGradientBoostingRegressor(random_state = global_seed)
    s = time.time()
    search = RandomizedSearchCV(estimator=xg, param_distributions=param_grid,n_iter = 50, cv=folds_setting_3,verbose = 1,scoring ='neg_root_mean_squared_error', random_state = global_seed, n_jobs = 16).fit(x_tr.drop(columns =['SMILES']), y_tr)
    xg_params = search.best_params_    
    print('XGBoost', search.best_params_, search.best_score_, 'Time taken', time.time()-s)
        
    param_grid = { 
              'n_estimators': [200,400,500],
              'criterion': ['squared_error'],
              'max_features' : intuniform(1, len(x_tr.columns), log = False),
              'min_samples_split':intuniform(2,20, log = False), 
              'min_samples_leaf':intuniform(1,20,log = False),
              'bootstrap':['True', 'False'],
              'min_weight_fraction_leaf':[0],
              'max_depth':[None],
              'max_leaf_nodes':[None],
              'min_impurity_decrease':[0],
    }
    rf = RandomForestRegressor(random_state = global_seed)
    s = time.time()
    search= RandomizedSearchCV(estimator=rf, n_iter = 50,param_distributions=param_grid,cv=folds_setting_3,verbose = 1,scoring ='neg_root_mean_squared_error',random_state = global_seed, n_jobs =16).fit(x_tr.drop(columns =['SMILES']), y_tr)
    rf_params =  search.best_params_
    print('RF', search.best_params_, search.best_score_, 'Time taken', time.time()-s)
        
    #cross validation predictions: basemodels predict 
    print('CV baselearners')
    i=0  
    chemicals = pd.Series(x_tr['SMILES'].unique())
    cv = model_selection.KFold(n_splits=5, shuffle = True, random_state = global_seed)
    temp = cv.split(chemicals)
    base_folds = []
    for tr, te in temp:
        smiles_train = chemicals.iloc[tr].tolist()
        smiles_test = chemicals.iloc[te].tolist()
        train = np.where(x_tr['SMILES'].isin(smiles_train))[0]
        test = np.where(x_tr['SMILES'].isin(smiles_test))[0]
        base_folds.append([train, test])

    x_tr.drop(columns =['SMILES'], inplace = True)
    x_te.drop(columns =['SMILES'], inplace = True)


    rf = RandomForestRegressor(random_state = global_seed, **rf_params)
    rf_pred = []

    xg = HistGradientBoostingRegressor(random_state = global_seed, **xg_params, random_state = global_seed)
    xg_pred = []

    sv = SVR(**sv_params)   
    sv_pred = []

    meta_test = []#the true values 

    for ind in base_folds:
        i+=1
        inner_start = time.time()
        print(i, 'in fold', z)
        inner_x_tr, inner_x_te = x_tr.iloc[ind[0]], x_tr.iloc[ind[1]]
        inner_y_tr, inner_y_te = y_tr.iloc[ind[0]], y_tr.iloc[ind[1]] 
        meta_test.append(inner_y_te)

        rf.fit(inner_x_tr, inner_y_tr)
        rf_pred.append(rf.predict(inner_x_te))
        print('RF trained')
        xg.fit(inner_x_tr, inner_y_tr)
        xg_pred.append(rf.predict(inner_x_te))
        print('Boost trained')
        sv.fit(inner_x_tr, inner_y_tr)
        sv_pred.append(rf.predict(inner_x_te))
        print('SVR trained')
        print('Inner fold finished in: ', time.time() - inner_start)

        
    #Train on full dataset for test set
    rf_test = []; xg_test = []; sv_test = []    
    rf.fit(x_tr, y_tr)
    rf_test.append(rf.predict(x_te))

    xg.fit(x_tr, y_tr)
    xg_test.append(rf.predict(x_te))

    sv.fit(x_tr, y_tr)
    sv_test.append(rf.predict(x_te))

    flat_rf = [r for sub in rf_pred for r in sub]
    flat_xg = [r for sub in xg_pred for r in sub]
    flat_sv = [r for sub in sv_pred for r in sub]
    meta_train = np.column_stack((rf_pred, xg_pred, sv_pred))
    meta_test = [r for sub in meta_test for r in sub]
    print("Training meta learner...")
        
    # meta learner
    lr = LinearRegression()
    lr.fit(meta_train, meta_test)
    print('Prediciting test set...')
    #prep data
    flat_rf_test = [r for sub in rf_test for r in sub]
    flat_xg_test = [r for sub in xg_test for r in sub]
    flat_sv_test = [r for sub in sv_test for r in sub]
    meta_train_test = np.column_stack((flat_rf_test, flat_xg_test, flat_sv_test))
    y_predictions = lr.predict(meta_train_test)
        
    j=0
    for index2, instance in x_te.iterrows():
        resultsdf = pd.concat([resultsdf, pd.DataFrame([{'Fold': z, 'Species': df_LC50.loc[index2, 'Test organisms (species)'],"Chemical":df_LC50.loc[index2,'SMILES'] , 
                                                        "Duration": df_LC50.loc[index2, 'Duration.MeanValue'],
                                                         "Actual": df_LC50.loc[index2, 'Value.MeanValue'], "Prediction": y_predictions[j]}])])
        j += 1
        
resultsdf.to_csv(f'{global_seed}ECOTOX Stacking Ensemble Internal.csv')
    
