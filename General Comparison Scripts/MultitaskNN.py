#Multitask Neural Network with Multiple Output Nodes

#Input: Phylum, Classes: training one output node with each training point

#Import Packages
import numpy as np
import torch
import deepchem as dc
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from math import sqrt

np.random.seed(50)
torch.manual_seed(50)
global_seed = 50

df_LC50 =  pd.read_csv("Internal Preprocessed Dataframe.csv")

#Internal 5 Fold Cross Validation
chemicals = pd.Series(df_LC50['SMILES'].unique())
cv = model_selection.KFold(n_splits=5, shuffle = True, random_state = global_seed)
temp = cv.split(chemicals)
folds_setting_3 = []
for tr, te in temp:
    #get smiles
    smiles_train = chemicals.iloc[tr].tolist()
    smiles_test = chemicals.iloc[te].tolist()
    tr = list(np.where(df_LC50['SMILES'].isin(smiles_train))[0])
    te = list(np.where(df_LC50['SMILES'].isin(smiles_test))[0])
    folds_setting_3.append([tr, te])
    
x = df_LC50.drop(columns = ['Value.MeanValue', 'CAS Number', "Superclass", 'Test organisms (species)', 'Phylum', 'Class'])


resultsdf = pd.DataFrame()
fold = 0
lr = 0.005
decay = 0.0001
for index in range(len(folds_setting_3)):
            x_train, x_test = x.iloc[folds_setting_3[index][0]], x.iloc[folds_setting_3[index][1]]
            chemicals = pd.Series(x_train['SMILES'].unique())
            trainchems, valchems = model_selection.train_test_split(chemicals, test_size=0.2, random_state =global_seed) # 20% of 80% = 16% of the whole dataset : 64% train, 16% validation, 20% test 
            x_val = x_train.loc[x_train['SMILES'].isin(valchems)]
            x_train = x_train.loc[x_train['SMILES'].isin(trainchems)]
                                
            temp = ['Test organisms (species)_'+ t for t in  df_LC50['Test organisms (species)'].unique().tolist()] 
            cols = ['Value.MeanValue']
            cols.extend(temp)

            x_train = x_train.drop(columns = ["SMILES"])
            x_val = x_val.drop(columns = ["SMILES"])
            x_test =  x_test.drop(columns = ["SMILES"])
            x_train = pd.get_dummies(x_train, columns = ["Phylum", "Class"])
            x_val = pd.get_dummies(x_val, columns = [ "Phylum", "Class"])
            x_test = pd.get_dummies(x_test, columns = ["Phylum", "Class"])
            x_test = x_test.reindex(columns = x_train.columns, fill_value=0)
            x_val = x_val.reindex(columns = x_train.columns, fill_value = 0)

            y_test = df_LC50.loc[x_test.index][['Value.MeanValue', 'Test organisms (species)']]
            y_val = df_LC50.loc[x_val.index][['Value.MeanValue', 'Test organisms (species)']]
            y_train = df_LC50.loc[x_train.index][['Value.MeanValue', 'Test organisms (species)']]
            y_train = pd.get_dummies(y_train, columns = [ 'Test organisms (species)'])
            y_val = pd.get_dummies(y_val,columns = [ 'Test organisms (species)'])
            y_test = pd.get_dummies(y_test, columns = [ 'Test organisms (species)'])
            #standard columns
            y_test = y_test.reindex(columns = cols, fill_value=0)
            y_train = y_train.reindex(columns = cols, fill_value=0)
            y_val = y_val.reindex(columns = cols, fill_value=0)
            # save for weights --> disregard missing values via weight of 0
            weights_train = y_train.drop(columns = ['Value.MeanValue'])
            weights_test = y_test.drop(columns = ['Value.MeanValue'])
            weights_val = y_val.drop(columns = ['Value.MeanValue'])
            #put into correct format

            for col in cols:
                if not col == 'Value.MeanValue':
                    y_train[col] = np.where(y_train[col] == 1,y_train['Value.MeanValue'],0)
                    y_test[col] = np.where(y_test[col] == 1,y_test['Value.MeanValue'],0)
                    y_val[col] = np.where(y_val[col] == 1,y_val['Value.MeanValue'],0)

            y_train.drop(columns = ['Value.MeanValue'], inplace = True)
            y_test.drop(columns = ['Value.MeanValue'], inplace = True)
            y_val.drop(columns = ['Value.MeanValue'], inplace = True)

            traindataset = dc.data.NumpyDataset(X=x_train, y= y_train, w = weights_train, ids=x_train.index)
            valdataset = dc.data.NumpyDataset(X=x_val, y= y_val, w = weights_val, ids=x_val.index)
            testdataset = dc.data.NumpyDataset(X=x_test, y= y_test, w = weights_test, ids=x_test.index)

            model = dc.models.MultitaskRegressor(
                len(cols)-1,
                n_features=traindataset.get_shape()[0][1],
                random_state = global_seed,
                layer_sizes=[512,128,128,128] , 
                dropouts=[.25, .5, 0, 0],
                learning_rate=lr,
                weight_decay_penalty=decay,
                batch_size=64,
                log_frequency = 1,
                model_dir="Model/")
            avg_rms = dc.metrics.Metric(dc.metrics.mean_squared_error, name = 'MSE', task_averager = np.mean, mode = 'regression', n_tasks = len(cols)-1)
            epoch = 0; best_val = np.inf; patience = 5; non_improved = 0; checkpoint = np.nan
            while non_improved < patience:
                print('epoch',epoch, 'non improved', non_improved, 'best value', best_val, 'cp', checkpoint)

                # train the model for one epoch
                print(model.fit(traindataset, nb_epoch=1))
                score = model.evaluate(valdataset, [avg_rms], per_task_metrics = True)[0]['MSE']
                print(score)
                print('epoch', epoch, 'mean',score)
                if score < best_val:
                    best_val = score
                    checkpoint = epoch
                    non_improved = 0
                else: 
                    non_improved +=1
                epoch+=1
            print('epochs taken', epoch, 'Check point ', checkpoint)

            pred = model.predict(testdataset)
            actual = []
            predictions = []
            num = 0
            for i, row in df_LC50.loc[folds_setting_3[index][1]].iterrows():
                col = 'Test organisms (species)_'+ row['Test organisms (species)']
                actual.append(y_test.loc[i][col])
                indexpred = y_test.columns.tolist().index(col)
                predictions.append(pred[num][indexpred][0])
                num +=1
            rmse = sqrt(metrics.mean_squared_error(actual,predictions))
            print("Model Build.", 'Epochs taken:', epoch, ' Mean RMSE score:',rmse)
            #rerun
            x_train, x_test = x.iloc[folds_setting_3[index][0]], x.iloc[folds_setting_3[index][1]]
            x_train = pd.get_dummies(x_train.drop(columns = ['SMILES']), columns = ["Phylum", "Class"])
            x_test = pd.get_dummies(x_test.drop(columns = ['SMILES']), columns = ["Phylum", "Class"])
            x_test = x_test.reindex(columns = x_train.columns, fill_value=0)

            y_test = df_LC50.loc[x_test.index][['Value.MeanValue', 'Test organisms (species)']]
            y_train = df_LC50.loc[x_train.index][['Value.MeanValue', 'Test organisms (species)']]

            temp = ['Test organisms (species)_'+ t for t in  df_LC50['Test organisms (species)'].unique().tolist()] 
            cols = ['Value.MeanValue']
            cols.extend(temp)

            y_train = pd.get_dummies(y_train, columns = [ 'Test organisms (species)'])
            y_test = pd.get_dummies(y_test, columns = [ 'Test organisms (species)'])
            #standard columns
            y_test = y_test.reindex(columns = cols, fill_value=0)
            y_train = y_train.reindex(columns = cols, fill_value=0)
            # save for weights --> disregard missing values via weight of 0
            weights_train = y_train.drop(columns = ['Value.MeanValue'])
            weights_test = y_test.drop(columns = ['Value.MeanValue'])
            #put into correct format
            # cond, val if true, val if false
            for col in cols:
                if not col == 'Value.MeanValue':
                    y_train[col] = np.where(y_train[col] == 1,y_train['Value.MeanValue'],0)
                    y_test[col] = np.where(y_test[col] == 1,y_test['Value.MeanValue'],0)
            y_train.drop(columns = ['Value.MeanValue'], inplace = True)
            y_test.drop(columns = ['Value.MeanValue'], inplace = True)

            traindataset = dc.data.NumpyDataset(X=x_train, y= y_train, w = weights_train, ids=x_train.index)
            testdataset = dc.data.NumpyDataset(X=x_test, y= y_test, w = weights_test, ids=x_test.index)

            n_layers = 4
            model = dc.models.MultitaskRegressor(
                len(cols)-1,
                n_features=traindataset.get_shape()[0][1],
                layer_sizes=[512,128,128,128] ,
                dropouts=[.25, .5, 0, 0],
                random_state = global_seed,
                learning_rate=lr,
                weight_decay_penalty=decay,
                batch_size=64,
                log_frequency = 1,
                model_dir="Model/")
            avg_rms = dc.metrics.Metric(dc.metrics.mean_squared_error, name = 'MSE', task_averager = np.mean, mode = 'regression', n_tasks = len(cols)-1)
            model.fit(traindataset, nb_epoch = epoch-1)

            pred = model.predict(testdataset)
            actual = []
            predictions = []
            num = 0
            for i, row in df_LC50.loc[folds_setting_3[index][1]].iterrows():
                col = 'Test organisms (species)_'+ row['Test organisms (species)']
                actual.append(y_test.loc[i][col])
                indexpred = y_test.columns.tolist().index(col)
                predictions.append(pred[num][indexpred][0])
                num +=1
            rmse = sqrt(metrics.mean_squared_error(actual,predictions))
            print("Rerun Model Build.", 'Epochs taken:', epoch, ' Mean RMSE score:',rmse)
            q = 0
            for ind, row in x_test.iterrows(): 
                if q % 100 == 0: print(q, 'test results added')
                temp = pd.DataFrame([{'Fold': fold, 'Species': df_LC50.loc[ind, 'Test organisms (species)'],
                                    "Chemical":df_LC50.loc[ind, 'SMILES'] , "Duration": df_LC50.loc[ind, 'Duration.MeanValue'],
                                    "Sanity": df_LC50.loc[ind, 'Value.MeanValue'],
                                    "Actual": actual[q], "Prediction": predictions[q]}])
                resultsdf = pd.concat([resultsdf, temp])
                q+=1
            resultsdf.reset_index(drop = True, inplace = True)
            fold +=1            
            

resultsdf.to_csv('{global_seed} ECOTOX Multitask NN Internal.csv')

#Input: just chemical descriptors and duration : one training point can train multiple output nodes 

np.random.seed(50)
torch.manual_seed(50)
global_seed = 50

df_LC50 =  pd.read_csv("Internal Preprocessed Dataframe.csv")

#Internal 5 Fold Cross Validation
chemicals = pd.Series(df_LC50['SMILES'].unique())
cv = model_selection.KFold(n_splits=5, shuffle = True, random_state = global_seed)
temp = cv.split(chemicals)
folds_setting_3 = []
for tr, te in temp:
    #get smiles
    smiles_train = chemicals.iloc[tr].tolist()
    smiles_test = chemicals.iloc[te].tolist()
    tr = list(np.where(df_LC50['SMILES'].isin(smiles_train))[0])
    te = list(np.where(df_LC50['SMILES'].isin(smiles_test))[0])
    folds_setting_3.append([tr, te])
    
x = df_LC50.drop(columns = ['Value.MeanValue', 'CAS Number', "Superclass", 'Test organisms (species)', 'Phylum', 'Class'])

resultsdf = pd.DataFrame()
fold = 0
lr = 0.01
decay = 0.00001

for index in range(len(folds_setting_3)):
            x_train, x_test = x.iloc[folds_setting_3[index][0]], x.iloc[folds_setting_3[index][1]]
            chemicals = pd.Series(x_train['SMILES'].unique())
            trainchems, valchems = model_selection.train_test_split(chemicals, test_size=0.2, random_state =global_seed) # 20% of 80% = 16% of the whole dataset : 64% train, 16% validation, 20% test 
            x_val = x_train.loc[x_train['SMILES'].isin(valchems)]
            x_train = x_train.loc[x_train['SMILES'].isin(trainchems)]
                                
            temp = [t for t in  df_LC50['Test organisms (species)'].unique().tolist()] 
            cols = ['Value.MeanValue']
            cols.extend(temp)

            y_test = df_LC50.loc[x_test.index][['Value.MeanValue', 'Test organisms (species)', 'SMILES', 'Duration.MeanValue']]
            y_valid = df_LC50.loc[x_val.index][['Value.MeanValue', 'Test organisms (species)','SMILES', 'Duration.MeanValue']]
            y_train = df_LC50.loc[x_train.index][['Value.MeanValue', 'Test organisms (species)', 'SMILES', 'Duration.MeanValue']]
            
            print('X', x_train.shape, x_test.shape, x_val.shape)
            x_train.drop_duplicates(inplace = True)
            x_test.drop_duplicates(inplace = True)
            x_val.drop_duplicates(inplace = True)
            print('X2', x_train.shape, x_test.shape, x_val.shape)
            columns_tr = []
            for i, row in x_train.iterrows(): 
                columns_tr.append((row['SMILES'], row['Duration.MeanValue']))

            columns_te = []
            for i, row in x_test.iterrows(): 
                columns_te.append((row['SMILES'], row['Duration.MeanValue']))
            
            columns_val = []
            for i, row in x_val.iterrows(): 
                columns_val.append((row['SMILES'], row['Duration.MeanValue']))
   
            y_tr = pd.DataFrame(0, index = columns_tr, columns = cols[1:])
            weights_tr = pd.DataFrame(0, index = columns_tr, columns = cols[1:])
            for i, row in y_train.iterrows(): 
                y_tr.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] = row['Value.MeanValue']
                weights_tr.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] =  1
                
            y_te = pd.DataFrame(0, index = columns_te , columns = cols[1:])
            weights_te = pd.DataFrame(0, index = columns_te , columns = cols[1:])
            for i, row in y_test.iterrows(): 
                y_te.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] = row['Value.MeanValue']
                weights_te.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] =  1
            
            y_val = pd.DataFrame(0, index = columns_val , columns = cols[1:])
            weights_val = pd.DataFrame(0, index = columns_val , columns = cols[1:])
            for i, row in y_valid.iterrows(): 
                y_val.loc[[(row['SMILES'],row['Duration.MeanValue'])],row['Test organisms (species)']] = row['Value.MeanValue']
                weights_val.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] =  1
                
            x_train = x_train.drop(columns = ["SMILES"])
            x_val = x_val.drop(columns = ["SMILES"])
            x_test =  x_test.drop(columns = ["SMILES"])

            print('y', y_tr.shape, y_te.shape, y_val.shape)
            traindataset = dc.data.NumpyDataset(X=x_train, y= y_tr, w = weights_tr, ids=x_train.index)
            valdataset = dc.data.NumpyDataset(X=x_val, y= y_val, w = weights_val, ids=x_val.index)
            testdataset = dc.data.NumpyDataset(X=x_test, y= y_te, w = weights_te, ids=x_test.index)

            model = dc.models.MultitaskRegressor(
                len(cols)-1,
                n_features=traindataset.get_shape()[0][1],
                layer_sizes=[512,128,128,128] ,
                radnom_state = global_seed,
                dropouts=[.25, .5, 0, 0],
                learning_rate=lr,
                weight_decay_penalty=decay,
                batch_size=64,
                seed = global_seed,
                log_frequency = 1,
                model_dir="Model/")
            avg_rms = dc.metrics.Metric(dc.metrics.mean_squared_error, name = 'MSE', task_averager = np.mean, mode = 'regression', n_tasks = len(cols)-1)
            epoch = 0; best_val = np.inf; patience = 5; non_improved = 0; checkpoint = np.nan
            while non_improved < patience:
                print('epoch',epoch, 'non improved', non_improved, 'best value', best_val)
                # train the model for one epoch
                score = model.evaluate(valdataset, [avg_rms], per_task_metrics = True)[0]['MSE']
                if score < best_val:
                    best_val = score  
                    checkpoint = epoch
                    non_improved = 0
                else: 
                    non_improved +=1
                epoch+=1

            pred = model.predict(testdataset)
            actual = []
            predictions = []
            for i, row in y_te.iterrows():
                column_id = 0
                for col in y_te.columns: 
                    if row[col] != 0: 
                        actual.append(row[col])
                        np_i = y_te.index.get_loc(i)
                        predictions.append(pred[np_i][column_id][0])
                    column_id+=1

            rmse = sqrt(metrics.mean_squared_error(actual,predictions))
            print("Model Build.", 'Epochs taken:', epoch, ' Mean RMSE score:',rmse)
            #rerun
            x_train, x_test = x.iloc[folds_setting_3[index][0]], x.iloc[folds_setting_3[index][1]]
                                
            temp = [t for t in  df_LC50['Test organisms (species)'].unique().tolist()] 
            cols = ['Value.MeanValue']
            cols.extend(temp)

            y_test = df_LC50.loc[x_test.index][['Value.MeanValue', 'Test organisms (species)', 'SMILES', 'Duration.MeanValue']]
            y_train = df_LC50.loc[x_train.index][['Value.MeanValue', 'Test organisms (species)', 'SMILES', 'Duration.MeanValue']]
            
            x_train.drop_duplicates(inplace = True)
            x_test.drop_duplicates(inplace = True)

            columns_tr = []
            for i, row in x_train.iterrows(): 
                columns_tr.append((row['SMILES'], row['Duration.MeanValue']))

            columns_te = []
            for i, row in x_test.iterrows(): 
                columns_te.append((row['SMILES'], row['Duration.MeanValue']))
            
            y_tr = pd.DataFrame(0, index = columns_tr, columns = cols[1:])
            weights_tr = pd.DataFrame(0, index = columns_tr, columns = cols[1:])
            for i, row in y_train.iterrows(): 
                y_tr.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] = row['Value.MeanValue']
                weights_tr.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] =  1
                
            y_te = pd.DataFrame(0, index = columns_te , columns = cols[1:])
            weights_te = pd.DataFrame(0, index = columns_te , columns = cols[1:])
            for i, row in y_test.iterrows(): 
                y_te.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] = row['Value.MeanValue']
                weights_te.loc[[(row['SMILES'],row['Duration.MeanValue'])], row['Test organisms (species)']] =  1
             
            x_train = x_train.drop(columns = ["SMILES"])
            x_test =  x_test.drop(columns = ["SMILES"])

            print('y', y_tr.shape, y_te.shape, y_val.shape)
            traindataset = dc.data.NumpyDataset(X=x_train, y= y_tr, w = weights_tr, ids=x_train.index)
            testdataset = dc.data.NumpyDataset(X=x_test, y= y_te, w = weights_te, ids=x_test.index)

            n_layers = 4
            model = dc.models.MultitaskRegressor(
                len(cols)-1,
                n_features=traindataset.get_shape()[0][1],
                layer_sizes=[512,128,128,128] ,
                random_state = global_seed,
                dropouts=[.25, .5, 0, 0],
                learning_rate=lr,
                weight_decay_penalty=decay,
                batch_size=64,
                log_frequency = 1,
                seed = global_seed,
                model_dir="Model/")
            avg_rms = dc.metrics.Metric(dc.metrics.mean_squared_error, name = 'MSE', task_averager = np.mean, mode = 'regression', n_tasks = len(cols)-1)
            model.fit(traindataset, nb_epoch = epoch-1)
            
            pred = model.predict(testdataset)

            actual = []
            predictions = []
            for i, row in y_te.iterrows():
                column_id = 0
                for col in y_te.columns: 
                    if row[col] != 0: 
                        actual.append(y_te.loc[[i]][col])
                        np_i = y_te.index.get_loc(i)
                        predictions.append(pred[np_i][column_id][0])
                        #add to resultsdf
                        temp = pd.DataFrame([{'Fold': fold, 'Species': col,
                                                      "Chemical":i[0], 
                                                  "Duration": i[1],
                                                  "Actual": row[col], "Prediction":pred[np_i][column_id][0]}])
                        resultsdf = pd.concat([resultsdf, temp])
                    column_id+=1

            rmse = sqrt(metrics.mean_squared_error(actual,predictions))
            print("Rerun Model Build.", 'Epochs taken:', epoch, ' Mean RMSE score:',rmse)
            resultsdf.reset_index(drop = True, inplace = True)
            fold +=1            
            

resultsdf.to_csv('{global_seed} ECOTOX Multitask NN Alternative Internal.csv')
