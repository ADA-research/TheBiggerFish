#MAML 
# Implemented using Learn2Learn and their examples: http://learn2learn.net/
#Install Packages
import pandas as pd
import numpy as np
import learn2learn as l2l
import torch
import torch.nn.functional as F
from torch import nn, optim
from math import sqrt, ceil
from sklearn import model_selection
from rdkit import Chem
from rdkit.Chem import AllChem

class Model(nn.Module):
    def __init__(self, input_size = 1048):
        super().__init__()
        hidden_size1 = 1024
        hidden_size2 = 128

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.25)
        self.fcout1 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fcout1(x)
        return x

#Performance Metrics
def rmse(predictions, targets):
    squared_error = ((predictions - targets)*(predictions - targets)).sum().data
    squared_error/= predictions.size(0)
    return sqrt(squared_error)


def within_a_factor(y_pred, y_te, device,factor= 10):
    base = torch.full(size = [len(y_te)], fill_value = 10).to(device)
    y_pred = torch.Tensor.pow(base, y_pred.reshape(-1))
    y_te = torch.Tensor.pow(base, y_te.reshape(-1))
    pos = 0
    for i in range(len(y_te)):
        #define range 
        low = y_te[i].item()/factor; high = y_te[i].item()*factor
        if low <= y_pred[i].item() <= high:
            pos +=1
    return (pos/ y_te.size(0))*100

#Function for testing
def fast_adapt_test(x_train, x_test, y_train, y_test, learner, loss, adaptation_steps, device):
    # if the species has training examples
    if x_train is not None:
        #get training instances and labels (from the species)
        data = torch.tensor(x_train.values, dtype = torch.float)
        labels = torch.reshape(torch.tensor(y_train.values, dtype = torch.float),(y_train.shape[0],1))
        data, labels = data.to(device), labels.to(device)
        # Adapt the model using n adaption steps: 1
        for step in range(adaptation_steps):
            train_error = loss(learner.forward(data), labels)
            learner.adapt(train_error)

    #Get test instances and labels (from the species)
    testdata = torch.tensor(x_test.values, dtype = torch.float)
    testlabels = torch.reshape(torch.tensor(y_test.values, dtype = torch.float),(y_test.shape[0],1))
    testdata, testlabels = testdata.to(device), testlabels.to(device)
    
    # Evaluate the adapted model
    learner.eval()
    predictions = learner(testdata) #train the model (via l2l)
    valid_error = loss(predictions, testlabels) #Calculate loss (MSE)
    return valid_error, rmse(predictions, testlabels),  within_a_factor(predictions,testlabels, device),  predictions

#Function for training
def fast_adapt(x, y, learner, loss, adaptation_steps, device):
    # Get training instances and labels (from the species)
    data = torch.tensor(x.values, dtype = torch.float)
    labels = torch.reshape(torch.tensor(y.values, dtype = torch.float),(y.shape[0],1))
    data, labels = data.to(device), labels.to(device)

    #To imitate the low resource testing setting, we sample up to 10 samples from the species for training
    
    adaptation_indices = np.zeros(data.size(0), dtype=bool) #boolean numpy array the length of how many rows data has

    indices = np.random.choice(np.arange(0,data.size(0)), min(10, ceil(data.size(0)/2)), replace= False) # Maximum 10 sampeles
    adaptation_indices[indices] = True
    
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices] #train
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices] #test/val

    # Adapt the model: fine tuning epochs
    learner.train()
    for step in range(adaptation_steps):
        train_error = loss(learner.forward(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels) # error
    return valid_error, rmse(predictions, evaluation_labels), within_a_factor(predictions,evaluation_labels, device)

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

#Load Fingerprints and folds
def loadQSARdata(df_LC50):
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
    return folds_setting_3, df_LC50

def loadFold(i, folds_setting_3, df_LC50, valid = True):
    x = df_LC50.drop(columns = ['Value.MeanValue', 'CAS Number', "Superclass", "Phylum", "Class"])
    y = df_LC50["Value.MeanValue"]

    x_tr, x_te = x.iloc[folds_setting_3[i][0]], x.iloc[folds_setting_3[i][1]].copy()
    y_tr, y_te = y.iloc[folds_setting_3[i][0]], y.iloc[folds_setting_3[i][1]].copy()

    if valid : # Return a validation set too
        chemicals = pd.Series(x_tr['SMILES'].unique())
        train_chem, val_chem = model_selection.train_test_split(chemicals, test_size=0.2, random_state = global_seed) # 20% of 80% = 16% of the whole dataset : 64% train, 16% validation, 20% test 
        x_val = x_tr.loc[x_tr['SMILES'].isin(val_chem)]
        x_tr = x_tr.loc[x_tr['SMILES'].isin(train_chem)]
        y_val = y_tr.loc[x_val.index]; y_tr = y_tr.loc[x_tr.index]
        x_val.drop(columns = ['SMILES'], inplace = True)
        x_tr.drop(columns = ['SMILES'], inplace = True)
        x_te.drop(columns = ['SMILES'], inplace = True)
        train_species = [s for s in x_tr['Test organisms (species)'].unique().tolist() if x_tr.loc[x_tr['Test organisms (species)']==s].shape[0] > 1]
        val_species = x_val['Test organisms (species)'].unique().tolist()
        test_species = x_te['Test organisms (species)'].unique().tolist()
        return x_tr, y_tr, train_species, x_val, y_val, val_species, x_te, y_te, test_species
    else: #No Validation set
        x_tr.drop(columns = ['SMILES'], inplace = True)
        x_te.drop(columns = ['SMILES'], inplace = True)
        train_species = [s for s in x_tr['Test organisms (species)'].unique().tolist() if x_tr.loc[x_tr['Test organisms (species)']==s].shape[0] > 1]
        test_species = x_te['Test organisms (species)'].unique().tolist()
        return x_tr, y_tr, train_species, x_te, y_te, test_species

def main(
        df_LC50, 
        folds_setting_3,
        rerun = False, 
        meta_lr=0.001,
        decay = 0.00001,
        fast_lr=0.4,
        meta_batch_size=64,
        val_batch_size = 64,
        adaptation_steps=1,
        cuda=True,
):
    if cuda:
        torch.cuda.manual_seed(global_seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    resultsdf = pd.DataFrame()
    rerunresultsdf = pd.DataFrame()
    for f in [0,1,2,3,4]:
        print('Fold', f)
        x_tr, y_tr, train_species, x_val, y_val, val_species, x_te, y_te, test_species = loadFold(f, folds_setting_3, df_LC50)
 
        meta_batch_size=len(train_species)
        val_batch_size = len(val_species)

        model = Model()
        model.to(device)
        #maml algortihm, opt, loss 
        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        opt = optim.Adam(maml.parameters(), meta_lr,weight_decay = decay) 
        loss = nn.MSELoss()

        train_rmse = []; train_f10 =[]; train_loss = []
        val_rmse = []; val_f10 = []; val_loss = []
        #Early stopping loop
        patience = 25; non_improved = 0; best_val = np.inf; iteration = 0
        opt.zero_grad()
        while non_improved < patience:
            meta_train_error = 0.0
            trainrmse = 0.0 ; trainf10 = 0.0
            valrmse = 0.0; valf10 = 0.0
            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone() #clone maml to not fine tune on tasks
                #For us: sample element in list and set up task data
                train_task =train_species[task]
                x_tr_task = x_tr.loc[x_tr['Test organisms (species)'] == train_task].copy()
                x_tr_task.drop(columns = ['Test organisms (species)'], inplace = True)
                y_tr_task = y_tr.loc[x_tr_task.index.tolist()]
                #fast adapt
                evaluation_error, rmse, f10 = fast_adapt(x_tr_task, y_tr_task,
                                                                learner,
                                                                loss,
                                                                adaptation_steps,
                                                                device)                                            
                #backpropogate evaluation error on batch task(s)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                trainrmse += rmse
                trainf10 += f10
                if task % 64 == 0:
                    # Average the accumulated gradients and optimize: Update the initialization parameters every 64 iterations
                    counter = 0
                    for p in maml.parameters():
                        p.grad.data.mul_(1.0 / 64)
                        counter +=1
                    opt.step()
                    opt.zero_grad()
            # Print some metrics
            train_rmse.append(trainrmse/meta_batch_size); train_f10.append(trainf10/meta_batch_size); train_loss.append(meta_train_error/meta_batch_size)
            print('\n')
            print('Fold', f, 'Iteration', iteration, "non imporved", non_improved, 'Best Val Score', best_val)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Metrics RMSE',train_rmse[-1], 'F10', train_f10[-1])
            iteration += 1 

            valrmse = 0.0; valf10 = 0.0; valloss = 0.0
            for task in range( val_batch_size):
                # Compute meta-testing loss
                learner = maml.clone()
                val_task = val_species[task]
                x_val_task = x_val.loc[x_val['Test organisms (species)'] == val_task].copy()
                x_val_task_temp = x_val_task.drop(columns = ['Test organisms (species)'])
                y_val_task = y_val.loc[x_val_task_temp.index.tolist()]
                if x_tr.loc[x_tr['Test organisms (species)'] == val_task].shape[0] > 0:
                    x_tr_task = x_tr.loc[x_tr['Test organisms (species)'] == val_task].copy()
                    x_tr_task.drop(columns = ['Test organisms (species)'], inplace = True)
                    y_tr_task = y_tr.loc[x_tr_task.index.tolist()]
                else: 
                    x_tr_task = None; y_tr_task = None      
                vloss, rmse, f10, _ = fast_adapt_test(x_tr_task, x_val_task_temp, y_tr_task, y_val_task,learner,loss,adaptation_steps,device)
                    
                valloss += vloss.item(); valrmse += rmse; valf10 += f10
            val_rmse.append((valrmse/val_batch_size)); val_f10.append((valf10/val_batch_size)); val_loss.append((valloss/val_batch_size))
            if val_loss[-1] <= best_val:
                best_val = val_loss[-1]
                #save the weights
                torch.save(maml.state_dict(),'bestmodel.pth')
                non_improved = 0
            else:
                non_improved +=1
            print('Meta Valid Error',val_loss[-1])
            print('Meta Valid Metrics RMSE',val_rmse[-1], 'F10', val_f10[-1])

        meta_test_error = 0.0
        testrmse = 0.0; testf10 = 0.0
        print(len(test_species))
        fold_resultsdf = pd.DataFrame()

        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        maml.load_state_dict(torch.load('bestmodel.pth'))

        for test_task in test_species:
            # Compute meta-testing loss
            learner = maml.clone()
            x_te_task = x_te.loc[x_te['Test organisms (species)'] == test_task].copy()

            x_te_task_temp = x_te_task.drop(columns = ['Test organisms (species)'])
            y_te_task = y_te.loc[x_te_task_temp.index.tolist()]
            if x_tr.loc[x_tr['Test organisms (species)'] == test_task].shape[0] > 0:
                x_tr_task = x_tr.loc[x_tr['Test organisms (species)'] == test_task].copy()
                x_tr_task.drop(columns = ['Test organisms (species)'], inplace = True)
                y_tr_task = y_tr.loc[x_tr_task.index.tolist()]
            else: 
                x_tr_task = None; y_tr_task = None      
            evaluation_error, rmse, f10, pred = fast_adapt_test(x_tr_task, x_te_task_temp, y_tr_task, y_te_task, #xtrain, xtest, ytrain, ytest
                                                                    learner,
                                                                    loss,
                                                                    adaptation_steps,
                                                                    device)
            meta_test_error += evaluation_error.item()
            testrmse += rmse; testf10 += f10
            pred = pred.flatten().cpu().detach().numpy()
            q = 0
            for index, row in x_te_task.iterrows(): 
                fold_resultsdf = pd.concat([fold_resultsdf, pd.DataFrame([{"Fold": f, 'Species': row['Test organisms (species)'], 'Chemical': df_LC50.loc[index, 'SMILES'], 'Duration': row['Duration.MeanValue'], 
                'Actual': df_LC50.loc[index, 'Value.MeanValue'], 'Prediction': pred[q]}])])
                q+=1
        print('Meta Test Error', meta_test_error / len(test_species))
        print('Meta Test RMSE', testrmse / len(test_species), 'F10', testf10/len(test_species))
        resultsdf = pd.concat([resultsdf, fold_resultsdf])

        if rerun: 
            x_tr, y_tr, train_species, x_te, y_te, test_species = loadFold(f, folds_setting_3, df_LC50, valid = False)
            model = Model()
            model.to(device)
            maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
            opt = optim.Adam(maml.parameters(), meta_lr,weight_decay = decay)
            loss = nn.MSELoss()
            print('Models initialised')
            #for iterations
            train_rmse = []; train_f10 =[]; train_loss = []
            val_rmse = []; val_f10 = []; val_loss = []
            print('Starting Rerun')
            opt.zero_grad()
            for epoch in range(iteration):
                meta_train_error = 0.0
                trainrmse = 0.0 ; trainf10 = 0.0
                valrmse = 0.0; valf10 = 0.0
                for task in range(meta_batch_size):
                    # Compute meta-training loss
                    learner = maml.clone() #clone maml to not fine tune on tasks
                    #For us: sample element in list and set up task data
                    train_task = train_species[task]
                    x_tr_task = x_tr.loc[x_tr['Test organisms (species)'] == train_task].copy()
                    x_tr_task.drop(columns = ['Test organisms (species)'], inplace = True)
                    y_tr_task = y_tr.loc[x_tr_task.index.tolist()]
                    #fast adapt
                    evaluation_error, rmse, f10 = fast_adapt(x_tr_task, y_tr_task,
                                                                        learner,
                                                                        loss,
                                                                        adaptation_steps,
                                                                        device)                                            
                    #backpropogate evaluation error on batch task(s)
                    evaluation_error.backward()
                    meta_train_error += evaluation_error.item()
                    trainrmse += rmse
                    trainf10 += f10
                    if task % 64 == 0:
                        # Average the accumulated gradients and optimize: Update the initialization parameters
                        counter = 0
                        for p in maml.parameters():
                            p.grad.data.mul_(1.0 / 64)
                            counter +=1
                        opt.step()
                        opt.zero_grad()
                # Print some metrics
                train_rmse.append(trainrmse/meta_batch_size); train_f10.append(trainf10/meta_batch_size); train_loss.append(meta_train_error/meta_batch_size)
                print('\n')
                print('Fold', f, 'Iteration', epoch, '/ ', iteration)
                print('Meta Train Error', meta_train_error / meta_batch_size)
                print('Meta Train Metrics RMSE',train_rmse[-1], 'F10', train_f10[-1])
                epoch += 1 
            torch.save(maml.state_dict(),'bestmodel.pth')
        #End test error
        meta_test_error = 0.0
        testrmse = 0.0; testf10 = 0.0
        print(len(test_species))
        fold_resultsdf = pd.DataFrame()

        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        maml.load_state_dict(torch.load('bestmodel.pth'))

        for test_task in test_species:
            # Compute meta-testing loss
            learner = maml.clone()
            x_te_task = x_te.loc[x_te['Test organisms (species)'] == test_task].copy()

            x_te_task_temp = x_te_task.drop(columns = ['Test organisms (species)'])
            y_te_task = y_te.loc[x_te_task_temp.index.tolist()]
            if x_tr.loc[x_tr['Test organisms (species)'] == test_task].shape[0] > 0:
                x_tr_task = x_tr.loc[x_tr['Test organisms (species)'] == test_task].copy()
                x_tr_task.drop(columns = ['Test organisms (species)'], inplace = True)
                y_tr_task = y_tr.loc[x_tr_task.index.tolist()]
            else: 
                x_tr_task = None; y_tr_task = None      
            evaluation_error, rmse, f10, pred = fast_adapt_test(x_tr_task, x_te_task_temp, y_tr_task, y_te_task, #xtrain, xtest, ytrain, ytest
                                                                    learner,
                                                                    loss,
                                                                    adaptation_steps,
                                                                    device)
            meta_test_error += evaluation_error.item()
            testrmse += rmse; testf10 += f10
            pred = pred.flatten().cpu().detach().numpy()
            q = 0
            for index, row in x_te_task.iterrows(): 
                fold_resultsdf = pd.concat([fold_resultsdf, pd.DataFrame([{"Fold": f, 'Species': row['Test organisms (species)'], 'Chemical': df_LC50.loc[index, 'SMILES'], 'Duration': row['Duration.MeanValue'], 
                'Actual': df_LC50.loc[index, 'Value.MeanValue'], 'Prediction': pred[q]}])])
                q+=1
        print('Meta Test Error', meta_test_error / len(test_species))
        print('Meta Test RMSE', testrmse / len(test_species), 'F10', testf10/len(test_species))

        rerunresultsdf = pd.concat([rerunresultsdf, fold_resultsdf])

    rerunresultsdf.to_csv(f'{global_seed}ECOTOX MAML Rerun Internal.csv') 
    resultsdf.to_csv(f'{global_seed}ECOTOX MAML Internal.csv')

if __name__ == '__main__':
    rerun = True
    seed = 50
    global_seed = seed
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    df_LC50 =  pd.read_excel("FINAL ECOTOX Internal LC50.xlsx")
    df_LC50 = fetch_fingerprints(df_LC50)
    
    folds_setting_3, df_LC50 = loadQSARdata(df_LC50)

    main(df_LC50, folds_setting_3,meta_lr=0.001,fast_lr = 0.3, decay = 0.01 ,rerun = rerun)