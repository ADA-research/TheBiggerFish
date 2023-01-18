# Multitask Neural Network 1 Output Node

#import Packages
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn import model_selection
from rdkit import Chem
from rdkit.Chem import AllChem
from TorchDataFrame import DataFrameTorch

global_seed = 50
np.random.seed(50)
torch.manual_seed(50) 

rerun = True

torch.set_printoptions(precision = 15)
#The neural network model: The architecture was determined by 50 iterations of a NAS TPE tuner
class Net(nn.Module):
    def __init__(self, input_size = 1443):
        super().__init__()
        hidden_size1 = 512; hidden_size2 = 128; hidden_size3 = 128; hidden_size4 = 128
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)

        self.fcout1 = nn.Linear(hidden_size4, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.fcout1(x) 
        return x

#How to train one epoch
def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Test epoch 
def test_epoch(model, device, test_loader, testinfo, columns = None, val = False):
    size = len(test_loader.dataset)
    print('test loaded size', size, len(test_loader))
    num_batches = len(test_loader)
    model.eval()
    if not val: res = pd.DataFrame()
    test_loss, rmse = 0, 0
    with torch.no_grad():
        for data, target, idex in test_loader: 
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)
            if not val:
                temp = pd.concat([testinfo.iloc[idex.numpy()].reset_index(drop=True),pd.Series(target.flatten().numpy()),pd.Series(output.flatten().numpy())], axis = 1, ignore_index = True)
                res = pd.concat([res, temp])
            # sum up batch loss
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            rmse += torch.square(torch.sub(target,output)).type(torch.float).sum().item()

    test_loss /= size
    rmse /= size
    rmse = np.sqrt(rmse)
    print('\nTest set: Average loss: {:.4f}, RMSE: {}\n'.format(
        test_loss, rmse))
    metrics = [rmse,test_loss]
    if not val: 
        res.columns = columns
        return res, metrics
    else:
        return metrics

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

if __name__ == '__main__':
    print("GPU:", torch.version.cuda, torch.cuda.is_available())

    df_LC50 =  pd.read_excel("FINAL ECOTOX Internal LC50.xlsx")
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
        tr = list(np.where(df_LC50['SMILES'].isin(smiles_train))[0])
        te = list(np.where(df_LC50['SMILES'].isin(smiles_test))[0])
        folds_setting_3.append([tr, te])
    
    x = df_LC50.drop(columns = ['Value.MeanValue', 'CAS Number', "Superclass"])
    y = df_LC50["Value.MeanValue"]
    test_info = df_LC50[['SMILES', "Test organisms (species)", "Duration.MeanValue", "Value.MeanValue"]]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    resultsdf = pd.DataFrame()
    for index in range(len(folds_setting_3)):
        x_tr, x_te = x.iloc[folds_setting_3[index][0]], x.iloc[folds_setting_3[index][1]]
        y_tr, y_te = y.iloc[folds_setting_3[index][0]], y.iloc[folds_setting_3[index][1]]
        chemicals = pd.Series(x_tr['SMILES'].unique())
        #Validation Set
        train, val = model_selection.train_test_split(chemicals, test_size=0.2, random_state = global_seed) # 20% of 80% = 16% of the whole dataset : 64% train, 16% validation, 20% test 
        x_val = x_tr.loc[x_tr['SMILES'].isin(val)]
        x_tr = x_tr.loc[x_tr['SMILES'].isin(train)]
        y_val = y_tr.loc[x_val.index]; y_tr = y_tr.loc[x_tr.index]
        x_tr = pd.get_dummies(x_tr.drop(columns = ['SMILES']), columns = ["Test organisms (species)", "Phylum", "Class"])
        x_val = pd.get_dummies(x_val.drop(columns = ['SMILES']), columns = ["Test organisms (species)", "Phylum", "Class"])
        x_te = pd.get_dummies(x_te.drop(columns = ['SMILES']), columns = ["Test organisms (species)", "Phylum", "Class"])
        x_te = x_te.reindex(columns = x_tr.columns, fill_value=0)
        x_val = x_val.reindex(columns = x_tr.columns, fill_value=0)
        te_info = test_info.iloc[folds_setting_3[index][1]]
        
        model = Net(input_size=x_tr.shape[1]).float().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay = 0.0001)

        train_loader = torch.utils.data.DataLoader(DataFrameTorch(x_tr,y_tr), batch_size=32, shuffle=True, drop_last = True)
        val_loader = torch.utils.data.DataLoader(DataFrameTorch(x_val,y_val,indexreturn = True), batch_size=32, shuffle=True, drop_last = True)
        test_loader = torch.utils.data.DataLoader(DataFrameTorch(x_te, y_te,indexreturn=True), batch_size=32, shuffle=True)
        
        columns =['SMILES', "Test organisms (species)", "Duration.MeanValue", "Value.MeanValue",'Actual','Prediction']
        patience = 5; non_improved = 0 ; best_val = np.inf
        epoch = 0
        while non_improved < patience:
            print('epoch',epoch, 'non improved', non_improved)
            epoch+=1
            # train the model for one epoch
            train_epoch(model, device, train_loader, optimizer, epoch)
            # test the model for one epoch
            metrics = test_epoch(model, device, val_loader, None, val = True) #valdation_loader
            #if the new loss is better than the previous
            if metrics[1] <= best_val:
                best_val = metrics[1]
                #save the weights
                torch.save(model.state_dict(),'nn params/bestmodel.pth')
                non_improved = 0
            else:
                non_improved +=1
        print('Epochs taken:', epoch)
        
        if rerun:
            x_tr, x_te = x.iloc[folds_setting_3[index][0]], x.iloc[folds_setting_3[index][1]]
            y_tr, y_te = y.iloc[folds_setting_3[index][0]], y.iloc[folds_setting_3[index][1]]
            x_tr = pd.get_dummies(x_tr.drop(columns = ['SMILES']), columns = ["Test organisms (species)", "Phylum", "Class"])
            x_te = pd.get_dummies(x_te.drop(columns = ['SMILES']), columns = ["Test organisms (species)", "Phylum", "Class"])
            x_te = x_te.reindex(columns = x_tr.columns, fill_value=0)
            model = Net(input_size=x_tr.shape[1]).float().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay = 0.0001)
            train_loader = torch.utils.data.DataLoader(DataFrameTorch(x_tr,y_tr), batch_size=32, shuffle=True, drop_last = True)
            test_loader = torch.utils.data.DataLoader(DataFrameTorch(x_te, y_te,indexreturn=True), batch_size=32, shuffle=True)
            for epoch_rerun in range((epoch)):
                 train_epoch(model, device, train_loader, optimizer, epoch)
            restemp, test = test_epoch(model, device, test_loader,te_info, columns = columns) 
        else:    
            model = Net(input_size=x_tr.shape[1]).float().to(device)
            model.load_state_dict(torch.load('nn params/bestmodel.pth'))
            restemp, test = test_epoch(model, device, test_loader,te_info, columns = columns) 
        restemp['Fold'] = index
        resultsdf = pd.concat([resultsdf,restemp])

    resultsdf.to_csv(f'{global_seed}ECOTOX NN Internal.csv')
        