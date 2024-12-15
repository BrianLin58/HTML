import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

val_split = 0.15                 # percentage of validation data in all data
batch_size = 128
num_epoch = 100
learning_rate = 0.001
weight_decay = 0.005
num_neuron = [128, 32]
num_layers = 2                   # number of layers
dropout = 0.7                    # dropout rate

train_file_path = "preprocess_5.csv"
test_file_path = ["preprocess_test_2.csv", "preprocess2024_test_2.csv"]
output_file_path = ["nn_2.csv", "nn2024_2.csv"]

class NN(nn.Module):
    def __init__(self, num_feature, num_neuron, num_layers, dropout):
        super(NN, self).__init__()
        self.num_feature = num_feature
        self.num_neuron = num_neuron
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.Sequential()
        
        self.layers.add_module(f'fc0', nn.Linear(num_feature, num_neuron[0])) # input layer
        self.layers.add_module(f'relu0', nn.ReLU())
        self.layers.add_module(f'dropout1', nn.Dropout(dropout))
        for i in range(num_layers-1): # hidden layers
            self.layers.add_module(f'fc{i+1}', nn.Linear(num_neuron[i], num_neuron[i+1]))
            self.layers.add_module(f'relu{i+1}', nn.ReLU())
            self.layers.add_module(f'dropout{i+1}', nn.Dropout(dropout))
        self.layers.add_module('out', nn.Linear(num_neuron[num_layers-1], 1)) # output layer
        self.layers.add_module('sigm', nn.Sigmoid())

    # define the forward pass of how X is passed among layers
    def forward(self, x):
        return self.layers(x)
    
def train(train_loader, val_loader, model, device):
    criterion = nn.BCELoss() # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    max_acc = 0
    best_model = None
    
    for epoch in range(num_epoch):
        # training
        model.train()
        train_correct, train_total = 0, 0
        for i, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_P = model(X).squeeze()
            loss = criterion(Y_P, Y)
            loss.backward()
            optimizer.step()
            Y_P [ Y_P >= 0.5 ] = 1
            Y_P [ Y_P < 0.5 ] = 0
            train_correct += (Y_P == Y).sum().item()
            train_total += Y.size(0)
        train_acc = train_correct / train_total

        # validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for i, (X, Y) in enumerate(val_loader):
                X = X.to(device)
                Y = Y.to(device)
                Y_P = model(X).squeeze()
                Y_P [ Y_P >= 0.5 ] = 1
                Y_P [ Y_P < 0.5 ] = 0
                val_correct += (Y_P == Y).sum().item()
                val_total += Y.size(0)
        val_acc = val_correct / val_total
        
        print(f'epoch {epoch+1}  train acc: {train_acc:.5f}  val acc: {val_acc:.5f}')
        # early stopping
        if val_acc > max_acc:
            max_acc = val_acc
            best_model = model.state_dict()
            print("best model updated")
        
    return best_model

def test(X, best_model, device, output_file_path):
    model.load_state_dict(best_model)
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        Y_P = model(X)
        Y_P [ Y_P >= 0.5 ] = 1
        Y_P [ Y_P < 0.5 ] = 0
    Y_P = Y_P.cpu().numpy().flatten()

    predictions = ["True" if pred == 1 else "False" for pred in Y_P]
    df = pd.DataFrame({
        "id": range(len(predictions)), 
        "home_team_win": predictions
    })
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reading train data
    data = pd.read_csv(train_file_path)
    data = data.to_numpy()
    
    X = data[:,1:]
    print("X ",X.shape)
    num_feature = X.shape[1]
    X = np.nan_to_num(X.astype(np.float32), nan=0.0)
    X = np.expand_dims(X, axis=1)
    X_tensor = torch.from_numpy(X)

    y = data[:,0] # home_team_win
    y = [1 if str(yi).strip().upper() == "TRUE" else 0 for yi in y]
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # np.savetxt("array.csv", y, delimiter=",", fmt='%s')

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int( len(dataset)*val_split )
    train_set, val_set = random_split(dataset, [1-val_split, val_split])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = NN(num_feature, num_neuron, num_layers, dropout)
    model = model.to(device)
    best_model = train(train_loader, val_loader, model, device)
    
    # reading test data
    for i in range(2):
        test_data = pd.read_csv(test_file_path[i])
        test_X = test_data.to_numpy()
        print("test_X ",i,test_X.shape)
        test_X = np.nan_to_num(test_X.astype(np.float32), nan=0.0)
        test_X = np.expand_dims(test_X, axis=1)
        test_X_tensor = torch.from_numpy(test_X)
        
        test(test_X_tensor, best_model, device, output_file_path[i])