import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

val_split = 0.1       # percentage of validation data in all data
batch_size = 128
num_epoch = 100
learning_rate = 0.001
weight_decay = 0.01
num_neuron = 100      # number of neurons in a single LSTM layer
num_layers = 1        # number of LSTM layers
dropout = 0.7         # dropout rate

train_file_path = "preprocess_4.csv"
test_file_path = "preprocess_test_2.csv"
output_file_path = "lstm_1.csv"

class LSTM_net(nn.Module):
    def __init__(self, num_feature, num_neuron, num_layers, dropout):
        super(LSTM_net, self).__init__()
        self.num_feature = num_feature
        self.num_neuron = num_neuron
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(num_feature, num_neuron, num_layers = num_layers, batch_first=True)
        # linear output layer (FFNN to output a value between 0 and 1)
        self.classifier = nn.Sequential ( nn.Dropout(dropout), nn.Linear(num_neuron, 1), nn.Sigmoid() )

    # define the forward pass of how X is passed among layers
    def forward(self, Input):
        x, _ = self.lstm(Input, None) # put Input into LSTM layers, output x in each time step
        x = x[:, -1, :]               # x has shape (batch_size, seq_len, num_neuron), obtain the output of the last time step
        x = self.classifier(x)        # put x into linear output layer
        return x
    
def train(train_loader, val_loader, model):
    model.train()
    criterion = nn.BCELoss() # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam optimizer
    max_acc = 0
    best_model = None
    
    for epoch in range(num_epoch):
        # training
        train_correct, train_total = 0, 0
        for i, (X, Y) in enumerate(train_loader):
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
                Y_P = model(X).squeeze()
                Y_P [ Y_P >= 0.5 ] = 1
                Y_P [ Y_P < 0.5 ] = 0
                loss = criterion(Y_P, Y)
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

def test(X, best_model):
    model.load_state_dict(best_model)
    model.eval()
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
    # reading train data
    data = pd.read_csv(train_file_path)
    timestamps = data.columns[3]
    data = data.sort_values(by=timestamps, ascending=True).reset_index(drop=True)
    data = data.to_numpy()
    
    X = np.delete(data, [0,3,4,5,8,9,10,11,12,39,40], axis=1) # remove id, date, is_night_game, home_team_win, home_team_rest,
    # away_team_rest, home_pitcher_rest, away_pitcher_rest, season, home_team_season, away_team_season
    num_feature = X.shape[1]
    X = np.nan_to_num(X.astype(np.float32), nan=0.0)
    X = np.expand_dims(X, axis=1)
    X_tensor = torch.from_numpy(X)

    y = data[:,5] # home_team_win
    y = [1 if str(yi).strip().upper() == "TRUE" else 0 for yi in y]
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # np.savetxt("array.csv", y, delimiter=",", fmt='%s')

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int( len(dataset)*val_split )
    train_set, val_set = random_split(dataset, [1-val_split, val_split])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # reading test data
    test_data = pd.read_csv(test_file_path)
    test_data = test_data.to_numpy()
    test_X = np.delete(test_data, [0,3,6,7,8,9,10,37,48], axis=1) # remove id, is_night_game, home_team_rest,
    # away_team_rest, home_pitcher_rest, away_pitcher_rest, season, home_team_season, away_team_season
    test_X = np.nan_to_num(test_X.astype(np.float32), nan=0.0)
    test_X = np.expand_dims(test_X, axis=1)
    test_X_tensor = torch.from_numpy(test_X)
    
    model = LSTM_net(num_feature, num_neuron, num_layers, dropout)
    best_model = train(train_loader, val_loader, model)
    test(test_X_tensor, best_model)