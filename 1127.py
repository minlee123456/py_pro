import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

dataraw = pd.read_csv('data/BTC-USD.csv'
                      , index_col='Date'
                      , parse_dates=['Date'])
dataset = pd.DataFrame(dataraw['Close'])

scaler = MinMaxScaler()
dataset_norm = dataset.copy()
dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])

totaldata = dataset.values
totaldatatrain = int(len(totaldata) * 0.7)
totaldataval = int(len(totaldata) * 0.1)
training_set = dataset_norm[0:totaldatatrain]
val_set = dataset_norm[totaldatatrain:totaldatatrain + totaldataval]
test_set = dataset_norm[totaldatatrain + totaldataval:]


def create_sliding_windows(data, len_data, lag):
    x, y = [], []
    for i in range(lag, len_data):
        x.append(data[i - lag:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


lag = 2


def gen_data(dataset, lag):
    dataset = np.array(dataset)
    x, y = create_sliding_windows(dataset, len(dataset), lag)
    x, y = (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))
    return x, y


x_train, y_train = gen_data(training_set, lag)
x_val, y_val = gen_data(val_set, lag)
x_test, y_test = gen_data(test_set, lag)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.lstm(x)
        out = self.fc(h[-1])
        return out

input_size = 1
hidden_size = 64
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 1000
batch_size = 256

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train.unsqueeze(-1))
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        outputs = model(x_val.unsqueeze(-1))
        val_loss = criterion(outputs.squeeze(), y_val)
        print(f"Epoch {epoch + 1}/{epochs},"
              f"Loss : {loss.item():.4f},"
              f"Val Loss: {val_loss.item():.4f}")

model.eval()
outputs = model(x_test.unsqueeze(-1)).detach().numpy()

outputs_invert_norm = scaler.inverse_transform(outputs.squeeze(0))


def rmse(x, y):
    return np.sqrt(np.mean((y - x) ** 2))


def mape(x, y):
    return np.mean(np.abs((x - y) / x)) * 100


dataset = dataset['Close'][totaldatatrain + totaldataval + lag:].values
print('RMSE:', rmse(dataset, outputs_invert_norm))
print('MAPE:', mape(dataset, outputs_invert_norm))

plt.figure(figsize=(10, 4))
plt.plot(dataset, label="Data test", color='red')
plt.plot(outputs_invert_norm,label='Predicted',color='blue')
plt.title('bt')
plt.xlabel('Day')
plt.ylabel('price')
plt.legend()
plt.show()
