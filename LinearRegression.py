# importing libraries
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import random_split

# generate random data
np.random.seed(0)
X = np.random.rand(500, 1) * 10
y = 2 * X + 3 + np.random.randn(500, 1)

# scatter plot
plt.scatter(X,y)
plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.grid(True, ls='--', alpha=0.2, color='grey')
plt.show()

# PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# splitting data into train and test
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size

train_dataset, test_dataset = random_split(
    list(zip(X_tensor, y_tensor)), [train_size, test_size]
)

# train and test set
X_train = torch.stack([x[0] for x in train_dataset])
y_train = torch.stack([x[1] for x in train_dataset])
X_test = torch.stack([x[0] for x in test_dataset])
y_test = torch.stack([x[1] for x in test_dataset])

# model for linear regression analysis
class Regression(nn.Module):
    def __init__(self, input_size, output_size):
        nn.Module.__init__(self) 
        self.linear = nn.Linear(input_size, output_size)
    # activation function 
    def forward(self, x):
        return self.linear(x)

# input size
input_size = 1  
# output size 
output_size = 1 
# creating an instance of Regression 
model = Regression(input_size, output_size)
# Mean square error
criterion = nn.MSELoss()
# learning rate
learning_rate = 0.001
# optimisation algortihms for convergence 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#number of iterations 
num_epochs = 200
# lsit to store test and train loss
train_loss_values = []  
test_loss_values = []   

for epoch in range(num_epochs):
    # zero the gradients
    optimizer.zero_grad()

    # forward pass, compute predicted y by passing X to the model
    y_train_pred = model(X_train)

    # compute loss
    train_loss = criterion(y_train_pred, y_train)
    
    # backward pass, compute gradient of the loss with respect to model parameters
    train_loss.backward()

    # update model parameters
    optimizer.step()

    # record the training loss
    train_loss_values.append(train_loss.item())

    # compute and record the test loss
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
        test_loss_values.append(test_loss.item())

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')


with torch.no_grad():
    # predictions for training and test set
    y_train_pred = model(X_train)
    y_test_pred = model(X_test)

    # metrics for training set
    train_mse = criterion(y_train_pred, y_train)
    train_rmse = torch.sqrt(train_mse)
    train_mae = torch.mean(torch.abs(y_train_pred - y_train))
    train_total_var = ((y_train - y_train.mean()) ** 2).sum()
    train_unexplained_var = ((y_train - y_train_pred) ** 2).sum()
    train_r_squared = 1 - train_unexplained_var / train_total_var

    # metrics for test set
    test_mse = criterion(y_test_pred, y_test)
    test_rmse = torch.sqrt(test_mse)
    test_mae = torch.mean(torch.abs(y_test_pred - y_test))
    test_total_var = ((y_test - y_test.mean()) ** 2).sum()
    test_unexplained_var = ((y_test - y_test_pred) ** 2).sum()
    test_r_squared = 1 - test_unexplained_var / test_total_var

# final metrics for both train and test sets
print(f'Final Train Metrics: MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R^2: {train_r_squared:.4f}')
print(f'Final Test Metrics: MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R^2: {test_r_squared:.4f}')

# predictions 
predicted = model(X_test).detach().numpy()

# scatter plot and best fit line
plt.scatter(X_test, y_test, label='Original data')
plt.plot(X_test, predicted, color='red', label='Fitted line')
plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.title('Linear Regression')
plt.grid(True, ls='--', alpha=0.2, color='grey')
plt.legend()
plt.show()

# loss plot
plt.figure(figsize=(10, 5))
plt.plot(train_loss_values, label='Training Loss')
plt.plot(test_loss_values, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.grid(True, ls='--', alpha=0.5, color='grey')
plt.legend()
plt.show()

## Author : Hemant Thapa
## Topic : Linear regression using neural nets (pytorch)
## Date: 13.02.2024
