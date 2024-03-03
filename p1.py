import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



def load_data(filename, n_examples, n_features):
  data_matrix = np.zeros((n_examples, n_features+1))
  labels = np.zeros(n_examples)
  with open(filename, 'r') as file:
    for i, line in enumerate(file):
      elements = line.strip().split()
      labels[i] = int(elements[0])

      for element in elements[1:]:
        index,value = element.split(":")
        index = int(index)-1
        data_matrix[i,index]= float(value)

      data_matrix[i,-1] = 1.0

  return labels, data_matrix


def logreg_loss(X,y,W):
  return np.log(1 + np.exp(-X.dot(W) * y)).mean()

def logreg_error(X,y,W):
  n = X.shape[0]
  preds = np.sign(np.dot(X,W))
  diff = np.sum(preds!=y)
  return diff/n

def logreg_grad(X,y,W):
  grad_denom = 1 + np.exp(-y * X.dot(W))
  grad_num = -X * y*np.exp(-y * X.dot(W))
  grad = grad_num / grad_denom
  return grad

def logreg_sgd(X,y, X_val,y_val,W0,alpha,sigma,T):
  metrics = {
        'epochs': [0],
        'training_loss': [logreg_loss(X, y, W0)],
        'training_error': [logreg_error(X, y, W0)],
        'validation_error': [logreg_error(X_val, y_val, W0)]
    }
  n = X.shape[0]
  W = W0.copy()
  print(f"original error: {logreg_error(X,y, W)}")
  print(f"original loss: {logreg_loss(X,y, W)}")
  for epoch in range(T):
    for j in range(n):
      example = np.random.randint(n)
      grad = logreg_grad(X[example],y[example],W)
      W -= alpha*(grad+ sigma*W)
    print(f"train error: {logreg_error(X,y, W)}")
    print(f"train loss: {logreg_loss(X,y, W)}")
    metrics['epochs'].append(epoch + 1)
    metrics['training_loss'].append(logreg_loss(X, y, W))
    metrics['training_error'].append(logreg_error(X, y, W))
    metrics['validation_error'].append(logreg_error(X_val, y_val, W))
  return W, metrics

def plot_metrics(metrics, filename='metrics_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epochs'], metrics['training_loss'], label='Training Loss')
    plt.plot(metrics['epochs'], metrics['training_error'], label='Training Error')
    plt.plot(metrics['epochs'], metrics['validation_error'], label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Training and Validation Metrics over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the figure to a file
    plt.close()

def load_data_pytorch(filename, n_examples, n_features):
    labels, data_matrix = load_data(filename, n_examples, n_features)
    labels = torch.tensor(labels, dtype=torch.float32)
    data_matrix = torch.tensor(data_matrix, dtype=torch.float32)
    return labels, data_matrix

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1) 
    
    def forward(self, x):
        return self.linear(x)
  
def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predicted = outputs.squeeze(-1) > 0  # Applying threshold at 0
            correct_predictions += (predicted.float() == labels).sum().item()
            total_predictions += labels.size(0)
    
    error_rate = 1 - correct_predictions / total_predictions
    return error_rate

def train(train_data, train_labels, val_loader, epochs, n_features, lr,batch_size=1, weight_decay=0.0001):
    model = LogisticRegressionModel(n_features+1)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr =lr, weight_decay=weight_decay)
    
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    metrics = {
        'epochs': [],
        'training_loss': [],
        'training_error': [],
        'validation_error': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print average loss per epoch
        average_loss = total_loss / len(train_loader)
        training_error = evaluate_model(model, train_loader)
        validation_error = evaluate_model(model, val_loader)
        metrics['epochs'].append(epoch + 1)
        metrics['training_loss'].append(average_loss)
        metrics['training_error'].append(training_error)
        metrics['validation_error'].append(validation_error)
        
    return model, metrics
      

if __name__ =="__main__":
  n_features=123
  #train_labels, train_data = load_data("a9a.train.txt", 32561, n_features)
  #val_labels, val_data = load_data("a9a.test.txt", 16281, n_features)
  #W0 = np.zeros(n_features + 1)
  #alphas = [0.01, 0.001, 0.0001]
  #sigmas = [0.0001, 0.001, 0.01]
  #for alpha in alphas:
    #for sigma in sigmas:
  #W,metrics = logreg_sgd(train_data, train_labels,val_data, val_labels,W0, 0.001, 0.0001,10)
  #plot_metrics(metrics, f"hpo_{alpha}_{sigma}.png")
  #print(f"Final Validation Error: {logreg_error(val_data, val_labels, W)}")
  #print(f"Final Validation Loss: {logreg_loss(val_data, val_labels, W)}")
  # Assuming `labels` are your original labels with values -1 or 1
  train_labels, train_data = load_data_pytorch("a9a.train.txt", 32561, n_features)
  val_labels, val_data = load_data_pytorch("a9a.test.txt", 16281, n_features)
  t_train_labels = (train_labels + 1) / 2  
  t_val_labels = (val_labels + 1) / 2  
  val_eval_loader = DataLoader(TensorDataset(val_data, t_val_labels), shuffle=False)
  model, metrics = train(train_data, t_train_labels, val_eval_loader, 10, 123,0.001,1, 0.0001)
  plot_metrics(metrics, "SGD_pytorch_plots.png")
  val_error = evaluate_model(model, val_eval_loader)
  print(f"Validation Error: {val_error}")


