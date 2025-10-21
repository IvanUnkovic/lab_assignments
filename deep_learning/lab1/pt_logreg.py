import data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
  def __init__(self, D, C):
    super(PTLogreg, self).__init__()
    self.W = nn.Parameter(torch.randn(D, C))
    self.b = nn.Parameter(torch.randn(C))

  def forward(self, X):
    s1 = torch.mm(X, self.W) + self.b
    h1 = torch.softmax(s1, dim=1)
    return h1

  def get_loss(self, X, Yoh_):
    
    criterion = nn.CrossEntropyLoss()
    results = self.forward(X)
    loss = criterion(results, Yoh_)
    return loss
    
  def get_reg_factor(self, param_lambda):
     return torch.norm(self.W) * param_lambda
     

def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
  optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

  print("Hyperparameters: param_delta={}, param_lambda={}".format(param_delta, param_lambda))

  for i in range(param_niter):
    loss = model.get_loss(X, Yoh_) + model.get_reg_factor(param_lambda)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    if i%1000==0:
        print("At {} the loss is :{}".format(i, loss))
    
def eval(model, X):
   with torch.no_grad():
      X_tensor = torch.tensor(X, dtype=torch.float32).clone().detach()
      result = model.forward(X_tensor)
      _, predicted_classes = torch.max(result, dim=1)
   return predicted_classes.numpy()

def find_h_l(results):
   sorted_results = sorted(results, key=lambda x: x[1])
   return sorted_results[-1], sorted_results[0]

   

if __name__ == "__main__":
    np.random.seed(100)
    X,Y_true = data.sample_gauss_2d(3, 100)
    Yoh_ = data.class_to_onehot(Y_true)
    X_t = torch.tensor(X, dtype=torch.float32)
    Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)


    param_deltas = [0.5, 0.1, 0.05, 0.01, 0.75]
    param_lambdas = [0.001, 0.01, 0.1]
    results = []
    for pd in param_deltas:
       for pl in param_lambdas:
            ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
            train(ptlr, X_t, Yoh_, 10000, pd, pl)
            Y_predict = eval(ptlr, X_t)
            accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_predict, Y_true)
            results.append(((pd, pl), accuracy, precision, confusion_matrix, ptlr))

          
    highest, lowest = find_h_l(results)

    print("Best results, param_delta:{}, param_lambda:{}".format(highest[0][0], highest[0][1]))
    print("Accuracy:{}".format(highest[1]))
    print("Precision:{}".format(highest[2]))
    print("CM:{}".format(highest[3]))
    
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda X_t: eval(highest[4], X_t), rect, offset=0)
    data.graph_data(X, Y_true, Y_predict, special=[])
    plt.show()

    print("Worst results, param_delta:{}, param_lambda:{}".format(lowest[0][0], lowest[0][1]))
    print("Accuracy:{}".format(lowest[1]))
    print("Precision:{}".format(lowest[2]))
    print("CM:{}".format(lowest[3]))
    
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda X_t: eval(lowest[4], X_t), rect, offset=0)
    data.graph_data(X, Y_true, Y_predict, special=[])
    plt.show()
    
    
