import torch
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt

class PTDeep(nn.Module):
    def __init__(self, config):
        super(PTDeep, self).__init__()
        self.num_layers = len(config)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(1, self.num_layers):
            self.weights.append(nn.Parameter(torch.randn(config[i-1], config[i])))
            self.biases.append(nn.Parameter(torch.randn(1, config[i])))

    def forward(self, X):
        
        h = X.view(X.size(0), -1)
        for i in range(self.num_layers - 1):
            h = torch.mm(h, self.weights[i]) + self.biases[i]
            if i < self.num_layers - 2:
                h = torch.sigmoid(h)
                #h=torch.relu(h)
        y_predictions = torch.softmax(h, dim=1)
        return y_predictions
    
    def get_loss(self, X, Yoh_):
        criterion = nn.CrossEntropyLoss()
        results = self.forward(X)
        Yoh_t = torch.tensor(Yoh_, dtype=torch.float32)
        _, class_labels = torch.max(Yoh_t, 1)
        loss = criterion(results, class_labels)
        return loss
    
    def count_params(self):
        total_param_count = 0
        for name, param in self.named_parameters():
            print("Name of the parameter: {}, shape of the parameter: {}".format(name, param.shape))
            total_param_count += np.prod(param.shape)
        print("Total number of parameters: {}".format(total_param_count))

    
def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
  optimizer = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

  for i in range(param_niter):
    loss = model.get_loss(X, Yoh_)
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

if __name__ == "__main__":
    np.random.seed(100)

    X,Y_true = data.sample_gmm_2d(4,2,40)
    Yoh_ = data.class_to_onehot(Y_true)
    X_t = torch.tensor(X, dtype=torch.float32)
    Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)
    
    dimensions = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    for dims in dimensions:
        ptd = PTDeep(dims)
        ptd.count_params()
        train(ptd, X_t, Yoh_, 10000, 0.1, 0.0001)
        Y_predict = eval(ptd, X_t)
        accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_predict, Y_true)

        print(accuracy)
        print(precision)
        print(confusion_matrix)

        rect=(np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(lambda X_t: eval(ptd, X_t), rect, offset=0)
        data.graph_data(X, Y_true, Y_predict, special=[])
        plt.show()
    
    
    
    X,Y_true = data.sample_gmm_2d(6,2,10)
    Yoh_ = data.class_to_onehot(Y_true)
    X_t = torch.tensor(X, dtype=torch.float32)
    Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)

    dimensions = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    for dims in dimensions:
        ptd = PTDeep(dims)
        ptd.count_params()
        train(ptd, X_t, Yoh_, 10000, 0.5, 0.001)
        Y_predict = eval(ptd, X_t)
        
        accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_predict, Y_true)
        _, recall, _ = data.eval_perf_binary(Y_predict, Y_true)
        average_precision = data.eval_AP(Y_predict)

        print("Accuracy:{}".format(accuracy))
        print("Precision:{}".format(precision))
        print("CM:{}".format(confusion_matrix))
        print("Recall:{}".format(recall))
        print("Average precision:{}".format(average_precision))

        rect=(np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(lambda X_t: eval(ptd, X_t), rect, offset=0)
        data.graph_data(X, Y_true, Y_predict, special=[])
        plt.show()
    
