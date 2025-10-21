import torch
import torchvision
import matplotlib.pyplot as plt
import pt_deep
import data

dataset_root = '/tmp/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

y_train_oh = data.class_to_onehot(y_train)

ptd = pt_deep.PTDeep([784,10])
pt_deep.train(ptd, x_train, y_train_oh, 10000, 0.1, 0.0001)
y_train_predict = pt_deep.eval(ptd, x_train)
accuracy, precision, confusion_matrix = data.eval_perf_multi(y_train_predict, y_train)

print(accuracy)
print(precision)
print(confusion_matrix)