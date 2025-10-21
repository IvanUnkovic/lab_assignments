import data
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class FCANN2():
    def __init__(self, x_size, z_size, y_size):
        self.W1 = np.random.normal(size=(x_size, z_size))
        self.W2 = np.random.normal(size=(z_size, y_size))
        self.b1 = np.random.normal(size=(z_size,))
        self.b2 = np.random.normal(size=(y_size,))

    def forward(self, x):
        self.s1 = x@self.W1 + self.b1
        self.h1 = np.maximum(0,self.s1)
        self.s2 = self.h1@self.W2 + self.b2
        self.h2 = softmax(self.s2)
        return self.h2
    
    def update_weights(self, X, Y_true, Y_preds, param_delta):
        grad_s = Y_preds - Y_true
        grad_w2 = self.h1.T@grad_s/X.shape[0]
        grad_b2 = np.sum(grad_s, axis=0)/X.shape[0]
        grad_h1 = grad_s@self.W2.T
        grad_s1 = grad_h1*(self.s1 > 0)
        grad_w1 = X.T@grad_s1/X.shape[0]
        grad_b1 = np.sum(grad_h1, axis=0)/X.shape[0]

        self.W1 -= param_delta * grad_w1
        self.b1 -= param_delta * grad_b1
        self.W2 -= param_delta * grad_w2
        self.b2 -= param_delta * grad_b2

    
def fcann2_train(X, Y, x_size, z_size, y_size, param_niter, param_delta, param_lambda):
    Y = data.class_to_onehot(Y)
    my_model = FCANN2(x_size, z_size, y_size)

    for iter in range(param_niter):
        results = my_model.forward(X)
        empirical_error = - np.sum(Y * np.log(results)) / X.shape[0]
        reg_term = (param_lambda / 2) * (np.sum(my_model.W1**2) + np.sum(my_model.W2**2))
        total_error = empirical_error + reg_term
        print("Iteration {}: {}".format(iter, total_error)) if iter%1000==0 else None

        my_model.update_weights(X,Y, results, param_delta)
        
    return my_model

def fcann2_classify(X, model):
    results = model.forward(X)
    class_labels = np.zeros(results.shape[0], dtype=int)
    for i, result in enumerate(results):
        class_labels[i] = np.argmax(result)
    return class_labels


X,Y_true = data.sample_gmm_2d(6, 2, 10)
my_model = fcann2_train(X, Y_true, 2, 5, 2, 10**4, 0.05, 0.001)

Y_predict = fcann2_classify(X, my_model)
accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_predict, Y_true)
_, recall, _ = data.eval_perf_binary(Y_predict, Y_true)
average_precision = data.eval_AP(Y_predict)
print("Accuracy:{}".format(accuracy))
print("Precision:{}".format(precision))
print("CM:{}".format(confusion_matrix))
print("Recall:{}".format(recall))
print("Average precision:{}".format(average_precision))


rect=(np.min(X, axis=0), np.max(X, axis=0))
data.graph_surface(lambda X: fcann2_classify(X, my_model), rect, offset=0)
data.graph_data(X, Y_true, Y_predict, special=[])
plt.show()
