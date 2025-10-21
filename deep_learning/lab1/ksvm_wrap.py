from sklearn.svm import SVC
import numpy as np
import data
import matplotlib.pyplot as plt

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm = SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.decision_function(X)

    def support(self):
        return self.svm.support_
    
if __name__ == "__main__":

    X,Y_true = data.sample_gmm_2d(6,2,10)

    model = KSVMWrap(X, Y_true)
    Y_predictions = model.predict(X)
    support_vectors = model.support()

    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_predictions, Y_true)
    _, recall, _ = data.eval_perf_binary(Y_predictions, Y_true)
    average_precision = data.eval_AP(Y_predictions)

    print(accuracy)
    print(precision)
    print(recall)
    print(average_precision)
    print(confusion_matrix)

    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(model.get_scores, rect, offset=0)
    data.graph_data(X, Y_true, Y_predictions, special=[support_vectors])
    plt.show()
    
    