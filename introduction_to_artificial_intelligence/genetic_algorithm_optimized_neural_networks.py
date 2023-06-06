import numpy as np
import argparse
import random
import math

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--nn', type=str)
    parser.add_argument('--popsize', type=int)
    parser.add_argument('--elitism', type=int)
    parser.add_argument('--p', type=float)
    parser.add_argument('--K', type=float)
    parser.add_argument('--iter', type=int)
    return parser.parse_args()

def is_equal(a, b):
    return math.isclose(a, b, rel_tol=1e-7)

def find_n_smallest(dictionary, n):
    sorted_values = sorted(dictionary.values())
    smallest = []
    for i in range(0, n):
        smallest.append(sorted_values[i])
    new_dict = {}
    for element in smallest:
        for key in dictionary:
            if is_equal(dictionary[key], element):
                new_dict[key]=element
                break
    return new_dict

def read_file(train_file):
    train_output_input = {}
    length = 0
    with open(train_file, 'r') as file:
        lines = file.readlines()
        first_line = True
        for line in lines:
            without_comma = line.strip().split(",")
            if first_line:
                first_line = False
                length = len(without_comma[:-1])
                continue
            else:
                train_output_input[tuple(without_comma[:-1])]=without_comma[-1]
    return train_output_input, length

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crossing(new, old1, old2, config):
    new.weights1 = (old1.weights1 + old2.weights1) / 2
    new.weights2 = (old1.weights2 + old2.weights2) / 2
    new.biases1 = (old1.biases1 + old2.biases1) / 2
    new.biases2 = (old1.biases2 + old2.biases2) / 2
    if config=="5s5s":
        new.weights3 = (old1.weights3 + old2.weights3) / 2
        new.biases3 = (old1.biases3 + old2.biases3) / 2
    return

def mutate_matrix(mat, std_dev, mutation_prob):
    mutated_mat = mat.copy()
    for i in range(len(mutated_mat)):
        if np.random.rand() < mutation_prob:
            mutation = np.random.normal(0, std_dev)
            mutated_mat[i] += mutation
    return mutated_mat

class NN():
    def __init__(self, num_columns, config):
        if config=="5s":
            self.weights1 = np.random.normal(loc=0.0, scale=0.01, size=(5, num_columns))
            self.biases1 = np.random.normal(loc=0.0, scale=0.01, size=(5, num_columns))
            self.weights2 = np.random.normal(loc=0.0, scale=0.01, size=(1, 5))
            self.biases2 = np.random.normal(loc=0.0, scale=0.01, size=(1, 1))
        elif config=="20s":
            self.weights1 = np.random.normal(loc=0.0, scale=0.01, size=(20, num_columns))
            self.biases1 = np.random.normal(loc=0.0, scale=0.01, size=(20, num_columns))
            self.weights2 = np.random.normal(loc=0.0, scale=0.01, size=(1, 20))
            self.biases2 = np.random.normal(loc=0.0, scale=0.01, size=(1, 1))
        elif config=="5s5s":
            self.weights1 = np.random.normal(loc=0.0, scale=0.01, size=(5, num_columns))
            self.biases1 = np.random.normal(loc=0.0, scale=0.01, size=(5, num_columns))
            self.weights2 = np.random.normal(loc=0.0, scale=0.01, size=(5, 5))
            self.biases2 = np.random.normal(loc=0.0, scale=0.01, size=(5, 1))
            self.weights3 = np.random.normal(loc=0.0, scale=0.01, size=(1, 5))
            self.biases3 = np.random.normal(loc=0.0, scale=0.01, size=(1, 1))
        self.config = config

    def forward5s(self, X):
        if (self.config=="5s" or self.config=="20s"):
            y_values = {}
            for x in X:
                values = np.array(x).astype(float)
                y_matrix = np.multiply(self.weights1, values) + self.biases1
                sigmoid_matrix = sigmoid(y_matrix)
                result = np.matmul(self.weights2, sigmoid_matrix) + self.biases2
                y_values[x] = result[0,0]
            return y_values
        elif self.config=="5s5s":
            y_values = {}
            for x in X:
                values = np.array(x).astype(float)
                y1_matrix = np.multiply(self.weights1, values) + self.biases1
                sigmoid_matrix = sigmoid(y1_matrix)
                y2_matrix = np.multiply(self.weights2, sigmoid_matrix) + self.biases2
                sigmoid_matrix_2 = sigmoid(y2_matrix)
                result = np.matmul(self.weights3, sigmoid_matrix_2) + self.biases3
                y_values[x] = result[0,0]
            return y_values
    
    def calculate_standard_error(self, real_values, y_values):
        sum = 0
        for element in real_values:
            sum += (float(real_values[element]) - float(y_values[element]))**2
        return sum/len(y_values)

def sum_of_fits(set_of_networs):
    sum=0
    for network in set_of_networs:
        sum+=1/(set_of_networs[network])
    return sum

def main():
    args = parser()
    train_input_output, num_of_x = read_file(args.train)
    set_of_networks={}
    new_set_of_networks = {}
    for iter in range(0, int(args.iter)):
        if iter==0:
            for network in range(0, int(args.popsize)):
                nn = NN(num_of_x, args.nn)
                y_values = nn.forward5s(train_input_output.keys())
                err = nn.calculate_standard_error(train_input_output, y_values)
                set_of_networks[nn] = err
        else:
            new_set_of_networks = find_n_smallest(set_of_networks, args.elitism)
            probs = {}
            for i in range(0, (args.popsize-args.elitism)):
                for network in set_of_networks:
                    fit = 1/set_of_networks[network]
                    big_sum = sum_of_fits(set_of_networks)
                    probs[network] = fit/(big_sum)
                    sorted_probs = (sorted(probs.values(), reverse=True))
                cumulative_probs = [(sum(sorted_probs[:i+1]), sorted_probs[i]) for i in range(len(sorted_probs))]
                selected = []
                random_float = random.random()
                for i in range(0, len(cumulative_probs)):
                    if random_float<=cumulative_probs[i][0]:
                        #u select su upisani errori networka koje su odabrane
                        selected.append(1/(cumulative_probs[i][1] *(big_sum)))
                        selected.append(1/(cumulative_probs[i-1][1] * (big_sum)))
                        break
                selected_networks = []
                for network in set_of_networks:
                    if is_equal(set_of_networks[network], selected[0]):
                        selected_networks.append(network)
                    if is_equal(set_of_networks[network], selected[1]):
                        selected_networks.append(network)
                nn1 = selected_networks[0]
                nn2 = selected_networks[1]
                nn3 = NN(num_of_x, args.nn)
                crossing(nn3, nn1, nn2, args.nn)
                nn3.weights1 = mutate_matrix(nn3.weights1, args.K, args.p)
                nn3.weights2 = mutate_matrix(nn3.weights2, args.K, args.p)
                nn3.biases1 = mutate_matrix(nn3.biases1, args.K, args.p)
                nn3.biases2 = mutate_matrix(nn3.biases2, args.K, args.p)
                if args.nn=="5s5s":
                    nn3.weights3 = mutate_matrix(nn3.weights3, args.K, args.p)
                    nn3.biases3 = mutate_matrix(nn3.biases3, args.K, args.p)
                y_values = nn3.forward5s(train_input_output.keys())
                err = nn3.calculate_standard_error(train_input_output, y_values)
                new_set_of_networks[nn3] = err
            set_of_networks.clear()
            set_of_networks.update(new_set_of_networks)
             
        if(iter+1)%2000==0:
            print("[Train error @{}]: {}".format(str(iter+1), min(set_of_networks.values())))
    test_input_output, _ = read_file(args.test)
    errors = []
    for network in set_of_networks.keys():
        y_values = network.forward5s(test_input_output.keys())
        err = network.calculate_standard_error(test_input_output, y_values)
        errors.append(err)
    print("[Test error]: {}".format(min(errors)))

if __name__=='__main__':
    main()

