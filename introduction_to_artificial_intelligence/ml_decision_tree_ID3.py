import argparse
import sys
from math import log2

class ID3:
    def __init__(self):
        pass
    def fit(self, features, class_label, values):
        return ID3_algorithm(values, values, features, class_label)
                
    def predict(self, features, class_label, values, test_class_values, tree):
        predictions = []
        helper = {}
        for value_line in values:
            prediction = traverse_decision_tree(tree, features, value_line)
            if prediction==None:
                possible_predictions = []
                for test_class_value in test_class_values:
                    helper[test_class_value]=0
                for test_class_value in test_class_values:
                    helper[test_class_value]+=1
                max_value = max(helper.values())
                for value in helper:
                    if helper[value]==max_value:
                        possible_predictions.append(value)
                sorted_possibilities = sorted(possible_predictions)
                prediction = sorted_possibilities[0]
            predictions.append(prediction)
        return predictions
    
def traverse_decision_tree(tree, features, value_line):
    if(len(sys.argv)==4 and sys.argv[3]=="0"):
        return tree
    root_feature = list(tree.keys())[0]
    root_feature_index = features.index(root_feature)
    root_feature_value = value_line[root_feature_index]

    subtree = tree[root_feature]
    subsubtree = None
    for key in subtree:
        if key==root_feature_value:
            subsubtree = subtree[root_feature_value]
            break
    if isinstance(subsubtree, dict):
        return traverse_decision_tree(subsubtree, features, value_line)
    else:
        return subsubtree                  

def print_decision_tree(decision_tree, level=1, path=""):
    if isinstance(decision_tree, dict):
        for attribute, subtree in decision_tree.items():
            for value, sub_subtree in subtree.items():
                new_path = f"{path} {level}:{attribute}={value}"
                if isinstance(sub_subtree, dict):
                    print_decision_tree(sub_subtree, level + 1, new_path)
                else:
                    print(f"{new_path} {sub_subtree}")
    
def ID3_algorithm(values, parent_values, features, class_label, level=0):
    
    if len(sys.argv)==4:
        if(level >= int(sys.argv[3])):
            possible_classes={}
            max_classes = []
            for values_line in values:
                possible_classes[values_line[-1]]=0
            for values_line in values:
                possible_classes[values_line[-1]]+=1
            max_value = max(possible_classes.values())
            for possible_class in possible_classes:
                if possible_classes[possible_class]==max_value:
                    max_classes.append(possible_class)
            return sorted(max_classes)[0]

    if(len(values)==0):
        parent_values_class_labels = {}
        for parent_values_line in parent_values:
            parent_values_class_labels[parent_values_line[-1]]=0
        for parent_values_line in parent_values:
            for label in parent_values_class_labels:
                if(label==parent_values_line[-1]):
                    parent_values_class_labels[label]+=1
        max_value = max(parent_values_class_labels.values())
        for label in parent_values_class_labels:
            if parent_values_class_labels[label]==max_value:
                max_class_label = label
                break
        return max_class_label
    values_class_labels = {}
    for values_line in values:
        values_class_labels[values_line[-1]]=0
    for values_line in values:
        for label in values_class_labels:
            if label==values_line[-1]:
                values_class_labels[label]+=1
    max_value = max(values_class_labels.values())
    for label in values_class_labels:
        if values_class_labels[label]==max_value:
            max_class_label = label
            break
    sub_values = []
    for value_line in values:
        if value_line[-1]==max_class_label:
            sub_values.append(value_line)
    if(len(features)==0 or values == sub_values):
        return max_class_label
    IGs = {}
    IGoutput = ""
    for feature in features:
        index = features.index(feature)
        IGs[feature] = IG(values, index)
        IGoutput+="IG({})={} ".format(feature, IGs[feature])
    print(IGoutput)
    max_value = max(IGs.values())
    for feature in IGs:
        if IGs[feature]==max_value:
            max_feature = feature
            break
    decision_tree = {}
    decision_tree[max_feature] = {}
    max_index = features.index(max_feature)
    feature_values = set()
    for value_line in values:
        feature_value = value_line[max_index]
        feature_values.add(feature_value)
    for feature_value in feature_values:
        new_values = []  
        for value_line in values:
            if value_line[max_index]==feature_value:
                new_values.append(value_line[:max_index] + value_line[max_index+ 1:])
        new_features = features[:max_index] + features[max_index + 1:]
        decision_tree[max_feature][feature_value] = ID3_algorithm(new_values, parent_values, new_features, class_label, level + 1)
    return decision_tree
    
def IG(values, index):
    K = set()
    numbers = {}
    for value_line in values:
        K.add(value_line[-1])
    for class_label_value in K:
        numbers[class_label_value]=0
    for value_line in values:
        for class_label_value in K:
            if class_label_value==value_line[-1]:
                numbers[class_label_value]+=1
    entropy = E(K, numbers)
    possible_values = {}
    total = 0
    for value_line in values:
        some_value = value_line[index]
        possible_values[some_value]=0
    for value_line in values:
        some_value = value_line[index]
        possible_values[some_value]+=1
    for possible_value in possible_values.values():
        total += possible_value
    sum = 0
    for possible_value in possible_values:
        for class_label_value in K:
            numbers[class_label_value]=0
        for value_line in values:
            if possible_value==value_line[index]:
                numbers[value_line[-1]]+=1
        sub_entropy = E(K, numbers)
        sum += (possible_values[possible_value]/total)*sub_entropy
    return entropy-sum    

def E(K, numbers):
    sum = 0
    total = 0
    for class_label_value in K:
        total += numbers[class_label_value]
    for class_label_value in K:
        if(numbers[class_label_value]==0):
            sum += 0
        else:
            sum += (numbers[class_label_value]/total)*log2(numbers[class_label_value]/total)
    return sum * (-1)

def load_dataset(dataset_path):
    f = open(dataset_path, 'r', encoding='utf-8')
    lines = f.readlines()
    isFirst = True
    values = []
    for line in lines:
        if isFirst:
            features = line.strip().split(",")[:-1]
            class_label = line.strip().split(",")[-1]
            isFirst = False
        else:
            split_line = line.strip().split(",")
            values.append(split_line)
    f.close()
    return features, class_label, values

def main():
    if(len(sys.argv)!=3 and len(sys.argv)!=4):
        sys.exit()
    train_dataset_path = sys.argv[1]
    test_dataset_path = sys.argv[2]
    features, class_label, values = load_dataset(train_dataset_path)
    test_features, test_class_label, test_values = load_dataset(test_dataset_path)
    new_test_values = []
    for test_value_line in test_values:
        new_test_values.append(test_value_line[:-1])
    test_class_values = []
    for test_value_line in test_values:
        test_class_values.append(test_value_line[-1])
    model = ID3()
    tree = model.fit(features, class_label, values)
    print(tree)
    predictions = model.predict(test_features, test_class_label, new_test_values, test_class_values, tree)
    predictions_output = "[PREDICTIONS]:"
    for prediction in predictions:
        predictions_output+=" {}".format(prediction)
    print("[BRANCHES]:")
    print_decision_tree(tree)
    print(predictions_output)
    counter = 0
    for i in range(0, len(test_class_values)):
        if(test_class_values[i]==predictions[i]):
            counter+=1
    accuracy = counter/len(test_class_values)
    rounded = "{:.5f}".format(accuracy).rstrip('.')
    print("[ACCURACY]: {}".format(rounded))
    unique_values = sorted(set(predictions + test_class_values))
    num_classes = len(unique_values)
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, act in zip(predictions, test_class_values):
        pred_index = value_to_index[pred]
        act_index = value_to_index[act]
        confusion_matrix[act_index][pred_index] += 1
    print("[CONFUSION_MATRIX]:")
    for row in confusion_matrix:
        row_values = " ".join(map(str, row))
        print(row_values)
    
if __name__=="__main__":
    main()
