from nlpdataset import NLPDataset
from vocab import Vocab
import embeddings
import batch_implementation
from torch.utils.data import DataLoader
from baseline import Baseline
import model_utilities
import torch
import torch.nn as nn
import torch.optim as optim

seed = 2052022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

lr=1e-3
epochs = 5

train_set = NLPDataset.read_file('sst_train_raw.csv')
valid_set = NLPDataset.read_file('sst_valid_raw.csv')
test_set = NLPDataset.read_file('sst_test_raw.csv')

text_vocab = Vocab.build_vocab(train_set.instances, isLabel=False)
label_vocab = Vocab.build_vocab(train_set.instances, isLabel=True)

def wrapper(batch):
    return batch_implementation.pad_collate_fn(batch, pad_index=0, text_vocab=text_vocab, label_vocab=label_vocab)

train_dataloader = DataLoader(dataset=train_set, batch_size=10, shuffle=True, collate_fn=wrapper)
valid_dataloader = DataLoader(dataset=valid_set, batch_size=32, shuffle=True, collate_fn=wrapper)
test_dataloader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, collate_fn=wrapper)

embeddings_dict = embeddings.load_embeddings("sst_glove_6b_300d.txt")
embeddings_matrix = embeddings.matrix(text_vocab, embeddings_dict, no_normal=False)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model = Baseline(embedding_matrix=embeddings_matrix).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(my_model.parameters(), lr=lr)

for epoch in range(epochs):
    train_loss = model_utilities.train(my_model, train_dataloader, optimizer, criterion, device)
    valid_loss, valid_accuracy, valid_f1, valid_conf_matrix = model_utilities.evaluate(my_model, valid_dataloader, criterion, device)
    print("Epoch: {}".format(epoch+1))
    print("Train Loss: {}".format(train_loss))
    print("Validation Loss: {}, Accuracy: {}, F1 Score: {}".format(valid_loss, valid_accuracy, valid_f1))
    print("Confusion Matrix: {}".format(valid_conf_matrix))


test_loss, test_accuracy, test_f1, test_conf_matrix = model_utilities.evaluate(my_model, test_dataloader, criterion, device)
print("Test Loss: {}, Accuracy: {}, F1 Score: {}".format(test_loss, test_accuracy, test_f1))
print("Confusion Matrix: {}".format(test_conf_matrix))