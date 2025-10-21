import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, embedding_matrix):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        avg_pool = (embedded.sum(dim=1) / lengths.unsqueeze(1).to(embedded.sum(dim=1).device)).float()
        out = F.sigmoid(self.fc1(avg_pool))
        out = F.sigmoid(self.fc2(out))
        return self.fc3(out)
    