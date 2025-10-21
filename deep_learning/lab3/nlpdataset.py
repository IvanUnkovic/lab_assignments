from instance import Instance
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index].text, self.instances[index].label
    
    @staticmethod
    def read_file(path):
        instances = []
        f = open(path, 'r')
        for line in f:
            parts = line.strip().split(", ")
            text = parts[0].split(" ")
            label = parts[1]
            instances.append(Instance(text, label))
        f.close()
        return NLPDataset(instances)