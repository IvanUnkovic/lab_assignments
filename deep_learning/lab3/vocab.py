import torch

class Vocab:
    def __init__(self, frequencies, isLabel, max_size=-1, min_freq=1):
        if isLabel:
            self.itos = []
            self.stoi = {}
            self.isLabel = True
            self.frequencies = frequencies

            sorted_frequencies = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)

            for token, freq in sorted_frequencies:
                if freq >= min_freq:
                    if (max_size == -1 or len(self.itos) < max_size):
                        self.stoi[token] = len(self.itos)
                        self.itos.append(token)
        else:
            self.stoi = {"<PAD>": 0, "<UNK>": 1}
            self.itos = ["<PAD>", "<UNK>"]
            self.frequencies = frequencies
            self.isLabel = False
            
            sorted_frequencies = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)

            for token, freq in sorted_frequencies:
                if freq >= min_freq:
                    if (max_size == -1 or len(self.itos) < max_size):
                        self.stoi[token] = len(self.itos)
                        self.itos.append(token)

    def encode(self, tokens):
        if self.isLabel:
            return torch.tensor([self.stoi[token] for token in tokens])
        else:
            return torch.tensor([self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens])
    
    @staticmethod
    def build_vocab(instances, isLabel, max_size=-1, min_freq=0):
        frequencies = {}
        if isLabel:
            for instance in instances:
                if instance.label in frequencies:
                    frequencies[instance.label]+=1
                else:
                    frequencies[instance.label] = 1
        else:
            for instance in instances:
                for token in instance.text:
                    if token in frequencies:
                        frequencies[token] += 1
                    else:
                        frequencies[token] = 1
        return Vocab(frequencies, isLabel, max_size, min_freq)
