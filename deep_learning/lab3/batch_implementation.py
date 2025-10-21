import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch, pad_index=0, text_vocab=None, label_vocab=None):
    texts, labels = zip(*batch)
    if text_vocab:
        texts = [text_vocab.encode(text).clone().detach() for text in texts]
    else:
        texts = [torch.tensor([int(token) for token in text]) for text in texts]
    if label_vocab:
        labels = label_vocab.encode(labels).clone().detach()
    else:
        labels = torch.tensor([int(label) for label in labels])

    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    return padded_texts, labels, lengths
