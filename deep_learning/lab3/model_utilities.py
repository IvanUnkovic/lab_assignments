from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for _, batch in enumerate(data_loader):
        texts, labels, lengths = batch
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model.forward(texts, lengths)
        loss = criterion(logits, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    dataset_labels = []
    dataset_predictions = []
    
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            texts, labels, lengths = batch
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            logits = model(texts, lengths)
            loss = criterion(logits, labels.float().view(-1, 1))
            total_loss += loss.item()
            
            preds = torch.round(torch.sigmoid(logits)).cpu().numpy()
            dataset_labels.extend(labels.cpu().numpy())
            dataset_predictions.extend(preds)
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(dataset_labels, dataset_predictions)
    f1_sc = f1_score(dataset_labels, dataset_predictions)
    conf_matrix = confusion_matrix(dataset_labels, dataset_predictions)


    return [avg_loss, accuracy, f1_sc, conf_matrix]
