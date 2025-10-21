import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train(model, data_loader, optimizer, criterion, device, clip_value=0.25):
    model.train()
    total_loss = 0
    for _, batch in enumerate(data_loader):
        texts, labels, lengths = batch
        texts, labels = texts.to(device), labels.to(device).float()
        lengths = lengths.to(device).float()
        
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels.view(-1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
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
            texts, labels = texts.to(device), labels.to(device).float()
            lengths = lengths.to(device).float()
            
            logits = model(texts)
            loss = criterion(logits, labels.view(-1, 1))
            total_loss += loss.item()
            
            predictions = torch.round(torch.sigmoid(logits)).cpu().numpy()
            dataset_labels.extend(labels.cpu().numpy())
            dataset_predictions.extend(predictions)
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(dataset_labels, dataset_predictions)
    f1_sc = f1_score(dataset_labels, dataset_predictions)
    conf_matrix = confusion_matrix(dataset_labels, dataset_predictions)

    
    return [avg_loss, accuracy, f1_sc, conf_matrix]
