import os
import torch
from torch.utils.data import DataLoader

from datasets import TextOnlyDataset
from models import TextOnlyModel, ResidualTextModel, AttentionTextModel, HybridAttentionModel
from utils import cal_f1_score


def trainer(model, train_loader, val_loader, num_epochs, device):
    '''
    Train the model on the training set and evaluate on the validation set.'''
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    early_stopping = 0
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (text_inputs, labels) in enumerate(train_loader):
            text_inputs = text_inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(text_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                f1, precision, recall, accuracy = cal_f1_score(logits, labels)
                if f1 > best_f1:
                    best_f1 = f1
                    early_stopping = 0
                print(f'Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, accuracy: {accuracy:.4f}')
        val_loss, f1, precision, recall, accuracy = evaluate(model, criterion, val_loader, device)
        print(f'Validation loss: {val_loss:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, accuracy: {accuracy:.4f}')
        print('-----------------------------------')


def evaluate(model, criterion, val_loader, device):
    '''
    Evaluate the model on the validation set.'''
    model.eval()
    total_loss = 0
    total_samples = 0
    all_labels = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    with torch.no_grad():
        for text_inputs, labels in val_loader:
            text_inputs = text_inputs.to(device)
            labels = labels.to(device)
            logits = model(text_inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * text_inputs.size(0)
            total_samples += text_inputs.size(0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_preds = torch.cat((all_preds, logits), dim=0)
        f1, precision, recall, accuracy = cal_f1_score(all_preds, all_labels)
    return total_loss / total_samples, f1, precision, recall, accuracy


def train_text_only_model():
    '''
    Train a text-only model on the podcast dataset.'''
    text_dim = 384
    data_path = f'/projects/0/prjs0864/tts/Sarcasm/podcast/processed_data/bert_embeddings_{text_dim}_labels.npz'
    dataset = TextOnlyDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # model = TextOnlyModel(text_dim=text_dim, hidden_dim=512, num_classes=2, dropout=0.5)
    # model = ResidualTextModel(text_dim=text_dim, hidden_dim=512, num_classes=2, dropout=0.5)
    model = AttentionTextModel(text_dim=text_dim, hidden_dim=512, num_classes=2, dropout=0.5)
    # model = HybridAttentionModel(text_dim=text_dim, hidden_dim=512, num_classes=2, dropout=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer(model, train_loader, val_loader, num_epochs=10, device=device)


def main():
    train_text_only_model()


if __name__ == '__main__':
    main()