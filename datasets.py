import os
import json
import numpy as np
from torch.utils.data import Dataset


class TextOnlyDataset(Dataset):
    def __init__(self, data_path, max_length=100):
        self.embs, self.labels = self._load(data_path)
        self.max_length = max_length
    
    def _load(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        embs = data['array1']
        labels = data['array2']
        return embs, labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        emb = self.embs[idx]
        label = self.labels[idx]
        return emb, label


class AudioOnlyDataset(Dataset):
    def __init__(self, data_path):
        self.embs, self.labels = self._load(data_path)
    
    def _load(self, data_path):
        print(f'Loading data from {data_path}...')
        with open(data_path, "r") as f:
            loaded_data = json.load(f)
        loaded_audio_embs = [item["audio_emb"] for item in loaded_data]
        loaded_labels = [item["label"] for item in loaded_data]
        return loaded_audio_embs, loaded_labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        emb = self.embs[idx]
        label = self.labels[idx]
        return emb, label


if __name__ == '__main__':
    data_path = '/projects/0/prjs0864/tts/Sarcasm/podcast/processed_data/bert_embeddings_384_labels.npz'
    dataset = TextOnlyDataset(data_path)
    print(f'Dataset length: {len(dataset)}')
    print(f'Dataset sample: {dataset[0][0].shape}')
    print(f'Dataset sample: {dataset[0][1]}')
        
    
