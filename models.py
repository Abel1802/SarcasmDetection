import torch
from torch import nn
from torch.nn import functional as F


class HybridAttentionModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_classes, dropout=0.3):
        super(HybridAttentionModel, self).__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_inputs):
        # Fully connected layer
        x = F.relu(self.fc1(text_inputs))  # [batch_size, hidden_dim]
        x = self.dropout(x)
        
        # Reshape for Conv1d
        x = x.unsqueeze(-1)  # 转为 [batch_size, hidden_dim, sequence_length]
        x_conv = F.relu(self.conv1(x))  # Apply Conv1d, 输出 [batch_size, hidden_dim, sequence_length]
        x_conv = x_conv.transpose(1, 2)  # 转回 [batch_size, sequence_length, hidden_dim]

        # Attention
        attn_output, _ = self.attention(x_conv, x_conv, x_conv)  # [batch_size, sequence_length, hidden_dim]
        x = attn_output + x_conv  # Residual connection

        # Final classification
        logits = self.fc2(x.mean(dim=1))  # 取 sequence 平均值作为分类输入
        return logits



class AttentionTextModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_classes, dropout=0.3):
        super(AttentionTextModel, self).__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_inputs):
        x = F.relu(self.fc1(text_inputs))
        x = self.dropout(x)
        # Add attention mechanism
        attn_output, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = attn_output.squeeze(0) + x  # Residual connection with attention
        logits = self.fc2(x)
        return logits


class ResidualTextModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_classes, dropout=0.3):
        super(ResidualTextModel, self).__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_inputs):
        x = F.relu(self.fc1(text_inputs))
        x = self.dropout(x)
        residual = x
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x += residual  # Residual connection
        logits = self.fc3(x)
        return logits


class TextOnlyModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_classes, dropout=0.3):
        super(TextOnlyModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, text_inputs):
        logits = self.classifier(text_inputs)
        return logits


class SarcasmDetectionModel(nn.Module):
    def __init__(self, text_model, audio_model, fusion_dim):
        super(SarcasmDetectionModel, self).__init__()
        self.text_model = text_model  # 预训练文本模型（如 BERT）
        self.audio_model = audio_model  # 预训练语音模型（如 Wav2vec 2.0）
        self.fusion_layer = nn.Linear(text_dim + audio_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # 假设二分类
        )

    def forward(self, text_inputs, audio_inputs):
        text_features = self.text_model(text_inputs)
        audio_features = self.audio_model(audio_inputs)
        combined_features = torch.cat((text_features, audio_features), dim=1)
        fused = self.fusion_layer(combined_features)
        logits = self.classifier(fused)
        return logits
