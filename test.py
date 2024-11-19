import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hidden_dim=16, num_layers=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transfoermer_encoder = encoder_layer
        self.positional_encoding = self.get_positional_encoding(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 8)
        self.relu3 = nn.ReLU
        self.fc4 = nn.Linear(8, output_dim)

    def get_positional_encoding(self, hidden_dim):
        # 位置编码
        positional_encoding = torch.zeros(10000, hidden_dim)
        position = torch.arange(0, 10000).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0).float()) / hidden_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        return positional_encoding

    def forward(self, src):
        src, _ = self.lstm(src)
        src = src + self.positional_encoding[:src.size(0), :]
        # Transformer处理
        src = src.permute(1, 0, 2)  # 调整维度以匹配Transformer的输入形状
        enc_output = self.transformer_encoders(src)
        # 只取最后一个时间步的输出
        last_output = enc_output[-1]
        # 通过额外的前馈网络层
        x = self.fc1(last_output)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        output = self.fc4(x)
        return output
