import torch
import torch.nn as nn

class CoordAttention(nn.Module):
    def __init__(self, feature_size, coord_size=3):
        super(CoordAttention, self).__init__()
        self.fc1 = nn.Linear(coord_size, 128)
        self.fc2 = nn.Linear(128, feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, coords):
        # 将坐标通过全连接层映射为注意力权重
        #print(coords.shape)
        batch = features.size(0)
        attention_weights = nn.ReLU()(self.fc1(coords))
        #print(attention_weights.shape)
        attention_weights = self.sigmoid(self.fc2(attention_weights))
        #print(attention_weights.shape)
        #print(features.shape)
        # 使用注意力权重调整特征
        weighted_features = features * attention_weights.view(batch, features.size(1), 1, 1, 1)
        return weighted_features