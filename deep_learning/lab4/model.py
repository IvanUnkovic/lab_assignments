import torch
import torch.nn as nn

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k//2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.conv2 = _BNReluConv(emb_size, emb_size)
        self.conv3 = _BNReluConv(emb_size, emb_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_features(self, img):
        x = self.conv1(img)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative, margin=1.0):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        pos_dist = torch.norm(a_x - p_x, p=2, dim=1)
        neg_dist = torch.norm(a_x - n_x, p=2, dim=1)
        triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        loss = triplet_loss.mean()
        return loss
