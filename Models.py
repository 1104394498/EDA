import torch
import torch.nn as nn
import torch.nn.functional as F


# generate adjacency matrix and adjacency sum (revealing how connected the pins in the generated map are)
class Generator(nn.Module):
    def __init__(self, input_x, input_y, conv_dims, adj_filter, input_dims=3, mul_num=5):
        super(Generator, self).__init__()
        self.input_x = input_x
        self.input_y = input_y
        self.mul_num = mul_num
        self.adj_filter = adj_filter
        layers = list()
        for dim0, dim1 in zip([input_dims] + conv_dims[:-1], conv_dims):
            layers.append(
                nn.Conv2d(in_channels=dim0, out_channels=dim1, kernel_size=3, padding=1, stride=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features=dim1))
            layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=conv_dims[-1], out_features=(input_x * input_y) ** 2)

    def forward(self, x):
        feature = self.features(x)
        out = F.adaptive_avg_pool2d(feature, output_size=(1, 1)).view(feature.size(0), -1)
        adj = self.fc(out)
        adj = adj.view(adj.shape[0], self.input_x * self.input_y, self.input_x * self.input_y)
        adj = torch.sigmoid(torch.matmul(adj, torch.transpose(adj, 1, 2))) * self.adj_filter
        return adj
