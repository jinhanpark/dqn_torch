import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):

    def __init__(self, config):
        super(ConvNet, self).__init__()
        cin = config.history_length

        conv_layers = []
        bn_layers = []
        fc_layers = []

        convw = config.in_width
        convh = config.in_height
        num_filters = 1
        for i, (num_filters, kernel_size,
                stride) in enumerate(config.cnn_archi):
            layer = nn.Conv2d(cin,
                              num_filters,
                              kernel_size=kernel_size,
                              stride=stride)
            conv_layers.append(layer)
            bn_layers.append(nn.BatchNorm2d(num_filters))
            cin = num_filters
            convw = (convw - (kernel_size - 1) - 1) // stride + 1
            convh = (convh - (kernel_size - 1) - 1) // stride + 1

        linear_input_size = convw * convh * num_filters
        for i, dim in enumerate(config.fc_archi):
            fc_layers.append(nn.Linear(linear_input_size, dim))
            linear_input_size = dim

        self.head = nn.Linear(linear_input_size, config.output_size)

        self.layers = nn.ModuleList(conv_layers + bn_layers + fc_layers)
        self.conv_layers = conv_layers
        self.bn_layers = bn_layers
        self.fc_layers = fc_layers

    def forward(self, x):
        # conv layers
        for i in range(len(self.conv_layers)):
            x = F.relu(self.bn_layers[i](self.conv_layers[i](x)))

        # flatten layer
        x = x.view(x.size(0), -1)

        # fc layers
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)

        # output layer
        return self.head(x)