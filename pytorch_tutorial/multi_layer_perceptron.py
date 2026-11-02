from torch import nn
import torch

# 一个具有两个隐藏层的多层感知机
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            # 第一个隐藏层
            nn.Linear(input_size, 30),
            nn.ReLU(),

            # 第二个隐藏层
            nn.Linear(30, 20),
            nn.ReLU(),

            # 输出层
            nn.Linear(20, output_size),
        )

    def forward(self, x):
        # 最后一层的输出称为 logits
        logits = self.layers(x)
        return logits

model = NeuralNetwork(50, 3)
# print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total number of trainable model parameters:", num_params)

# print(model.layers[0].weight.shape)

torch.manual_seed(123)
X = torch.rand((1,50))
out = model(X)
print(out)