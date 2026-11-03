import torch.nn.functional as F
import torch

from pytorch_tutorial import multi_layer_perceptron
from data_loader import train_loader, test_loader
from pytorch_tutorial.data_loader import X_train

torch.manual_seed(123)
model = multi_layer_perceptron.NeuralNetwork(input_size=2, output_size=2)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

num_epochs = 3



for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        # 把上一轮梯度归0， 防止意外的梯度累计
        optimizer.zero_grad()
        loss.backward()
        # 优化器使用梯度更新模型的参数
        optimizer.step()

        # logging
        print(f"Epoch {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Loss: {loss:.3f}"
              )

model.eval()
with torch.no_grad():
    output = model(X_train)
print(output)
torch.set_printoptions(sci_mode=False)
probes = torch.softmax(output, dim=1)
print(probes)
prediction = torch.argmax(probes, dim=1)
print(prediction)

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total = 0.0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

            prediction = torch.argmax(logits, dim=1)
            compare = labels == prediction
            correct += torch.sum(compare)
            total += len(compare)

    return (correct / total).item()

print(compute_accuracy(model, test_loader))