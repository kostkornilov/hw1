from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 7 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 64, 5, 1, 5 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 3 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 2, 3 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100, bias=False)
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.head(x)
        return x
