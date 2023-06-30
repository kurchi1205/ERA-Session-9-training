import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_channels = 10
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), groups=3, padding=1, bias=False), #rin = 1  rout = 3
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=3, out_channels=self.num_channels, kernel_size=(1,1), bias=False), #rin = 3  rout = 3
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(3,3), dilation=2, bias=False), #rin = 3  rout = 7
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels),
            nn.Dropout(0.05),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels , out_channels=self.num_channels * 2, kernel_size=(3,3), padding=1, bias=False), #rin = 7  rout = 9
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 2),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels * 2, kernel_size=(3,3), padding=1, bias=False), #rin = 9  rout = 13
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 2),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels * 2, kernel_size=(3,3), stride=2, bias=False, padding=1), #rin = 13  rout = 15
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 2),
            nn.Dropout(0.05),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels * 4, kernel_size=(3,3), padding=1, bias=False),  #rin = 15  rout = 19
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 4),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels * 4, out_channels=self.num_channels * 4, kernel_size=(3,3),padding=1, bias=False), #rin = 19  rout = 23
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 4),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels * 4, out_channels=self.num_channels * 4, kernel_size=(3,3), stride=2, bias=False, padding=1), #rin = 23  rout = 27
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 4),
            nn.Dropout(0.05),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels * 4, out_channels=self.num_channels * 8, kernel_size=(3,3), padding=1, bias=False),  #rin = 27  rout = 35
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 8),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels * 8, out_channels=self.num_channels * 8, kernel_size=(3,3), padding=1, bias=False), #rin = 35  rout = 39
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 8),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=self.num_channels * 8, out_channels=self.num_channels * 8, kernel_size=(3,3), dilation=2, bias=False), #rin = 39  rout = 55
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels * 8),
            nn.Dropout(0.02),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels * 8, out_channels=10, kernel_size=(1,1))
        )
    

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
      