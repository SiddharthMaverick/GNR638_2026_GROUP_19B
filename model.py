
import my_framework as nn
from train import CLASSES
# --- Model Definition ---
class LeNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=0) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, CLASSES)

    def forward(self, x):
        x = self.conv1(x); x = self.relu1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.relu2(x); x = self.pool2(x)
        x.shape = (x.shape[0], 16 * 5 * 5) 
        x = self.fc1(x); x = self.relu3(x)
        x = self.fc2(x); x = self.relu4(x)
        x = self.fc3(x)
        return x
        
    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() + 
                self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters())
