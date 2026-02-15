
import my_framework as nn

# --- Model Definition ---
class LeNet(nn.Module):
    def __init__(self,num_classes=10):
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
        self.fc3 = nn.Linear(84, num_classes)

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

   # Simple CNN with 3 convolution layers and 2 Fully connected layers

class SimpleCNN3(nn.Module):
    def __init__(self, num_classes=10):
        # Conv Layer 1: 3 → 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # downsample 32x32 → 16x16

        # Conv Layer 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # downsample 16x16 → 8x8

        # Conv Layer 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # downsample 8x8 → 4x4

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.relu1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.relu2(x); x = self.pool2(x)
        x = self.conv3(x); x = self.relu3(x); x = self.pool3(x)

        # Flatten
        x.shape = (x.shape[0], 128 * 4 * 4)

        # Fully connected layers
        x = self.fc1(x); x = self.relu4(x)
        x = self.fc2(x)
        return x

    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() +
                self.conv3.parameters() + self.fc1.parameters() +
                self.fc2.parameters())



# Mobile Net architecture
# --- Depthwise Separable Convolution Block ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        # Depthwise convolution: one filter per channel
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()
        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x); x = self.relu1(x)
        x = self.pointwise(x); x = self.relu2(x)
        return x

    def parameters(self):
        return self.depthwise.parameters() + self.pointwise.parameters()


# --- MobileNet (simplified for CIFAR-100) ---
class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        # Initial standard conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Depthwise separable blocks
        self.dsconv1 = DepthwiseSeparableConv(32, 64, stride=1)
        self.dsconv2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.dsconv3 = DepthwiseSeparableConv(128, 128, stride=1)
        self.dsconv4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.dsconv5 = DepthwiseSeparableConv(256, 256, stride=1)
        self.dsconv6 = DepthwiseSeparableConv(256, 512, stride=2)

        # A few more depthwise separable blocks
        self.dsconv7 = DepthwiseSeparableConv(512, 512, stride=1)
        self.dsconv8 = DepthwiseSeparableConv(512, 512, stride=1)

        # Global average pooling (simplified as MaxPool2d here)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=1)

        # Final classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.relu1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)
        x = self.dsconv5(x)
        x = self.dsconv6(x)
        x = self.dsconv7(x)
        x = self.dsconv8(x)
        x = self.pool(x)
        x.shape = (x.shape[0], 512)  # flatten
        x = self.fc(x)
        return x

    def parameters(self):
        params = self.conv1.parameters()
        params += (self.dsconv1.parameters() + self.dsconv2.parameters() +
                   self.dsconv3.parameters() + self.dsconv4.parameters() +
                   self.dsconv5.parameters() + self.dsconv6.parameters() +
                   self.dsconv7.parameters() + self.dsconv8.parameters() +
                   self.fc.parameters())
        return params


