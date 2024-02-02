import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, num_bboxes=2):
        super(YOLOv1, self).__init__()

        self.num_classes = num_classes
        self.num_bboxes = num_bboxes

        # Backbone CNN layers
        self.conv_layers = nn.Sequential(
            # (1)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # (3, 448, 448) -> (64, 224, 224)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 224, 224) -> (64, 112, 112)
            
            # (2)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # (64, 112, 112) -> (192, 112, 112)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (192, 112, 112) -> (192, 56, 56)
            
            # (3)
            nn.Conv2d(192, 128, kernel_size=1),  # (128, 56, 56)
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (256, 56, 56)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),  # (256, 56, 56)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (512, 56, 56)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (512, 28, 28)
            
            # (4)
            nn.Conv2d(512, 256, kernel_size=1),  # (256, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),  # (256, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),  # (256, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1),  # (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # (1024, 28, 28)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (1024, 14, 14)
            
            # (5)
            nn.Conv2d(1024, 512, kernel_size=1),  # (512, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # (1024, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),  # (512, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # (1024, 14, 14)
            
            # (6)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),  # (1024, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),  # (1024, 14, 14)
            nn.LeakyReLU(0.1),
        )

        # Fully connected layers for predictions
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.num_bboxes * 5 + self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def test():

    # Instantiate the YOLOv1 model
    num_classes = 20  # Example: COCO dataset has 80 classes
    num_bboxes = 2    # Number of bounding boxes predicted per grid cell
    yolo_model = YOLOv1(num_classes, num_bboxes)

    # Print the model architecture
    print(yolo_model)

if __name__ == "__main__":
    test()