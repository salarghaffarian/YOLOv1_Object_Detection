import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, split_size=7, num_classes=20, num_bboxes=2):
        super(YOLOv1, self).__init__()
        self.S = split_size     # Is the split size for the input image. = 7 in Original YOLOv1 model.
        self.C = num_classes    # Number of classes in the original YOLOv1 is specified as 20. 
        self.B = num_bboxes     # Number of bounding boxes in each grid is 2 in YOLOv1.

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
            nn.Conv2d(192, 128, kernel_size=1),  # (192, 56, 56) -> (128, 56, 56)
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (128, 56, 56) -> (256, 56, 56)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),  # (256, 56, 56) -> (256, 56, 56)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (256, 56, 56) -> (512, 56, 56)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (512, 56, 56) -> (512, 28, 28)
            
            # (4)
            nn.Conv2d(512, 256, kernel_size=1),  # (512, 28, 28) -> (256, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (256, 28, 28) -> (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),  # (512, 28, 28) -> (256, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (256, 28, 28) -> (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),  # (512, 28, 28) -> (256, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (256, 28, 28) -> (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1),  # (512, 28, 28) -> (512, 28, 28)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # (512, 28, 28) -> (1024, 28, 28)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (1024, 28, 28) -> (1024, 14, 14)
            
            # (5)
            nn.Conv2d(1024, 512, kernel_size=1),  # (1024, 14, 14) -> (512, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # (512, 14, 14) -> (1024, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),  # (1024, 14, 14) -> (512, 14, 14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # (512, 14, 14) -> (1024, 7, 7)
            
            # (6)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),  # (1024, 7, 7) -> (1024, 7, 7)
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),  # (1024, 7, 7) -> (1024, 7, 7)
            nn.LeakyReLU(0.1),
        )

        # Fully connected layers for predictions
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))    # 4096 -> (7*7*(2*5+20)) = 49*30 = 1470
            #TODO: adding sigmoid layer here.  ???  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def test():

    # Instantiate the YOLOv1 model
    num_classes = 20  # Example: COCO dataset has 80 classes
    num_bboxes = 2    # Number of bounding boxes predicted per grid cell
    split_size = 7
    yolo_model = YOLOv1(split_size, num_classes, num_bboxes)

    x = torch.randn((10, 3, 448, 448))
    print(f'Model shape: {yolo_model(x).shape}')
    # Print the model architecture
    # print(yolo_model)
    
if __name__ == "__main__":
    test()
