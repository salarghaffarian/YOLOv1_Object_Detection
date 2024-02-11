import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT 
from tqdm import tqdm
from torch.utils.data import dataloader
from model import YOLOv1
from dataset import Yolov1Dataset
from utils import (
    intersection_over_union,  # Compate the IoU for the bbox1 & bbox2 and get the best box for each grid cell (best_box) and the IoU value (iou_maxes).
    non_max_suppression,      # Remove the bounding boxes under a specific IoU threshold.
    mean_average_precision,   
    cellboxes_to_boxes, 
    get_bboxes, 
    plot_image,
    save_checkpoint, 
    load_checkpoint,
)

from loss import YoloLossss

seed = 123
torch.manual_seed(seed)


# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16     # Batch size = 64 in the paper
WEIGHT_DECAY = 0    # Weight decay for the optimizer
EPOCHS = 100        # Number of epochs
NUM_WORKERS = 2     # Number of workers for the dataloader
PIN_MEMORY = True   # For faster data loading
LOAD_MODEL = False  # Load the model from a checkpoint
LOAD_MODEL_FILE = "overfit.pth.tar"      # Load the model from a checkpoint file
IMG_DIR = "data/images"                  # Directory for the images
LABEL_DIR = "data/labels"                # Directory for the labels

in_img_size_height = 448   # pixel
in_img_size_width = 448    # pixel

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
     
transform = Compose([transforms.Resize((in_img_size_height, in_img_size_width)),   # Resize the image to (448, 448)
                     transforms.ToTensor()]                                        # Convert the image to tensor
                     )

def train_func(train_loader, model, optimizer, loss_func):
    loop = tqdm(train_loader, leave=True)    # the parameter leave=True will keep the progress bar in the terminal after the loop is finished.
    mean_loss = []

    epoch = 0                                  # Define the variable 'epoch'
    for batch_idx, (x, y) in enumerate(loop):  # x: image, y: label
        x, y = x.to(DEVICE), y.to(DEVICE)      # Move the image and label to the device
        out = model(x)                         # Forward pass out tensor size=(N, S*S*(C+B*5)), where N=batch size, S=split size, C=Number of classes, B=number of bboxes in a cell
        loss = loss_func(out, y)               # loss size is 
        mean_loss.append(loss.item())          # Append the loss to the mean_loss list
        optimizer.zero_grad()                  # Set the gradients to zero before backpropagation, because PyTorch accumulates the gradients on subsequent backward passes.
        loss.backward()                        # Backward pass
        optimizer.step()                       # Update the weights

        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")    # update the progressbar with the current epoch
        loop.set_postfix(loss=loss.item())                   # update the progress bar with the loss value of the current batch

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


   
