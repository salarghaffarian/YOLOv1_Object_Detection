import torch
from torch.utils.data import Dataset
import os
import pandas as pd 
from PIL import Image 

'''
>> For Creating a custom dataset

A custom dataset should inherit the torch.utils.data.Dataset class.
The custom dataset should override the following methods:
    __init__ : The constructor method which is used to instantiate the dataset object
    __len__ : Returns the length of the dataset
    __getitem__ : Returns a sample from the dataset at the given index, which is used to read the image and label from the directories and apply the transform to the image.

The transform is used to apply the image transformation to the input image.

>> Regarding Yolov1Dataset:

    - is used to create a custom dataset for the YOLOv1 model.
    - is used to read the image and label directories from the csv file.
    - The csv file contains the image and label directories.
    - The csv file is used to read the image and label directories from the csv file.
    - The image and label directories are used to read the image and label from the directories.
    - The image and label are read from the directories and transformed using the transform method.
    - The transform method is used to apply the image transformation to the input image.
'''

class Yolov1Dataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        Initializes the dataset object.

        Args:
            csv_file (str): Path to the CSV file containing the annotations.
            img_dir (str): Directory path where the images are stored.
            label_dir (str): Directory path where the labels are stored.
            S (int): Number of grid cells in each dimension (default: 7). >>>  This is the split size for the input image.
            B (int): Number of bounding boxes per grid cell (default: 2).
            C (int): Number of classes (default: 20).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        def __len__(self):
            """
            Returns the length of the dataset.
            
            Returns:
                int: Length of the dataset.
            """
            return len(self.annotations)
        
        def __getitem__(self, index):
            """
            Returns a sample from the dataset at the given index.
            """

            # Getting the labels from the label directory
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            boxes = []
            with open(label_path) as f:
                    for label in f.readlines():
                        class_label, x, y, width, height =  [float(i) if float(i)!=int(float(i)) else int(i) for i in label.replace("\n", "").split()]
                        boxes.append([class_label, x, y, width, height])  # Boxes for each txt file are a list of lists including class_label, x, y, width, and height. 
            
            boxes = torch.tensor(boxes)   # Converting the list of boxes to a tensor. Depending on the number of objects inside the image, the shape of the tensor will be (n_object, 5)

            # Getting the image from the image directory
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = Image.open(img_path)

            # Doing the transformation
            if self.transform:
                image, boxes = self.transform(image, boxes)     # Note: The reason to have image and boxes as the input instead of single image is that some image transformation methods may change the bounding box coordinates too. e.g. rotating
            
            # converting the images into the required tensor format
            label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))  # (S, S, C+5*B) >>> 7x7x(20+5*2) = 7x7x30 #  

            for box in boxes:   # Length of boxes is the number of objects in the image.
                class_label, x, y, width, height = box.tolist()
                class_label = int(class_label)
                i, j = int(self.S * y), int(self.S * x)                      # Index of the cell in which the center of the object lies
                x_cell, y_cell = self.S * x - j, self.S * y - i              # The x and y coordinates of the center of the object with respect to the cell                     
                width_cell, height_cell = width * self.S, height * self.S    # The width and height of the object with respect to the cell

                # In the label matrix, which is 
                if label_matrix[i, j, self.C] == 0:                    # if (self.C = 20) This is the 20th index of the label matrix. If it is 0, then it means that there is no object in the cell & and if it is 1, then it means that there is an object in the cell.
                    label_matrix[i, j, self.C] = 1                     # for the objectness score
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])   
                    label_matrix[i, j, self.C+1:self.C+5] = box_coordinates   # for the bounding box coordinates
                    label_matrix[i, j, class_label] = 1   # for the class labels (one-hot encoding)
                
            return image, label_matrix   # The label matrix size is (S=7, S=7, C+5*B=30) >>> 7x7x30 for now. #!!!
            
            #TODO: consider the size of label_matrix in the Dataset and Dataloader classes and the loss function for validate its size to be (7, 7, 30) instead of (7, 7, 25)
        
                         


                            

                            
                     
            


