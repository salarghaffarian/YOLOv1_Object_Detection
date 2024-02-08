import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S   # split the image into SxS grid cells
        self.B = B   # number of bounding boxes in each grid cell
        self.C = C   # number of classes
        self.lambda_noobj = 0.5   # weight for the no object loss
        self.lambda_coord = 5     # weight for the bounding box coordinates loss

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # Calculate the IoU for the bbox1 & bbox2 and get the best box for each grid cell (best_box) and the IoU value (iou_maxes).
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)     # Iobj_i


        '''
        (1) Box coordinates loss

        This part focuses on the loss of the bounding box coordinates, which is the x, y, w, h values. 
        It compares the IoU of the predicted bounding box and the target bounding box. 
        The loss is calculated by the mean squared error of the predicted bounding box and the target bounding box.
        There are two bounding boxes in each grid cell, so the loss is calculated based on the bounding box with the highest IoU.
        The loss is calculated only if the object exists in the grid cell.
        '''
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), 
                            torch.flatten(box_targets, end_dim=-2))
        
        '''
        (2) Object loss

        This part focuses on the loss of the objectness, which is the confidence score of the bounding box.
        It compares the predicted objectness and the target objectness. 
        The Objectness is the confidence score of the bounding box containing an object.
        '''

        pred_box = (best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21])
        object_loss = self.mse(torch.flatten(exists_box * pred_box), 
                            torch.flatten(exists_box * target[..., 20:21]))
        

        '''
        (3) No object loss
    
        '''
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
                                  torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))       # for the first bounding box
        
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                                   torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))      # for the second bounding box
        
        '''
        (4) Class loss
        '''
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
                             torch.flatten(exists_box * target[..., :20], end_dim=-2))
        
        loss = (self.lambda_coord * box_loss          # (1) First two rows of the Loss function in the paper
                + object_loss                         # (2) Object loss
                + self.lambda_noobj * no_object_loss  # (3) No object loss
                + class_loss)                         # (4) Class loss

        return loss
    
        






