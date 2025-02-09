import torch
import torch.nn as nn

from torch.autograd import Variable

from . import base

# run using torch 1.13.1
class PrototypicalGlobalLocalTripletLoss(base.Loss):
    
    def __init__(self, num_classes, margin_global, margin_local, magnitude=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.magnitude = magnitude
        self.build_anchors()
        distance_function = nn.PairwiseDistance(p=2)
        self.pdist = distance_function.cuda()
        self.triplet_loss_global = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist,
            swap=False,
            margin = margin_global
            )
        self.triplet_loss_local = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist,
            swap=False,
            margin = margin_local
            )
    
    def build_anchors(self):
        self.anchors = torch.zeros((self.num_classes, self.num_classes), device='cuda')
        self.magnitude = self.magnitude
        for i in range(self.num_classes): # num_classes
            self.anchors[i][i] = self.magnitude
       
    def forward(self, x_embeddings, y_embeddings):
        
        # reshape
        x_embeddings = x_embeddings.permute(0, 2, 3, 1).contiguous()
        shape = x_embeddings.size()
        
        # compute argmax
        x_pred = torch.argmax(x_embeddings, dim=3)
        
        ###################################
        # LOCAL ATTRACTION AND REPULSION #
        ###################################
        
        x_pred_local = x_pred.view(shape[0], shape[1] * shape[2])
        x_embeddings_local = x_embeddings.view(shape[0], shape[1] * shape[2], shape[3]) 
        y_embeddings_local = y_embeddings.view(shape[0], shape[1] * shape[2])
        
        # loop images in batch (to compute local attraction and repulsion)
        triplet_loss_local = Variable(torch.Tensor([0])).cuda()
        total_local = 0
        for b in range(shape[0]):
            for c in range(self.num_classes):
            
                # --
                # positives (TP) (Easy local triplets positives)
                anchors_index = torch.where((x_pred_local[b] == c) & (y_embeddings_local[b] == c))[0]
                anchors = torch.index_select(x_embeddings_local[b], dim=0, index=anchors_index)               

                if anchors_index.size()[0] == 0:
                    continue
                
                # --
                # positives (FN) (False negatives are hard local triplets positives)
                positive_index = torch.where((x_pred_local[b] != c) & (y_embeddings_local[b] == c))[0]
                positives = torch.index_select(x_embeddings_local[b], dim=0, index=positive_index) 

                # TN + FP
                negative_index = torch.where(y_embeddings_local[b] != c)[0] 
                negatives = torch.index_select(x_embeddings_local[b], dim=0, index=negative_index) 

                # if no pixels of class x are found, do not compute
                if positive_index.size()[0] == 0 or negative_index.size()[0] == 0:
                    continue
                
                # random samples
                anchors = anchors[torch.randperm(anchors.size()[0])]
                positives = positives[torch.randperm(positives.size()[0])]
                negatives = negatives[torch.randperm(negatives.size()[0])]
                
                # slice triplets
                max_triplets = min([anchors.size()[0], negatives.size()[0], positives.size()[0]])
                
                anchors = anchors[0:max_triplets]
                negatives = negatives[0:max_triplets]
                positives = positives[0:max_triplets]
                
                # compute local prototype triplet
                triplet_loss_local += self.triplet_loss_local(anchors, positives, negatives) 
                total_local += 1
                    
        # avg all triplets loss of images
        triplet_loss_local = triplet_loss_local / total_local
                
        ###################################
        # GLOBAL ATTRACTION AND REPULSION #
        ###################################
        
        x_pred_global = x_pred.view(shape[0] * shape[1] * shape[2])
        x_embeddings_global = x_embeddings.view(shape[0] * shape[1] * shape[2], shape[3]) 
        y_embeddings_global = y_embeddings.view(shape[0] * shape[1] * shape[2])
        
        # loop anchors 
        triplet_loss_global = Variable(torch.Tensor([0])).cuda()
        total_global = 0
        for c, k in enumerate(self.anchors):

            anchors = k
            
            # positives (TP) (Easy positives)
            positive_index = torch.where((x_pred_global == c) & (y_embeddings_global == c))[0]
            positives = torch.index_select(x_embeddings_global, dim=0, index=positive_index)
            
            # negative (FP) (Hard negatives)
            negative_index = torch.where((x_pred_global == c) & (y_embeddings_global != c))[0]
            negatives = torch.index_select(x_embeddings_global, dim=0, index=negative_index)
            
            # if no pixels of class x are found, do not compute
            if positive_index.size()[0] == 0 or negative_index.size()[0] == 0:
                continue
            
            # random samples
            positives = positives[torch.randperm(positives.size()[0])]
            negatives = negatives[torch.randperm(negatives.size()[0])]
            
            # slice triplets
            max_triplets = min([negatives.size()[0], positives.size()[0]])
            
            negatives = negatives[0:max_triplets]
            positives = positives[0:max_triplets]
            
            triplet_loss_global += self.triplet_loss_global(anchors, positives, negatives) 
            total_global += 1
            
        triplet_loss_global = triplet_loss_global / total_global   
        
        return triplet_loss_global.to(dtype=torch.float64) + triplet_loss_local.to(dtype=torch.float64)

class L1Loss(nn.L1Loss, base.Loss):
    pass

class MSELoss(nn.MSELoss, base.Loss):
    pass

class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass

class NLLLoss(nn.NLLLoss, base.Loss):
    pass

class BCELoss(nn.BCELoss, base.Loss):
    pass

class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
