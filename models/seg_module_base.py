import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

import numpy as np
import torch.nn as nn
from torch.nn.functional import interpolate

# to save the time to compute the batch
import time

class SegmentationModuleBase(pl.LightningModule):
    
    def __init__(self, args, model, loss, batch_size, trainset, train_sampler, valset, valid_sampler, num_classes):
        super(SegmentationModuleBase, self).__init__()
        self.args = args
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        # dataset to train
        self.trainset = trainset
        self.train_sampler = train_sampler
        self.valset = valset
        self.valid_sampler = valid_sampler
        self.num_classes = num_classes
        # define metric
        self.jaccard_train = MulticlassJaccardIndex(ignore_index=255, num_classes=self.num_classes)
        self.jaccard_val = MulticlassJaccardIndex(ignore_index=255, num_classes=self.num_classes)
    
    def forward(self, x, y):
        if self.args.feature_extractor:
            return self.model(x)
        else:
            if self.args.model == 'deeplabv3_resnet101' or self.args.model == 'deeplabv3':
                return self.model(x)['out']
            elif self.args.model == 'seg_former': 
                outs = self.model(pixel_values=x, labels=y)[1]
                outs = interpolate(outs, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
                return outs
            else:
                return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x, y = x.float(), y.long()
        x_embeddings = self.forward(x, y)
        
        # save predictions to plot tsne
        self.last_X = x_embeddings
        self.last_y = y
        
        start_time = time.time()
        loss_val = self.loss(x_embeddings, y)
        end_time = time.time()
        
        # compute time
        total_time = end_time - start_time
        
        # save log time
        self.log('total_time', total_time, prog_bar=False, logger=True)
        
        # log loss
        self.log('train_loss', loss_val, prog_bar=True, logger=True)
        
        self.jaccard_train(x_embeddings, y)
        self.log('jac_train', self.jaccard_train, prog_bar=True, logger=False)
        
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        
        return loss_val

    def training_epoch_end(self, outs):
        self.log('train_jac_epoch', self.jaccard_train, on_step=False, on_epoch=True) 
        
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x, y = x.float(), y.long()
        x_embeddings = self.forward(x, y)
        
        # save predictions to plot tsne
        self.last_X = x_embeddings
        self.last_y = y
        
        loss_val = self.loss(x_embeddings, y)
        self.log("val_loss", loss_val)
        
        self.jaccard_val(x_embeddings, y)
        self.log('jac_val', self.jaccard_val, prog_bar=False, logger=True)

        return {'loss': loss_val, 'jac_val': self.jaccard_val._forward_cache}
        
    def validation_epoch_end(self, outs):
        mean_outputs = {}
        for k in outs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outs]).mean()
        print(mean_outputs)
        
        self.log('val_jac_epoch', mean_outputs['jac_val'], on_step=False, on_epoch=True) 
            
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def configure_optimizers(self):
        if self.args.optim == 'adam': 
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
            return [optimizer]
        elif self.args.optim == 'nadam':
            optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
            return [optimizer]
        elif self.args.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
            return [optimizer]
        else: # sgd 
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.w_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs, eta_min=0.0001)
            return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.trainset, sampler=self.train_sampler, batch_size=self.batch_size, num_workers=self.args.num_workers, pin_memory=True, shuffle=False, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, sampler=self.valid_sampler, batch_size=self.batch_size, num_workers=self.args.num_workers, pin_memory=True, shuffle=False, drop_last=True)