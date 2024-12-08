from utils.model import *
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import cv2
import sys
import numpy as np
import datetime


class LanGuideMedSegWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(LanGuideMedSegWrapper, self).__init__()
        
        self.model = LanGuideMedSeg(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()

    def configure_optimizers(self):

        # optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.hparams.lr)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model.forward(x)

    def shared_step(self,batch,batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds,y)
        # out_path = "/home/work/bui/LanGuideMedSeg-MICCAI2023/results/mosmed/GD/"
        # # Save results for illustration
        # for i in range(x[0].size(0)):
        #     img = x[0][i].cpu().numpy()
        #     mask = y[i].cpu().numpy()
        #     mask = np.stack([mask, mask, mask])
        #     mask = np.squeeze(mask)

        #     img = img.transpose((1, 2, 0))
        #     img = (img* 255.0).astype(np.float16)
        #     mask = mask.transpose((1, 2, 0))
        #     mask = (mask * 255.0).astype(np.uint8)
        #     pred = (preds[i] > 0.5).cpu().numpy()
        #     pred = (pred * 255.0).astype(np.uint8)
        #     pred = np.stack([pred, pred, pred])
        #     pred = np.squeeze(pred)
        #     pred = pred.transpose((1, 2, 0)).astype(np.uint8)
        #     # cv2.imwrite(out_path + f"img_{ids[i]}.png", img)
        #     # cv2.imwrite(out_path + f"mask_{ids[i]}", mask)
        #     cv2.imwrite(out_path + f"pred_{ids[i]}", pred)

        return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}    
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)

class MMIUNet_Wrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(MMIUNet_Wrapper, self).__init__()
        
        self.model = MMIUNet_V2(args.bert_type, args.vision_type, args.project_dim)
        model_dict = self.model.state_dict()
        pre_dict = torch.load('./pretrained/convnext_tiny_22k_224.pth') 
        matched_dict = {k: v for k, v in pre_dict['model'].items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(matched_dict)
        self.model.load_state_dict(model_dict)
        print(f'Load {len(matched_dict)} successfully')
        
        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters()

    def configure_optimizers(self):

        # optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.hparams.lr)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model.forward(x)

    def shared_step(self,batch,batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        # out_path = "/home/work/bui/LanGuideMedSeg-MICCAI2023/results/mosmed/MMIUNet/"
        # Save results for illustration
        # for i in range(x[0].size(0)):
        #     img = x[0][i].cpu().numpy()
        #     mask = y[i].cpu().numpy()
        #     mask = np.stack([mask, mask, mask])
        #     mask = np.squeeze(mask)

        #     img = img.transpose((1, 2, 0))
        #     img = (img* 255.0).astype(np.float16)
        #     mask = mask.transpose((1, 2, 0))
        #     mask = (mask * 255.0).astype(np.uint8)
        #     pred = (preds[i] > 0.5).cpu().numpy()
        #     pred = (pred * 255.0).astype(np.uint8)
        #     pred = np.stack([pred, pred, pred])
        #     pred = np.squeeze(pred)
        #     pred = pred.transpose((1, 2, 0)).astype(np.uint8)
        #     cv2.imwrite(out_path + f"mask_{ids[i]}", mask)
        #     cv2.imwrite(out_path + f"pred_{ids[i]}", pred) 

        return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}  
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)
