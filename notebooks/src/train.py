from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")

from .dataset import BatchIndexedDataset

def prepare_dataloader(X, y, features, scenario, random=True):
    dataset = BatchIndexedDataset(X, y, features, scenario)
    if random:
        sampler = data.sampler.BatchSampler(
          data.sampler.RandomSampler(dataset),
          batch_size=1000,
          drop_last=False)
    else:
        sampler = data.sampler.BatchSampler(
            data.sampler.SequentialSampler(dataset),
          batch_size=1000,
          drop_last=False)      
    return data.DataLoader(dataset, sampler=sampler, batch_size=None)

def predict(model, train_X, dev_X, test_X, train_y, dev_y, test_y, features, test_features, scenario, epochs=15):
    ## Train classifier
    train_loader = prepare_dataloader(train_X, train_y, features, scenario)
    val_loader = prepare_dataloader(dev_X, dev_y, features, scenario)
    test_loader = prepare_dataloader(test_X, test_y, test_features, scenario, random=False)
    
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='valid_loss',
        mode='min'
    )
    
    trainer = pl.Trainer(gpus=0, max_epochs=epochs, progress_bar_refresh_rate=20,
                        checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_loader, val_loader)
    
    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    ## Predict for the best model
    model.eval()
    model = model.to(device)
    
    test_predictions = [] 
    true_labels = []
    with torch.no_grad():
        for batch_text_X, batch_text_features, batch_text_y in test_loader:
            test_predictions.append(model(batch_text_X.to(device), batch_text_features.to(device)))
            true_labels.append(batch_text_y)
    
    test_predictions = torch.cat(test_predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    
    return test_predictions, true_labels
        
class Classifier(pl.LightningModule):
    def __init__(self, model, output_type, lr=7*1e-3, output_dims=None):
        super().__init__()
        self.model = model
        self.output_type = output_type
        self.lr = lr
        self.output_dims = output_dims

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, x, features):
        x = self.model(x, features)
        return x

    def training_step(self, batch, batch_idx):
        output_dims = self.output_dims
        x, features, y = batch
        
        if self.output_type != 'onehot':
            y = y.float()
        output = self.forward(x, features)
        
        if self.output_type == 'onehot':
            loss = 0
            for cls_idx in range(len(output_dims)):
                start_idx =  sum(output_dims[:cls_idx])
                end_idx =  start_idx + output_dims[cls_idx]
                loss = loss + nn.CrossEntropyLoss()(output[:, start_idx:end_idx], y[:, cls_idx].long())
        else:
            loss = nn.MSELoss()(output, y)
        
        self.log('train_loss',  loss, on_epoch=True)

        return loss

    def training_epoch_end(self, outs):
        epoch_acc = self.train_acc.compute()
    
    def validation_step(self, batch, batch_idx):
        output_dims = self.output_dims
        x, features, y = batch
        y = y.float()
        output = self.forward(x, features)
        
        if self.output_type == 'onehot':
            loss = 0
            for cls_idx in range(len(output_dims)):
                start_idx =  sum(output_dims[:cls_idx])
                end_idx =  start_idx + output_dims[cls_idx]
                loss = loss + nn.CrossEntropyLoss()(output[:, start_idx:end_idx], y[:, cls_idx].long())
        else:
            loss = nn.MSELoss()(output, y)
            
        self.log('valid_loss', loss, prog_bar=True)
        
        return {'loss': loss, 'true_labels': output, 'predictions': y}

    def validation_epoch_end(self, outs):
        val_epoch_acc = self.valid_acc.compute()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=self.lr)

        return optimizer