import torch.utils.data as data
import torch
device = torch.device("cpu")

class BatchIndexedDataset(data.Dataset):
    def __init__(self, X, y, additional_data, scenario):
        self.X = X
        self.y = torch.tensor(y)
        self.scenario = scenario
        
        embeddings, features, biases = additional_data
        
        self.embeddings = torch.tensor(embeddings).to(device)
        self.annotator_features = torch.tensor(features).to(device)
        self.annotator_biases = torch.tensor(biases).to(device)

    def __getitem__(self, index):
        revs_X = self.X[index, 0]
        workers_X = self.X[index, 1]

        batch_X = self.embeddings[revs_X]
        batch_features = torch.empty((len(index), 0))
        batch_y = self.y[index]

        if self.scenario == 's3':
            batch_X = torch.zeros((len(index), 1))
            batch_features = self.annotator_biases[workers_X]
            
        elif self.scenario == 's2':
            batch_features = self.annotator_features[workers_X]
            
        elif self.scenario == 's4':
            batch_features = self.annotator_biases[workers_X]

        elif self.scenario == 's5':
            batch_features = torch.cat([self.annotator_features[workers_X], self.annotator_biases[workers_X]], dim=1)
            
        return batch_X.float().to(device), batch_features.float().to(device), batch_y.to(device)
    
    def __len__(self):
        return len(self.y)