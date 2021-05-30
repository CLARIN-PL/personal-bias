import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cpu")

class Net(nn.Module):

    def __init__(self, classes_num, text_feature_num, additional_feature_num):
        super().__init__()
        self.text_feature_num = text_feature_num
        self.additional_feature_num = additional_feature_num
        hidden_dim = 50
        
        self.fc_text = nn.Linear(text_feature_num, hidden_dim) 
        
        if additional_feature_num:
            self.fc_features = nn.Linear(additional_feature_num, hidden_dim) 
            self.fc_last = nn.Linear(2*hidden_dim, classes_num) 
        else:
            self.fc_last = nn.Linear(hidden_dim, classes_num) 
            
        self.dp = nn.Dropout(0.5)
        self.softplus = nn.Softplus()
            
    def forward(self, x, features, text_lengths=None):
        text_x = self.fc_text(x)
        text_x = self.softplus(text_x)
        
        if self.additional_feature_num:
            features_x = self.fc_features(features)
            return self.fc_last(torch.cat([text_x, features_x], dim=1))
        else:
            return self.fc_last(text_x)

class LSTMNet(nn.Module):

    def __init__(self, classes_num, text_feature_num, additional_feature_num, word_embeddings):
        super().__init__()
        self.text_feature_num = text_feature_num
        self.additional_feature_num = additional_feature_num
        
        self.embedding = torch.nn.Embedding.from_pretrained(word_embeddings, 
                                            padding_idx=0)
        
        self.hidden_dim = 32
        self.rnn = nn.LSTM(word_embeddings.shape[1], 
                           self.hidden_dim, 
                           num_layers=1, 
                           bidirectional=False, 
                           dropout=0.5, 
                           batch_first=True)
        
        if additional_feature_num:
            self.fc_features = nn.Linear(additional_feature_num, self.hidden_dim) 
            self.fc_last = nn.Linear(2*self.hidden_dim, classes_num) 
        else:
            self.fc_last = nn.Linear(self.hidden_dim, classes_num) 
            
        self.dp = nn.Dropout(0.5)
        self.softplus = nn.Softplus()
        
    def forward(self, tokens, features):
        x = self.embedding(tokens.long())

        lens_X = (tokens != 0).sum(dim=1).to('cpu')
        lens_X[lens_X == 0] = 1
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lens_X, batch_first=True, enforce_sorted=False).to(device)
        
        x, (hidden, cell) = self.rnn(x)
        
        if self.additional_feature_num:
            features_x = self.fc_features(features)
            return self.fc_last(torch.cat([hidden.view(-1, self.hidden_dim), features_x], dim=1))
        else:
            return self.fc_last(hidden.view(-1, self.hidden_dim))