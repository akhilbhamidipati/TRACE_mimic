import torch

class RnnPredictor(torch.nn.Module):
    def __init__(self, encoding_size, hidden_size, n_classes=1):
        super(RnnPredictor, self).__init__()
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = torch.nn.LSTM(input_size=encoding_size,  hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        if n_classes == 1:
            self.linear = torch.nn.Linear(hidden_size, n_classes)
        else:
            self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_size, 32), 
                                              torch.nn.Linear(32, n_classes))

    def forward(self, x, return_full_seq=False):
        # x is of shape (num_samples, seq_len, num_features)
        output, (h_n, _) = self.rnn(x)
        preds = []
        for hidden_states in output:
            preds.append(self.linear(hidden_states).squeeze()) # of shape (seq_len, num_classes)
        
        # of shape (num_samples, seq_len, n_classes) or (num_samples, seq_len) if n_classes ==1. The ijk'th element is the prediction of class k for sample i at time j
        preds = torch.stack(preds) # Note this must be passed through sigmoid after output
        if return_full_seq:
            return preds
        else:
            return preds[:, -1] # of shape (num_samples, n_classes)