import datetime
import math
import numpy as np
import random
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_auc_score
from statsmodels.tsa import stattools
import torch
from torch.utils import data

from causal_CNN_encoder import CausalCNNEncoder
from load_data import load_data, load_labels
from RNN_predictor import RnnPredictor
from utils import detect_incr_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
interval = 1

nghd_sizes = {}
num_neg_samples_removed = {}


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size # This is the encoding_size from the encoder. i.e. the dimension of the latent state
                                                # This is 2*self.input_size because we concatenate two encodings
        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,)) # returns output of the discriminator, its a scaler wrapped in a tensor
    

class TNCDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, window_size, eta=3, state=None, adf=False, acf=False, acf_plus=False, ACF_nghd_Threshold=0.4, ACF_out_nghd_Threshold=0.5):
        super(TNCDataset, self).__init__()
        self.time_series = x # Time series of shape (num_samples, 1, num_features, signal_length) if we have no maps, (num_samples, 2, num_features, signal_length) if we do
        self.T = x.shape[-1] # length of the time series
        self.window_size = window_size
        self.have_map = True if self.time_series.shape[1] == 2 else False
        # self.sliding_gap = int(window_size*25.2) # Commented out because its not used..
        # self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap # Commented out because its not used..
        self.mc_sample_size = mc_sample_size # num of montecarlo samples for estimating the expectations in the loss
        self.state = state # State of the patients. We have a value for each time step for each patient
        self.adf = adf # Boolean for if we want to do the ADF test
        self.acf = acf # Boolean for if we want to do Autocorrelation for nghd determination
        self.acf_plus = acf_plus # Boolean for if we want to do Autocorrelation for nghd choice AND use it for remove negative samples that are correlated to the nghd
        self.ACF_nghd_Threshold = ACF_nghd_Threshold
        self.ACF_out_nghd_Threshold = ACF_out_nghd_Threshold

        if not self.adf and not self.acf and not self.acf_plus:
            self.eta = eta
            self.nghd_size = 3*window_size*eta
        
        if not self.adf:
            self.acf_avgs = [] # Will store a list of acf values for a given sample. Will be modified on each call to _find_neighbors 
            for i in range(len(x)):
                acfs = []
                sample = x[i]
                for f in range(sample.shape[-2]):
#                     if len(torch.where(sample[1, f, :]==1)[0]) > sample.shape[-2]*0.4: # If more than 40% the data for this feature is observed, compute acf for it. else dont
                    acfs.append(torch.abs(torch.Tensor(stattools.acf(sample[0, f, :], nlags=sample.shape[-1] - 1))))
                acfs = torch.stack(acfs)
                self.acf_avgs.append(torch.mean(acfs, axis=0))
            


    def __len__(self):
        # self.augmentation is used when there are very few samples of data, but they are long. In that case, we may wish to break up the big samples into medium samples
        # e.g. if we have 20 really long samples, we may split them each into 2 samples, meaning we now have 40 samples
        return len(self.time_series) #*self.augmentation

    def __getitem__(self, index):
        '''When a TNCDataset object element is accessed with data[index] notation (but more importantly when you loop through it), it will return some window W_t of the index'th sample timeseries,
        a tensor X_close of self.mc_sample_size windows in the neighborhood and X_distant a tensor of self.mc_sample_size
        windows outside of the neighborhood, as well as y_t which is the approximated patient state
        over the window W_t'''
        end_T = self.T
        start_T = 0 # end_T and start_T represent the start of actual data and the end of actual data. i.e. the time range where each edge of the range does not have missingness (there can obviously be missing values in the middle though)
        if self.have_map: # if we have the map
            # then we can check for places where the data goes to all 0's (i.e. a time point after which all data is missing for any one feature)
            x_map = self.time_series[index][1] # of shape (num_features, signal_length). 0's in the map indicate missingness in the data

            x_map = 1-x_map # Switch to 0's indicating observed, 1's indicating missingness
            result = x_map[0]
            for i in range(1, x_map.shape[0]):
                result = torch.logical_and(result, x_map[i])
            # result is a vector that is 1 for time steps that were totally missing. 0 indicates at least one feature was observed
            x_map = 1-x_map # switch back to 1 indicating observed, 0 meaning missingness
            result = ~result # Now result has 0 to indicate time steps that were completely missing, 1 for steps that had some observed values
            end_offset = 1
            start_offset = 0
            while True:
                if result[x_map.shape[1] - end_offset] == 1.:
                    break 
                
                end_offset += 1

            while True:
                if result[start_offset] == 1.:
                    break
                
                start_offset += 1
            
            # end_offset is now the number of indexes at the end in which all features are missing. e.g. if offset = 5, then for the last 5 time steps all features are missing.
            # start_offset is now the number of indexes at the beinning where all features are missing.
            
            end_T = end_T - end_offset + 1
            start_T += start_offset

        index = index%len(self.time_series) # index for a sample of the full dataset self.time_series
        
        t = np.random.randint(start_T + 2*self.window_size, end_T-2*self.window_size) # randomly select t, the center of the window

        # self.time_series[ind] returns a 2D matrix for the ind'th sample. Shape is (num_features, signal_length)
        W_t = self.time_series[index][:, :, t-self.window_size//2:t+self.window_size//2] # Generate the window, this is W_t from the paper
        # plt.savefig('./plots/%s_seasonal.png'%index) 

        
        X_close = self._find_neighbors(self.time_series[index], t, start_T, end_T, index)
       
        X_distant = self._find_non_neighbors(self.time_series[index], t, start_T, end_T, index)
        

        if self.state is None: # If we have no patient state values
            y_t = -1
        else:
            if len(self.state.shape) == 1:
                y_t = self.state[index]
            elif len(self.state.shape) == 2:
                # self.state is of shape (num_samples, signal_length)
                y_t = torch.round(torch.mean(self.state[index][t-self.window_size//2:t+self.window_size//2]))


        
        # W_t is of shape (num_features, window_size)
        # X_close is of shape (mc_sample_size, num_features, window_size), so its a 'list' of mc_sample_size windows from the nghd
        # X_distant is of shape (mc_sample_size, num_features, window_size), so its a 'list' of mc_sample_size windows from outside the nghd
        # y_t is an integer
        return W_t, X_close, X_distant, y_t

    def _find_neighbors(self, x, t, start_T, end_T, index):
        '''Will find the neighborhood centered at t. x is a tensor for a single sample, shape is (1, num_features, signal_length) if no map, (2, num_features, signal_length) if we have map.
        Note: The T parameter is at most self.T, but can be less if the sample has mising values at the end.'''
        delta = self.window_size
        if self.adf:
            
            corr = []
            for w_t in range(self.window_size, 4*self.window_size, self.window_size): # Stepping by window_size chunks
                # 4*window_size is the farthest we'll consider away from t for the neighborhood
                
                try:
                    p_val = 0
                    for f in range(x.shape[-2]): # iterating through features
                        # Do ADF test for each feature separately on the window [t-w_t, t+w_t]
                        # x[:, 0, :, :] just isolates the data, leaves out map
                        p = stattools.adfuller(np.array(x[0, f, :][max(start_T,t - w_t):min(end_T, t + w_t)].reshape(-1,)))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    
                    corr.append(p_val/x.shape[-2]) # append the average p value over the features to corr
                except: # ??? why try except?
                    corr.append(0.6) # Why add .6?
            # .01 is the p value threshhold
            self.eta = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            
            # self.nghd_size is 1 standard deviation away from the mean of the nghd distribution. This far away, we will consider samples 'out of the neighborhood'
            self.nghd_size = self.eta*delta

        elif self.acf or self.acf_plus:
            acf_seq = self.acf_avgs[index]

            # Find first index where acf is < ACF_nghd_Threshold. i.e. the lag where autocorrelation is < ACF_nghd_Threshold. If there isn't, set it to the len of acfs
            avg_index = np.where(acf_seq < self.ACF_nghd_Threshold)[0][0] if len(np.where(acf_seq < self.ACF_nghd_Threshold)[0]) > 0 else len(acf_seq)

            self.eta = 0
            while self.eta*self.window_size <= avg_index and self.eta < 10:
                self.eta += 1
            self.nghd_size = self.eta*delta
        
        '''
        elif self.acf_plus_plus:

            self.acf_avgs = [] # Will store a list of acf values for a given sample. Will be modified on each call to _find_neighbors 
            
            acf_seq = []
            sample = x.squeeze()
            for f in range(sample.shape[-2]):
                acf_seq.append(torch.abs(torch.Tensor(stattools.acf(sample[0, f, t:end_T], nlags=end_T-t - 1))))
            acf_seq = torch.stack(acf_seq)
            acf_seq = torch.mean(acf_seq, axis=0) # Average across features

            first_index = np.where(acf_seq < self.ACF_nghd_Threshold)[0][0] if len(np.where(acf_seq < self.ACF_nghd_Threshold)[0]) > 0 else len(acf_seq)

            self.eta = 0
            while self.eta*self.window_size <= first_index and self.eta < 10:
                self.eta += 1
            self.nghd_size = self.eta*delta
        '''

        # Logging nghd sizes
        if self.nghd_size in nghd_sizes:
            nghd_sizes[self.nghd_size] += 1
        else:
            nghd_sizes[self.nghd_size] = 0
            
        ## Random from a Gaussian
        # t_p is a tensor of time values that will act as the centers of windows *in* the nbhd. There are self.mc_sample_size of them
        t_p = [int(t+np.random.randn()*self.nghd_size) for _ in range(self.mc_sample_size)]
        # Selecting time values that will allow windows to fit
        t_p = [max(start_T + self.window_size//2+1,min(t_pp, end_T-self.window_size//2)) for t_pp in t_p]
        
        # Stacking together windows in the same nghd
        x_p = torch.stack([x[:, :, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p
    
    def _find_non_neighbors(self, x, t, start_T, end_T, index):
        '''Will find non neighbors of the neighbordhood centered at t. x is a tensor for a single sample, shape is (1, num_features, signal_length) if we have no map, (2, num_features, signal_length) if we do
        Note: The T parameter is at most self.T, but can be less if the sample has mising values at the end.'''
        if self.acf_plus:
            # Recall self.nghd_size is the size of 1 standard deviation of the normal distribution that defines our neighborhood
            if t-start_T < 2*self.nghd_size: # if t is so close to the start that the nghd starts at start_T, then only select samples from the right
                t_n = np.random.randint(t + self.nghd_size + self.window_size//2, end_T - self.window_size//2, self.mc_sample_size)

            elif end_T - t < 2*self.nghd_size: # if t is so close to the end that the nghd ends at end_T, then only select samples from the left
                t_n = np.random.randint(start_T + self.window_size//2 + 1, end_T - self.nghd_size - self.window_size//2, self.mc_sample_size)


            else: # In the case where t falls somewhere in between, select negative samples proportionally on each side of the neighborhood
                start_of_nghd = t - self.nghd_size
                end_of_nghd = t + self.nghd_size

                proportion_to_right = (end_T - end_of_nghd)/((end_T - end_of_nghd) + (start_of_nghd - start_T))

                mc_sample_right = int(self.mc_sample_size*proportion_to_right)
                mc_sample_left = self.mc_sample_size-mc_sample_right

                t_left = np.random.randint(start_T + self.window_size//2 + 1, start_of_nghd, mc_sample_left)
                t_right = np.random.randint(end_of_nghd + self.window_size//2, end_T - self.window_size//2, mc_sample_right)

                t_n = np.concatenate([t_left, t_right])

            '''
            if t>(end_T-start_T)/2:
                # if t is in the second half of the time series, take non neighbors from the first half
                t_n = np.random.randint(start_T + self.window_size//2, max((t - self.nghd_size + 1), start_T + self.window_size//2+1), self.mc_sample_size)
            else:
                # if t is in the first half of the time series, take non neighbors from the second half
                t_n = np.random.randint(min((t + self.nghd_size), (end_T - self.window_size-1)), (end_T - self.window_size//2), self.mc_sample_size)
            '''

            t_n_final = []
            avg_acfs = self.acf_avgs[index]
            neg_sample_lags = np.array([abs(t-t_prime) for t_prime in t_n])
            for t_prime in t_n:
                include_t_n = True
                
                acf_avgs = self.acf_avgs[index]
                if abs(acf_avgs[abs(t-t_prime)]) > self.ACF_out_nghd_Threshold:
                    include_t_n = False
                
                if include_t_n:
                    t_n_final.append(t_prime)
            t_n_final = np.array(t_n_final)
            
            # Logging how many samples we are removing from the original negatives chosen
            diff = (len(t_n) - len(t_n_final))
            if diff in num_neg_samples_removed:
                num_neg_samples_removed[diff] += 1
            else:
                num_neg_samples_removed[diff] = 0

            
            # if we cut down some negative samples, we'll repeat the existing ones so we have mc_sample_size of them
            if len(t_n_final > 0):
                t_n = t_n_final
                while len(t_n) < self.mc_sample_size:
                    t_n = np.concatenate([t_n, t_n])
                t_n = t_n[0:self.mc_sample_size]
            else:
                # if we have removed all negative samples, grab the ones that had the lowest average correlation across features
                best_lags_inds = np.argsort(avg_acfs[neg_sample_lags])[:self.mc_sample_size]
                best_neg_sample_lags = neg_sample_lags[best_lags_inds]
                t_n_final = []
                for t_prime in t_n:
                    if abs(t-t_prime) in best_neg_sample_lags:
                        t_n_final.append(t_prime)
                
                assert len(t_n_final) != 0
                t_n = t_n_final
                while len(t_n) < self.mc_sample_size:
                    print("While loop in TNCDataset entered..")
                    t_n = np.concatenate([t_n, t_n])
                t_n = t_n[0:self.mc_sample_size]

        else:
            if t>(end_T-start_T)/2:
                # if t is in the second half of the time series, take non neighbors from the first half
                t_n = np.random.randint(start_T + self.window_size//2, max((t - self.nghd_size + 1), start_T + self.window_size//2+1), self.mc_sample_size)
            else:
                # if t is in the first half of the time series, take non neighbors from the second half
                t_n = np.random.randint(min((t + self.nghd_size), (end_T - self.window_size-1)), (end_T - self.window_size//2), self.mc_sample_size)
       
        if len(t_n) > 0:
            x_n = torch.stack([x[:, :, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])
        else:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > (end_T-start_T) / 2:
                x_n = x[:, :,start_T + rand_t:start_T + rand_t+start_T+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, :, end_T - rand_t - self.window_size:end_T - rand_t].unsqueeze(0)
        return x_n
    

def get_encoder(encoder_type, encoder_hyper_params):
    '''Takes in a string (e.g. 'Transformer'), and a dictionary of hyperparameters,
    and instantiates the appropriate encoder and returns it.'''
    return CausalCNNEncoder(**encoder_hyper_params)


def epoch_run(loader, disc_model, encoder, device, pruning_mask, w=0, optimizer=None, train=True, acf_plus=False, compute_pruning_mask=False):
    if train: # Puts encoder and discriminator into train mode
        encoder.train()
        disc_model.train()
        torch.save(encoder, 'pretrained_encoder.pt')
        torch.save(disc_model, 'pretrained_discriminator.pt')
    else:
        encoder.eval()
        disc_model.eval()
    # loader is a dataloader containing train or validation data (used as test data here, not used for hyperparamater tuning)
    loss_fn = torch.nn.BCEWithLogitsLoss() # sigmoid followed by binary cross entropy
    encoder.to(device)
    disc_model.to(device)
    
    # pruning_mask is of shape (encoding_size,). 1 for dimensions to keep, 0 for ones to prune
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    epoch_correlations = []
    for x_t, x_p, x_n, _ in loader:
        # x_t is of shape (batch_size, m, num_features, window_size), where m=1 if we have no maps, m=2 if we do. It is a window of data
        # x_p is of shape (batch_size, mc_sample_size, m, num_features, window_size) (where m=1 if we have no maps, m=2 if we do), so its a 'list' 
        # that is batch_size long (one for each sample in the batch), containing mc_sample_size windows from inside
        # the neighborhood
        # x_n is of shape (batch_size, mc_sample_size, m, num_features, window_size) (where m=1 if we have no maps, m=2 if we do), so its a 'list' 
        # that is batch_size long (one for each sample in the batch), containing mc_sample_size windows from outside
        # the neighborhood
        # _ is a list of integers representing the avg patient state for each of the batch_size windows

        mc_sample_size = x_p.shape[1]
        batch_size, m, num_features, window_size = x_t.shape

        x_p = x_p.reshape((-1, m, num_features, window_size))
        x_n = x_n.reshape((-1, m, num_features, window_size))
        # x_p and x_n are now of shape (batch_size * mc_sample_size, m, num_features, window_size) instead of 
        # (batch_size, mc_sample_size, m, num_features, window_size)

        x_t = torch.repeat_interleave(x_t, mc_sample_size, dim=0)
        # x_t is now of shape (batch_size * mc_sample_size, m, num_features, window_size). The batch_size windows
        # have been repeated mc_sample_size times, so we can do element wise comparision between x_t, x_p, and x_n,
        # and have a direct comparison between a window, a positive sample, and a (potentially) negative sample

        neighbors = torch.ones((len(x_p))).to(device)
        non_neighbors = torch.zeros((len(x_n))).to(device)
        x_t, x_p, x_n = x_t.squeeze().to(device), x_p.squeeze().to(device), x_n.squeeze().to(device)
        
        z_t = encoder(x_t, return_pruned=False)
        z_p = encoder(x_p, return_pruned=False)
        z_n = encoder(x_n, return_pruned=False) # z_t, z_p, and z_n are now all of size (batch_size*mc_sample_size, encoding_size)
        
        if compute_pruning_mask:
            encodings = torch.vstack([z_t[:, pruning_mask], z_p[:, pruning_mask], z_n[:, pruning_mask]]) # Now of shape (num_encodings, pruned encoding_size)
            encodings = torch.transpose(encodings, 0, 1) # Swap dims 0 and 1. Now each row is an encoding dimension
            corrs = torch.corrcoef(encodings)
            corrs = torch.abs(corrs)
            epoch_correlations.append(corrs)
        
        z_t *= pruning_mask
        z_p *= pruning_mask
        z_n *= pruning_mask # Sets encoding dimensions where pruning_mask is =0, to 0.

        
        d_p = disc_model(z_t, z_p)
        d_n = disc_model(z_t, z_n)
        
        p_loss = loss_fn(d_p, neighbors)
        n_loss = loss_fn(d_n, non_neighbors)
        n_loss_u = loss_fn(d_n, neighbors)
        if not acf_plus:
            loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2
        else: #acf_plus is True
            loss = (p_loss + n_loss)/2
        
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        p_acc = torch.sum(torch.nn.Sigmoid()(d_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(torch.nn.Sigmoid()(d_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
        
        x_t, x_p, x_n = x_t.to('cpu'), x_p.to('cpu'), x_n.to('cpu') # Move back to cpu
    
    if compute_pruning_mask:
        epoch_correlations = torch.stack(epoch_correlations) # Of shape (num_batches, pruned encoding_size, pruned encoding_size). Note: pruned encoding_size is just encoding_size at the start
        num_batches, _, _ = epoch_correlations.shape
        # print('Epoch correlations:')
        # print(torch.mean(epoch_correlations, dim=0)) # average over batch

        removed = [i for i in range(len(pruning_mask)) if pruning_mask[i] == 0]
        # print('removed: ', removed)
        # print('pruning_mask: ', pruning_mask)
        remaining_indices = torch.where(pruning_mask==1)[0]
        # print('remaining indices: ', remaining_indices)
        # print('epoch_correlations shape: ', epoch_correlations.shape)
        counter = 0
        for i in range(epoch_correlations.shape[1]):
            if counter >= 1: # Allowed to remove only 1 dimension at a time
                break
            for j in range(i+1, epoch_correlations.shape[1]):
                if counter >= 1:
                    break
                # print(i, j)
                if torch.sum(epoch_correlations[:, i, j] > 0.9) == num_batches and i not in removed:
                    removed.append(i)
                    remaining_indices[i] = -1
                    counter += 1
        #             print('removed dim: ', i)
        # print('num indices removed: ', counter)
        remaining_indices = torch.Tensor([i for i in remaining_indices if i != -1]).int()
        # print('remaining_indices after pruning: ', remaining_indices)
        pruning_mask = torch.zeros_like(pruning_mask)
        for ind in remaining_indices:
            pruning_mask[ind] = 1
        
        # print('pruning_mask after pruning: ', pruning_mask)
        # update encoder
        encoder.pruning_mask = pruning_mask
        encoder.pruned_encoding_size = int(torch.sum(encoder.pruning_mask))
    return epoch_loss/batch_count, epoch_acc/batch_count, pruning_mask
    

def learn_encoder(data_maps, encoder_type, encoder_hyper_params, pretrain_hyper_params, window_size, w, batch_size, lr=0.001, decay=0.005, mc_sample_size=20,
                  n_epochs=100, data_type='simulation', device='cpu', n_cross_val_encoder=1, cont=False, ETA=None, ADF=True, ACF=False, ACF_PLUS=False, ACF_nghd_Threshold=0.4, ACF_out_nghd_Threshold=0.4):
    
    # x is of shape (num_samples, num_features, signal_length) OR (num_samples, 2, num_features, signal_length) if we have maps for
    # our data which indicate where we have missing values. for each sample (which is of shape (2, num_features, signal_length)), sample[0] would be the data, and sample[1] is the map
    accuracies, losses = [], []
    for cv in range(n_cross_val_encoder):
        random.seed(21*cv)
        print("LEARN ENCODER CV: ", cv)
    
        encoder = get_encoder(encoder_type=encoder_type, encoder_hyper_params=encoder_hyper_params)
        encoder = encoder.to(device)
        pruning_mask = torch.ones(encoder_hyper_params['encoding_size']).bool().to(device)
        disc_model = Discriminator(encoder_hyper_params['encoding_size'], device)
        params = list(disc_model.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)

        performance = []
        best_acc = 0
        best_loss = np.inf
        epoch_start = 0

        inds = np.arange(len(data_maps))
        random.shuffle(inds)
        data_maps = data_maps[inds]

        train_data = data_maps[0:int(0.8*len(data_maps))]
        validation_data = data_maps[int(0.8*len(data_maps)):]

        print("ETA, ADF, ACF, ACF_PLUS: ", ETA, ADF, ACF, ACF_PLUS)
        if not ADF and not ACF and not ACF_PLUS:
            print("NOTE: ADF AND ACF ARE TURNED OFF FOR TNC")
            print("ETA IS MANUALLY SET TO ", ETA)
        elif ACF and not ADF:
            print("USING AUTOCORRELATION")
        elif not ACF and ADF:
            print("USING ADF")
        elif ACF_PLUS:
            print("USING ACF_PLUS")

        trainset = TNCDataset(x=train_data, mc_sample_size=mc_sample_size,
                                window_size=window_size, eta=ETA, adf=ADF, acf=ACF, acf_plus=ACF_PLUS, ACF_nghd_Threshold=ACF_nghd_Threshold, ACF_out_nghd_Threshold=ACF_out_nghd_Threshold)

        print('Done with TNCDataset for train data. Moving on to validation data...')
        validset = TNCDataset(x=validation_data, mc_sample_size=mc_sample_size,
                                window_size=window_size, eta=ETA, adf=ADF, acf=ACF, acf_plus=ACF_PLUS, ACF_nghd_Threshold=ACF_nghd_Threshold, ACF_out_nghd_Threshold=ACF_out_nghd_Threshold)

        print("Done making TNCDataset object for validation data")

        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

        if epoch_start < n_epochs-1:
            for epoch in range(epoch_start, n_epochs+1):
                if epoch % 30 == 0 and not ADF and epoch < n_epochs - 30 and epoch != 0:
                    compute_mask = True
                else:
                    compute_mask = False

                epoch_loss, epoch_acc, pruning_mask = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer,
                                                    w=w, train=True, device=device, acf_plus=ACF_PLUS, compute_pruning_mask=compute_mask, pruning_mask=pruning_mask)
                validation_loss, validation_acc, _ = epoch_run(valid_loader, disc_model, encoder, train=False, w=w, device=device, acf_plus=ACF_PLUS, compute_pruning_mask=False, pruning_mask=pruning_mask)

                performance.append((epoch_loss, validation_loss, epoch_acc, validation_acc))

                if epoch%10 == 0:
                    print('(cv:%s)Epoch %d Encoder Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Validation Loss: %.5f \t Validation Accuracy: %.5f'
                        % (cv, epoch, epoch_loss, epoch_acc, validation_loss, validation_acc))


                elif ADF: # If ADF, save checkpoint every epoch
                    print('(cv:%s)Epoch %d Encoder Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Validation Loss: %.5f \t Validation Accuracy: %.5f'
                        % (cv, epoch, epoch_loss, epoch_acc, validation_loss, validation_acc))

                if best_loss > validation_loss:
                    best_acc = validation_acc
                    best_loss = validation_loss

            accuracies.append(best_acc)
            losses.append(best_loss)

            train_loss = [t[0] for t in performance]
            validation_loss = [t[1] for t in performance]
            train_acc = [t[2] for t in performance]
            validation_acc = [t[3] for t in performance]
        
    # print("nghd sizes:", nghd_sizes, flush=True)
    # print("Recall, each nghd_size is the size of the standard deviation of the normal distribution defining the nghd")
    # print("num_neg_samples_removed: ", num_neg_samples_removed)
    
    print('=======> Performance Summary:')
    print('Accuracy: %.2f +- %.2f'%(100*np.mean(accuracies), 100*np.std(accuracies)))
    print('Loss: %.4f +- %.4f'%(np.mean(losses), np.std(losses)))
    return encoder


def linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size):
    if train:
        classifier.train()
    else:
        classifier.eval()
        
    epoch_losses = []
    epoch_predictions = []
    epoch_labels = []
    
    for data_batch, label_batch in dataset:
        data_batch = data_batch[:, :, :, -(3600//interval):] # Take just the last hr of data
        label_batch = label_batch[-(3600//interval):]
        
        encoding_batch, encoding_mask = encoder.forward_seq(data_batch, return_encoding_mask=True)
        encoding_batch = encoding_batch.to(device)
        
        label_batch = torch.Tensor([1 in label for label in label_batch]).to(device)
        
        # So now encoding_batch is of shape (num_samples, num_windows_per_sample, encoding_size)
            # and train_labels is of shape (num_samples,)
        predictions = torch.squeeze(classifier(encoding_batch)) # of shape (bs,)
        
        pos_weight = torch.Tensor([10]).to(device)
        if train:
            # Ratio of num negative examples divided by num positive examples is pos_weight
            # pos_weight = torch.Tensor([negative_encodings.shape[0] / max(positive_encodings.shape[0], 1)]).to(device)
            #print('pos_weight: ', pos_weight)
            
            #print('No positive weight set')
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Applies sigmoid to outputs passed in so we shouldn't have sigmoid in the model. 
            loss = loss_fn(predictions, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(predictions, label_batch)
        
        epoch_loss = loss.item()
        
        # Apply sigmoid to predictions since we didn't apply it for the loss function since the loss function does sigmoid on its own.
        predictions = torch.nn.Sigmoid()(predictions)
        
        # Move Tensors to CPU and remove gradients so they can be converted to NumPy arrays in the sklearn functions
        #window_labels = window_labels.cpu().detach()
        predictions = predictions.cpu().detach()
        encoding_batch = encoding_batch.cpu() # Move off GPU memory
        #neg_window_labels = neg_window_labels.cpu()
        #pos_window_labels = pos_window_labels.cpu()
        label_batch = label_batch.cpu()

        epoch_losses.append(epoch_loss)
        epoch_predictions.append(predictions)
        epoch_labels.append(label_batch)
        
    return epoch_predictions, epoch_losses, epoch_labels


def train_linear_classifier(X_train, y_train, X_validation, y_validation, X_TEST, y_TEST, encoding_size, num_pre_positive_encodings, encoder, window_size, batch_size=32, return_models=False, return_scores=False, pos_sample_name='arrest', data_type='ICU', classification_cv=0, encoder_cv=0, ckpt_path="../ckpt",  plt_path="../DONTCOMMITplots", classifier_name=""):
    '''
    Trains a classifier to predict positive events in samples.
    X_train is of shape (num_train_samples, 2, num_features, seq_len)
    y_train is of shape (num_train_samples, seq_len)

    '''
    print("Training Linear Classifier", flush=True)
    classifier = RnnPredictor(encoding_size=encoding_size, hidden_size=8).to(device)
    print('X_train shape: ', X_train.shape)
    print('batch_size: ', batch_size)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    TEST_dataset = torch.utils.data.TensorDataset(X_TEST, y_TEST)
    TEST_data_loader = torch.utils.data.DataLoader(TEST_dataset, batch_size=batch_size, shuffle=True)
    
    params = list(classifier.parameters())
    lr = .001
    weight_decay = .005
    print('Learning Rate for classifier training: ', lr)
    print('Weight Decay for classifier training: ', weight_decay)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, 501):
        encoder.eval()
        # linear_classifier_epoch_run(dataset, train, classifier, optimizer, data_type, window_size, encoder, encoding_size):
        epoch_train_predictions, epoch_train_losses, epoch_train_labels = linear_classifier_epoch_run(dataset=train_data_loader, train=True,
                                                    classifier=classifier,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        
        classifier.eval()
        epoch_validation_predictions, epoch_validation_losses, epoch_validation_labels = linear_classifier_epoch_run(dataset=validation_data_loader, train=False,
                                                    classifier=classifier,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)

        
        

        epoch_TEST_predictions, epoch_TEST_losses, epoch_TEST_labels = linear_classifier_epoch_run(dataset=TEST_data_loader, train=False,
                                                    classifier=classifier,
                                                    optimizer=optimizer, data_type=data_type, window_size=window_size, encoder=encoder, encoding_size=encoding_size)
        
        # TRAIN 
        # Compute average over all batches in the epoch
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_predictions = torch.cat(epoch_train_predictions)
        epoch_train_labels = torch.cat(epoch_train_labels)
        
        epoch_train_auroc = roc_auc_score(epoch_train_labels, epoch_train_predictions)
        # Compute precision recall curve
        train_precision, train_recall, _ = precision_recall_curve(epoch_train_labels, epoch_train_predictions)
        # Compute AUPRC
        epoch_train_auprc = auc(train_recall, train_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve

        train_losses.append(epoch_train_loss)

        # VALIDATION
        epoch_validation_loss = np.mean(epoch_validation_losses)
        epoch_validation_predictions = torch.cat(epoch_validation_predictions)
        epoch_validation_labels = torch.cat(epoch_validation_labels)

        epoch_validation_auroc = roc_auc_score(epoch_validation_labels, epoch_validation_predictions)
        # Compute precision recall curve
        valid_precision, valid_recall, _ = precision_recall_curve(epoch_validation_labels, epoch_validation_predictions)
        # Compute AUPRC
        epoch_validation_auprc = auc(valid_recall, valid_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
        valid_losses.append(epoch_validation_loss)
        
        # TEST
        epoch_TEST_loss = np.mean(epoch_TEST_losses)
        epoch_TEST_predictions = torch.cat(epoch_TEST_predictions)
        epoch_TEST_labels = torch.cat(epoch_TEST_labels)

        epoch_TEST_auroc = roc_auc_score(epoch_TEST_labels, epoch_TEST_predictions)
        # Compute precision recall curve
        TEST_precision, TEST_recall, _ = precision_recall_curve(epoch_TEST_labels, epoch_TEST_predictions)
        # Compute AUPRC
        epoch_TEST_auprc = auc(TEST_recall, TEST_precision) # precision is the y axis, recall is the x axis, computes AUC of this curve
        
        if epoch%10==0 or detect_incr_loss(valid_losses, 5):
            print('Epoch %d Classifier Loss =====> Training Loss: %.5f \t Training AUROC: %.5f \t Training AUPRC: %.5f\t Validation Loss: %.5f \t Validation AUROC: %.5f \t Validation AUPRC %.5f\t TEST Loss: %.5f \t TEST AUROC: %.5f \t TEST AUPRC %.5f'
                                % (epoch, epoch_train_loss, epoch_train_auroc, epoch_train_auprc, epoch_validation_loss, epoch_validation_auroc, epoch_validation_auprc, epoch_TEST_loss, epoch_TEST_auroc, epoch_TEST_auprc))
            epoch_train_predictions_thresholded = torch.clone(epoch_train_predictions)
            epoch_train_predictions_thresholded[epoch_train_predictions_thresholded >= 0.5] = 1
            epoch_train_predictions_thresholded[epoch_train_predictions_thresholded < 0.5] = 0

            epoch_validation_predictions_thresholded = torch.clone(epoch_validation_predictions)
            epoch_validation_predictions_thresholded[epoch_validation_predictions_thresholded >= 0.5] = 1
            epoch_validation_predictions_thresholded[epoch_validation_predictions_thresholded < 0.5] = 0

            epoch_TEST_predictions_thresholded = torch.clone(epoch_TEST_predictions)
            epoch_TEST_predictions_thresholded[epoch_TEST_predictions_thresholded >= 0.5] = 1
            epoch_TEST_predictions_thresholded[epoch_TEST_predictions_thresholded < 0.5] = 0
            
            print("Train classification report: ")
            print('epoch_train_labels shape: ', epoch_train_labels.shape, 'epoch_train_predictions_thresholded shape: ', epoch_train_predictions_thresholded.shape)
            print(classification_report(epoch_train_labels.to('cpu'), epoch_train_predictions_thresholded, target_names=['normal', pos_sample_name]))
            print("Validation classification report: ")
            print(classification_report(epoch_validation_labels.to('cpu'), epoch_validation_predictions_thresholded, target_names=['normal', pos_sample_name]))
            print()
            print("TEST classification report: ")
            print(classification_report(epoch_TEST_labels.to('cpu'), epoch_TEST_predictions_thresholded, target_names=['normal', pos_sample_name]))
            print()
        
            if detect_incr_loss(valid_losses, 5):
                break

        if return_models and return_scores:
            return (classifier, epoch_validation_auroc, epoch_validation_auprc, epoch_TEST_auroc, epoch_TEST_auprc)
        if return_models:
            return (classifier)
        if return_scores:
            return (epoch_validation_auroc, epoch_validation_auroc)


def trace(data_type, encoder_type, encoder_hyper_params, learn_encoder_hyper_params, classification_hyper_params, pretrain_hyper_params, circulatory_failure=False):
    
    global DATA_TYPE
    DATA_TYPE = data_type
    global ENCODER_TYPE
    ENCODER_TYPE = encoder_type
    global ECODER_HYPER_PARAMS
    ECODER_HYPER_PARAMS = encoder_hyper_params
    global LEARN_ENCODER_HYPER_PARAMS
    LEARN_ENCODER_HYPER_PARAMS = learn_encoder_hyper_params
    global CLASSIFICATION_HYPER_PARAMS
    CLASSIFICATION_HYPER_PARAMS = classification_hyper_params
    global PRETRAIN_HYPER_PARAMS
    PRETRAIN_HYPER_PARAMS = pretrain_hyper_params
    
    window_size = learn_encoder_hyper_params['window_size']
    length_of_hour = int(60*60/interval)
    
    pos_sample_name = 'arrest'
    
    signal_list = ["SpO2", "RESP", "HR", "ABP1", "ABP2", "ABP3"]
    
    sliding_gap = int(((60*60/5)/6)/interval) # equal to 24, which is 2 minutes in this freq. 1 hr in this time freq, dividided by 6 (10 min) divided by 5 (2 min).
    pre_positive_window = int(2*(60*60/interval))
    num_pre_positive_encodings = int(pre_positive_window/window_size)
    
    train_labels, test_labels = load_labels()
    train_data, test_data = load_data()
    TEST_mixed_data_maps = test_data
    TEST_mixed_labels = test_labels
    train_mixed_data_maps = train_data
    train_mixed_labels = train_labels

    # Used for training encoder
    TEST_encoder_data_maps = test_data
    train_encoder_data_maps = train_data

    # reshape to (num_samples, 1, num_features, signal_length)
    TEST_mixed_data_maps = torch.reshape(TEST_mixed_data_maps, (TEST_mixed_data_maps.shape[0], 1, TEST_mixed_data_maps.shape[1], TEST_mixed_data_maps.shape[2]))
    train_mixed_data_maps = torch.reshape(train_mixed_data_maps, (train_mixed_data_maps.shape[0], 1, train_mixed_data_maps.shape[1], train_mixed_data_maps.shape[2]))
    train_encoder_data_maps = torch.reshape(train_encoder_data_maps, (train_encoder_data_maps.shape[0], 1, train_encoder_data_maps.shape[1], train_encoder_data_maps.shape[2]))
    TEST_encoder_data_maps = torch.reshape(TEST_encoder_data_maps, (TEST_encoder_data_maps.shape[0], 1, TEST_encoder_data_maps.shape[1], TEST_encoder_data_maps.shape[2]))
    
    print('Entering learn_encoder')
    print("Current Time ", datetime.datetime.now())
    learn_encoder(data_maps=torch.vstack([train_encoder_data_maps, TEST_encoder_data_maps]), 
    encoder_type=encoder_type, encoder_hyper_params=encoder_hyper_params, 
    pretrain_hyper_params=pretrain_hyper_params, **learn_encoder_hyper_params)

    print('Finished training encoder')
    print("Current Time ", datetime.datetime.now())
        
    del train_encoder_data_maps
    del TEST_encoder_data_maps # Don't need these after training encoder
    
    classifier_validation_aurocs = []
    classifier_validation_auprcs = []
    classifier_TEST_aurocs = []
    classifier_TEST_auprcs = []
    
    for encoder_cv in range(learn_encoder_hyper_params['n_cross_val_encoder']):
        for classification_cv in range(classification_hyper_params['n_cross_val_classification']):
            print('SETTING SEED TO 111*CV + 2*(classification_cv+1)')
            seed_val = 111*encoder_cv+2*(classification_cv+1)
            random.seed(seed_val)
            print("Seed set to: ", seed_val)
            
            print('Classification CV: ', classification_cv)
            print('Encoder for CV ', encoder_cv)
            
            encoder = get_encoder(encoder_type, encoder_hyper_params).to(device)
            encoder.eval()
            
            print('Original shape of train data: ')
            print(train_mixed_data_maps.shape)
            # shuffle for this cv:
            inds = np.arange(len(train_mixed_data_maps))
            random.shuffle(inds)
            # print("First 15 inds: ", inds[:15])
            train_mixed_data_maps_cv = train_mixed_data_maps[inds]
            train_mixed_labels_cv = train_mixed_labels[inds]
            
            n_valid = int(0.2*len(train_mixed_data_maps_cv))
            validation_mixed_data_maps_cv = train_mixed_data_maps_cv[0:n_valid]
            validation_mixed_labels_cv = train_mixed_labels_cv[0:n_valid]

            print("Size of valid data: ", validation_mixed_data_maps_cv.shape)
            print("Size of valid labels: ", validation_mixed_labels_cv.shape)
            print("num positive valid samples: ", sum([1 in validation_mixed_labels_cv[ind] for ind in range(len(validation_mixed_labels_cv))]))
            
            train_mixed_data_maps_cv = train_mixed_data_maps_cv[n_valid:]
            train_mixed_labels_cv = train_mixed_labels_cv[n_valid:]
            print("Size of train data: ", train_mixed_data_maps_cv.shape)
            print("Size of train labels: ", train_mixed_labels_cv.shape)
            print("num positive train samples: ", sum([1 in train_mixed_labels_cv[ind] for ind in range(len(train_mixed_labels_cv))]))

            print('Size of TEST data: ', TEST_mixed_data_maps.shape)
            print('Size of TEST labels: ', TEST_mixed_labels.shape)
            print("Num positive TEST samples: ", sum([1 in TEST_mixed_labels[ind] for ind in range(len(TEST_mixed_labels))]))
            
            print('Mean of signals for train:',
                  torch.mean(train_mixed_data_maps_cv[:, 0, 0, :]),
                  torch.mean(train_mixed_data_maps_cv[:, 0, 1, :]),
                  torch.mean(train_mixed_data_maps_cv[:, 0, 2, :]),
                  torch.mean(train_mixed_data_maps_cv[:, 0, 3, :]),
                  torch.mean(train_mixed_data_maps_cv[:, 0, 4, :]),
                  torch.mean(train_mixed_data_maps_cv[:, 0, 5, :])
                )

            print('Mean of signal 1 for validation:',
                  torch.mean(validation_mixed_data_maps_cv[:, 0, 0, :]),
                  torch.mean(validation_mixed_data_maps_cv[:, 0, 1, :]),
                  torch.mean(validation_mixed_data_maps_cv[:, 0, 2, :]),
                  torch.mean(validation_mixed_data_maps_cv[:, 0, 3, :]),
                  torch.mean(validation_mixed_data_maps_cv[:, 0, 4, :]),
                  torch.mean(validation_mixed_data_maps_cv[:, 0, 5, :])
                )

            print('Mean of signal 1 for TEST:',
                  torch.mean(TEST_mixed_data_maps[:, 0, 0, :]),
                  torch.mean(TEST_mixed_data_maps[:, 0, 1, :]),
                  torch.mean(TEST_mixed_data_maps[:, 0, 2, :]),
                  torch.mean(TEST_mixed_data_maps[:, 0, 3, :]),
                  torch.mean(TEST_mixed_data_maps[:, 0, 4, :]),
                  torch.mean(TEST_mixed_data_maps[:, 0, 5, :])
                )

            train_pos_sample_inds = [ind for ind in range(len(train_mixed_labels_cv)) if 1 in train_mixed_labels_cv[ind]]
            train_neg_sample_inds = [ind for ind in range(len(train_mixed_labels_cv)) if 1 not in train_mixed_labels_cv[ind]]
            
            inds_for_classification = train_pos_sample_inds.copy()
            
            print("TRAINING LINEAR CLASSIFIER")
            classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv, y_train=train_mixed_labels_cv, 
            X_validation=validation_mixed_data_maps_cv, y_validation=validation_mixed_labels_cv, 
            X_TEST=TEST_mixed_data_maps, y_TEST=TEST_mixed_labels,
            encoding_size=encoder.pruned_encoding_size, batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings, encoder=encoder, window_size=encoder_hyper_params['window_size'], return_models=True, return_scores=True, pos_sample_name=pos_sample_name, 
            data_type=data_type, classification_cv=classification_cv, encoder_cv=encoder_cv)

            classifier_validation_aurocs.append(valid_auroc)
            classifier_validation_auprcs.append(valid_auprc)
            classifier_TEST_aurocs.append(TEST_auroc)
            classifier_TEST_auprcs.append(TEST_auprc)

    print("CLASSIFICATION VALIDATION RESULT OVER CV")
    print("AUC: %.2f +- %.2f, AUPRC: %.2f +- %.2f"% \
        (np.mean(classifier_validation_aurocs), 
        np.std(classifier_validation_aurocs), 
        np.mean(classifier_validation_auprcs), 
        np.std(classifier_validation_auprcs)))

    print("CLASSIFICATION TEST RESULT OVER CV")
    print("AUC: %.2f +- %.2f, AUPRC: %.2f +- %.2f"% \
        (np.mean(classifier_TEST_aurocs), 
        np.std(classifier_TEST_aurocs), 
        np.mean(classifier_TEST_auprcs), 
        np.std(classifier_TEST_auprcs)))
    
    print("Finished running on ", datetime.datetime.now())