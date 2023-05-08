import torch

class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]
    

class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)



class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        # for each output channel, it applies a different filter over each input channel then sums the results.
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None # Isn't this just applying another 1d convolution that just isn't dilated? Is there a simpler way to up or down sample?

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)



class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels # in the first layer the number of channels is based on the data, so in_channels. In the rest of the layers, input/output is channels.
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param encoding_size Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param window_size windows of data to encode at a time. It should be ensured that this evenly
            divides into the length of the time series passed in. E.g. window_size is 120, and 
            x.shape[-1] (where x is passed into forward) is 120*c for some natural number c
    """
    def __init__(self, in_channels, channels, depth, reduced_size, encoding_size, kernel_size, device, window_size):
        super(CausalCNNEncoder, self).__init__()
        self.encoding_size = encoding_size
        self.device = device
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, encoding_size)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        ).to(device)

        self.window_size = window_size
        self.pruning_mask = torch.ones(self.encoding_size).bool() # This will be used to prune encoding dimensions that 
        self.pruned_encoding_size = int(torch.sum(self.pruning_mask))
        
    def forward(self, x, return_pruned=True):
        x = x.to(self.device)
        if len(tuple(x.shape)) == 4 and x.shape[1] == 2: # Meaning maps are included
            x = torch.reshape(x, (x.shape[0], x.shape[2]*2, x.shape[3]))
        
        elif len(tuple(x.shape)) == 3 and x.shape[0] == 2: # Maps are included
            x = torch.unsqueeze(x, 0) # Make it a batch of size 1, shape is (1, 2, num_features, seq_len)
            x = torch.reshape(x, (x.shape[0], x.shape[2]*2, x.shape[3]))
        
        elif len(tuple(x.shape)) == 3 and x.shape[0] != 2: # Maps aren't include. x is of shape (bs, num_featuers, seq_len)
            pass
        
        

        if return_pruned:
            return self.network(x)[:, self.pruning_mask]
        else:
            return self.network(x)
    
    def forward_seq(self, x, return_encoding_mask=False, sliding_gap=None, return_pruned=True):
        '''Takes a tensor of shape (num_samples, 2, num_features, seq_len) of timeseries data.
        
        Returns a tensor of shape (num_samples, seq_len/winow_size, encoding_size)'''
#         print("self.window_size", self.window_size)
#         print("x.shape", x.shape)
        assert x.shape[-1] % self.window_size == 0
        if len(tuple(x.shape)) == 3 and x.shape[1] == 2: # If a 3D tensor with maps is passed in, add a new dimension of size 1 to make it a batch of size 1
            x = torch.unsqueeze(x, 0)

        if sliding_gap:
            # This is a tensor of indices. If the data is of shape (num_samples, 2, num_features, 10), and window_size = 4 and sliding_gap=2, then inds is an array
            # of [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9]
            inds = torch.cat([torch.arange(ind, ind+self.window_size) for ind in range(0, x.shape[-1]-self.window_size+1, sliding_gap)])
            
            # Now for each sample we have the window_size windows concatenated for each sliding gap on the last axis.
            # So if window_size is 120 and sliding_gap is 20, then for each sample, the time dimension will go 
            # [0, 1, 2, ..., 119, 20, 21, ..., 139, 140, ...]
            x = torch.index_select(input=x, dim=3, index=inds)

        if return_encoding_mask:
            if x.shape[1] == 2:
                encoding_mask  = torch.sum(x[:, 1, :, :]==0, dim=1) # Of shape (num_samples, seq_len). the ij'th element is = num_features if that time step was fully imputed, <num_features otherwise
                encoding_mask[torch.where(encoding_mask < x.shape[2])] = 0 # Of shape (num_samples, seq_len). the ij'th element is = num_features if that time step was fully imputed, 0 otherwise
                encoding_mask = encoding_mask[:, self.window_size-1::self.window_size] # Of shape (num_samples, seq_len/self.window_size). On the second dim, only keeps every self.window_size'th element
                encoding_mask[encoding_mask == x.shape[2]] = -1 # Now the ij'th element is -1 if the j'th encoding from sample i was derived from fully imputed data. 0 otherwise
            else:
                encoding_mask = torch.ones(x.shape[0], x.shape[-1]//self.window_size)
        
        if len(tuple(x.shape)) == 4:
            num_samples, num_channels, num_features, seq_len = x.shape
            #print('entering forward_seq!')
            #print('num_samples, two, num_features, seq_len', num_samples, two, num_features, seq_len)
            x = torch.reshape(x, (num_samples, num_features*num_channels, seq_len)) # now x is of shape (num_samples, num_channels*num_features, seq_len)
            #print(x.shape, '==', num_samples, 2*num_features, seq_len)
            x = x.permute(1, 0, 2)
            x = x.reshape(num_channels*num_features, -1) # Now of shape (2*num_features, num_samples*seq_len)
            #print(x.shape, '==', 2*num_features, num_samples*seq_len)
            x = torch.stack(torch.split(x, self.window_size, dim=1)) # Now of shape (num_samples*(seq_len/window_size), num_channels*num_features, window_size)
            #print(x.shape, '==', num_samples*(seq_len/self.window_size), num_channels*num_features, self.window_size)

            encodings = self.forward(x, return_pruned=return_pruned) # encodings is of shape (num_samples*(seq_len/window_size), encoding_size)
            #print(encodings.shape, '==', num_samples*(seq_len/self.window_size), self.pruned_encoding_size)
            encodings = encodings.reshape(num_samples, int(seq_len/self.window_size), self.pruned_encoding_size)
        
        elif len(tuple(x.shape)) == 3:
            num_samples, num_features, seq_len = x.shape
            x = x.permute(1, 0, 2)
            x = x.reshape(num_features, -1) # Now of shape (num_features, num_samples*seq_len)
            #print(x.shape, '==', 2*num_features, num_samples*seq_len)
            x = torch.stack(torch.split(x, self.window_size, dim=1)) # Now of shape (num_samples*(seq_len/window_size), num_features, window_size)
            

            encodings = self.forward(x, return_pruned=return_pruned) # encodings is of shape (num_samples*(seq_len/window_size), encoding_size)
            #print(encodings.shape, '==', num_samples*(seq_len/self.window_size), self.pruned_encoding_size)
            encodings = encodings.reshape(num_samples, int(seq_len/self.window_size), self.pruned_encoding_size)
        
        if return_encoding_mask:
            return encodings, encoding_mask
        else:
            return encodings