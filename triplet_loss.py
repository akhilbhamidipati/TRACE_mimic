import numpy as np
import random
import torch

from causal_CNN_encoder import CausalCNNEncoder
from load_data import load_data, load_labels
from trace_model import train_linear_classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
interval = 1


class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        batch=batch.to(device)
        train=train.to(device)
        encoder = encoder.to(device)
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = self.compared_length
        # length_pos_neg = np.random.randint(1, high=length + 1)


        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = self.compared_length

        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg + np.random.randint(0,self.compared_length)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )


        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]).to(device))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss

def epoch_run(data, encoder, device, window_size, optimizer=None, train=True):
    if train:
        encoder.train()
    else:
        encoder.eval()
    encoder = encoder.to(device)
    loss_criterion = TripletLoss(compared_length=window_size, nb_random_samples=10, negative_penalty=1)

    epoch_loss = 0
    acc = 0
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data).to(device), torch.zeros((len(data),1)).to(device))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    i = 0
    for x_batch,y in data_loader:
        loss = loss_criterion(x_batch.to(device), encoder, torch.Tensor(data).to(device))
        epoch_loss += loss.item()
        i += 1
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss/i, acc/i


def learn_encoder(x, window_size, data, encoding_size, lr=0.001, decay=0, n_epochs=100, device='cpu', n_cross_val=1):
    for cv in range(n_cross_val):
        encoder = CausalCNNEncoder(in_channels=6, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        
        params = encoder.parameters()
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        train_loss, test_loss = [], []
        best_loss = np.inf
        
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], encoder, device, window_size, optimizer=optimizer, train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], encoder, device, window_size, optimizer=optimizer, train=False)
            if epoch%10==0:
                print('\nEpoch ', epoch)
                print('Train ===> Loss: ', epoch_loss, '\t Test ===> Loss: ', epoch_loss_test )
            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            if epoch_loss_test<best_loss:
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict()
                }
                best_loss = epoch_loss_test
                

def triplet_loss(data_type, lr, cv):
    length_of_hour = int(60*60/interval)
    pos_sample_name = 'arrest'
    
    signal_list = ["SpO2", "RESP", "HR", "ABP1", "ABP2", "ABP3"]
    window_size = 60
    encoding_size = 6#16
    n_epochs = 150
    lr = 1e-3

    num_pre_positive_encodings = 3600
    
    train_labels, test_labels = load_labels()
    train_data, test_data = load_data()
    TEST_mixed_data_maps = test_data
    TEST_mixed_labels = test_labels
    train_mixed_data_maps = train_data
    train_mixed_labels = train_labels
    # reshape to (num_samples, 1, num_features, signal_length)
    TEST_mixed_data_maps = torch.reshape(TEST_mixed_data_maps, (TEST_mixed_data_maps.shape[0], 1, TEST_mixed_data_maps.shape[1], TEST_mixed_data_maps.shape[2]))
    train_mixed_data_maps = torch.reshape(train_mixed_data_maps, (train_mixed_data_maps.shape[0], 1, train_mixed_data_maps.shape[1], train_mixed_data_maps.shape[2]))
    
    learn_encoder(train_mixed_data_maps[:,0,:,:], window_size, encoding_size=encoding_size, lr=lr, decay=1e-5, data=data_type, n_epochs=n_epochs, device=device, n_cross_val=1)
    
    classifier_validation_aurocs = []
    classifier_validation_auprcs = []
    classifier_TEST_aurocs = []
    classifier_TEST_auprcs = []
    
    for encoder_cv in range(cv):
        print('Encoder CV: ', encoder_cv)
        seed_val = 111*encoder_cv+2
        random.seed(seed_val)
        print("Seed set to: ", seed_val)
        
        encoder = CausalCNNEncoder(in_channels=6, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        encoder.eval()
        
        # Shuffle data
        print('Original shape of train data: ')
        print(train_mixed_data_maps.shape)
        # shuffle for this cv:
        inds = np.arange(len(train_mixed_data_maps))
        random.shuffle(inds)
        print("First 15 inds: ", inds[:15])
        
        n_valid = int(0.2*len(train_mixed_data_maps))
        validation_mixed_data_maps_cv = train_mixed_data_maps[inds][0:n_valid]
        validation_mixed_labels_cv = train_mixed_labels[inds][0:n_valid]
        print("Size of valid data: ", validation_mixed_data_maps_cv.shape)
        print("Size of valid labels: ", validation_mixed_labels_cv.shape)
        print("num positive valid samples: ", sum([1 in validation_mixed_labels_cv[ind] for ind in range(len(validation_mixed_labels_cv))]))

        train_mixed_data_maps_cv = train_mixed_data_maps[inds][n_valid:]
        train_mixed_labels_cv = train_mixed_labels[inds][n_valid:]
        print("Size of train data: ", train_mixed_data_maps_cv.shape)
        print("Size of train labels: ", train_mixed_labels_cv.shape)
        print("num positive train samples: ", sum([1 in train_mixed_labels_cv[ind] for ind in range(len(train_mixed_labels_cv))]))

        print('Size of TEST data: ', TEST_mixed_data_maps.shape)
        print('Size of TEST labels: ', TEST_mixed_labels.shape)
        print("Num positive TEST samples: ", sum([1 in TEST_mixed_labels[ind] for ind in range(len(TEST_mixed_labels))]))
        
        print("TRAINING LINEAR CLASSIFIER")
        classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv, y_train=train_mixed_labels_cv,
                X_validation=validation_mixed_data_maps_cv, y_validation=validation_mixed_labels_cv, 
                X_TEST=TEST_mixed_data_maps, y_TEST=TEST_mixed_labels, window_size=window_size,
                encoding_size=encoder.encoding_size, batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings,
                encoder=encoder, return_models=True, return_scores=True, pos_sample_name=pos_sample_name,
                data_type=data_type, classification_cv=0, encoder_cv=encoder_cv, plt_path="./DONTCOMMITplots", ckpt_path="./ckpt")
        
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
        
