import numpy as np
import random
import torch

from causal_CNN_encoder import CausalCNNEncoder
from load_data import load_data, load_labels
from trace_model import train_linear_classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
interval = 1

def epoch_run(data, ds_estimator, auto_regressor, encoder, device, window_size, n_size=5, optimizer=None, train=True):
    if train:
        encoder.train()
        ds_estimator.train()
        auto_regressor.train()
    else:
        encoder.eval()
        ds_estimator.eval()
        auto_regressor.eval()
    encoder.to(device)
    ds_estimator.to(device)
    auto_regressor.to(device)
    
    epoch_loss = 0
    acc = 0
    for n_i, sample in enumerate(data):
        # Recall each sample is of shape (2, num_features, signal_length)
        rnd_t = np.random.randint(5*window_size, sample.shape[-1]-5*window_size) # Choose random time in the timeseries
        sample = torch.Tensor(sample[..., max(0,(rnd_t-20*window_size)):min(sample.shape[-1], rnd_t+20*window_size)]) # sample is now redefined as being of length 40*window_size centered at rnd_t

        T = sample.shape[-1]
        windowed_sample = np.split(sample[..., :(T // window_size) * window_size], (T // window_size), -1) # splits the sample into window_size pieces
    
        windowed_sample = torch.tensor(np.stack(windowed_sample, 0), device=device) # of shape (num_samples, num_features, window_size) where num_samples = ~40
        encodings = encoder(windowed_sample) # of shape (num_samples, encoding_size)
        
        window_ind = torch.randint(2,len(encodings)-2, size=(1,)) # window_ind is the last window we'll have access to in the AR model. After that, we want to predict the future
        _, c_t = auto_regressor(encodings[max(0, window_ind[0]-10):window_ind[0]+1].unsqueeze(0)) # Feeds the last 10 encodings preceding and including the window_ind windowed sample into the AR model.
        density_ratios = torch.bmm(encodings.unsqueeze(1),
                                       ds_estimator(c_t.squeeze(1).squeeze(0)).expand_as(encodings).unsqueeze(-1)).view(-1,) # Just take the dot product of the encodings and the output of the estimator?
        r = set(range(0, window_ind[0] - 2))
        r.update(set(range(window_ind[0] + 3, len(encodings))))
        rnd_n = np.random.choice(list(r), n_size)
        X_N = torch.cat([density_ratios[rnd_n], density_ratios[window_ind[0] + 1].unsqueeze(0)], 0)
        if torch.argmax(X_N)==len(X_N)-1:
            acc += 1
        labels = torch.Tensor([len(X_N)-1]).to(device)
        loss = torch.nn.CrossEntropyLoss()(X_N.view(1, -1), labels.long())
        epoch_loss += loss.item()
        if n_i%20==0:
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return epoch_loss / len(data), acc/(len(data))


def learn_encoder(x, window_size, encoding_size, lr=0.001, decay=0, n_size=5, n_epochs=50, data='simulation', device='cpu', n_cross_val=1):
    accuracies = []
    for cv in range(n_cross_val):
        encoder = CausalCNNEncoder(in_channels=6, channels=8, depth=2, reduced_size=30, encoding_size=encoding_size, kernel_size=3, window_size=window_size, device=device)
        
        ds_estimator = torch.nn.Linear(encoder.encoding_size, encoder.encoding_size) # Predicts future latent data given context vector
        auto_regressor = torch.nn.GRU(input_size=encoding_size, hidden_size=encoding_size, batch_first=True)
        params = list(ds_estimator.parameters()) + list(encoder.parameters()) + list(auto_regressor.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        best_acc = 0
        best_loss = np.inf
        train_loss, test_loss = [], []
        
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], ds_estimator, auto_regressor, encoder, device, window_size, optimizer=optimizer,
                                        n_size=n_size, train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], ds_estimator, auto_regressor, encoder, device, window_size, n_size=n_size, train=False)
            if epoch%20==0:
                print('\nEpoch ', epoch)
                print('Train ===> Loss: ', epoch_loss, '\t Accuracy: ', acc)
                print('Test ===> Loss: ', epoch_loss_test, '\t Accuracy: ', acc_test)
            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            if epoch_loss_test<best_loss:
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict()
                }
                best_loss = epoch_loss_test
                best_acc = acc_test
        
        accuracies.append(best_acc)
        
    print('=======> Performance Summary:')
    print('Accuracy: %.2f +- %.2f' % (100 * np.mean(accuracies), 100 * np.std(accuracies)))
        

def cpc(data_type, lr,  cv):
    length_of_hour = int(60*60/interval)
    window_size = 60
    encoding_size = 6#16
    n_epochs = 400
    lr = 1e-3
    pos_sample_name = 'arrest'
    
    signal_list = ["SpO2", "RESP", "HR", "ABP1", "ABP2", "ABP3"]
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
        # print("First 15 inds: ", inds[:15])
        
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


        print("TRAINING LINEAR CLASSIFIER")
        
        classifier, valid_auroc, valid_auprc, TEST_auroc, TEST_auprc = train_linear_classifier(X_train=train_mixed_data_maps_cv, y_train=train_mixed_labels_cv,
                X_validation=validation_mixed_data_maps_cv, y_validation=validation_mixed_labels_cv,
                X_TEST=TEST_mixed_data_maps, y_TEST=TEST_mixed_labels, window_size=window_size,
                encoding_size=encoding_size, batch_size=20, num_pre_positive_encodings=num_pre_positive_encodings,
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

    