import numpy as np
import random
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
import torch

from causal_CNN_encoder import CausalCNNEncoder
from load_data import load_data, load_labels
from RNN_predictor import RnnPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
interval = 1


def epoch_run(data_loader, classifier, encoder, window_size, optimizer=None, train=False, num_pre_positive_encodings=None):
    if train:
        encoder.train()
        classifier.train()
    else:
        encoder.eval()
        classifier.eval()

    encoder = encoder.to(device)
    classifier = classifier.to(device)

    epoch_losses = []
    epoch_predictions, epoch_labels = [], []
    for data_batch, label_batch in data_loader:
        data_batch = data_batch[:, :, :, -(3600//interval):] # Take just the last hr of data
        label_batch = label_batch[-(3600//interval):]
        
        data_batch = data_batch.to(device)
        # data is of shape (num_samples, num_encodings_per_sample, encoding_size)
        encoding_batch, encoding_mask = encoder.forward_seq(data_batch, return_encoding_mask=True)
        encoding_batch = encoding_batch.to(device)
        label_batch = label_batch.to(device)
        
            
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
            loss = loss_fn(predictions, label_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(predictions, label_batch.float())
        
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
    
    return epoch_losses, epoch_predictions, epoch_labels


def end_to_end(data_type, n_cv):
    overal_loss, overal_auc, overal_auprc = [], [], []
    
    for cv in range(n_cv):
        random.seed(111*cv+2)
        window_size = 60
        signal_list = ["SpO2", "RESP", "HR", "ABP1", "ABP2", "ABP3"]
#         pre_positive_window = int(2*(60*60/interval))
        num_pre_positive_encodings = 3600
        pos_sample_name = 'Arrest'
        
        train_labels, test_labels = load_labels()
        train_data, test_data = load_data()
        TEST_mixed_data_maps = test_data
        TEST_mixed_labels = test_labels
        train_mixed_data_maps = train_data
        train_mixed_labels = train_labels
        # reshape to (num_samples, 1, num_features, signal_length)
        TEST_mixed_data_maps = torch.reshape(TEST_mixed_data_maps, (TEST_mixed_data_maps.shape[0], 1, TEST_mixed_data_maps.shape[1], TEST_mixed_data_maps.shape[2]))
        train_mixed_data_maps = torch.reshape(train_mixed_data_maps, (train_mixed_data_maps.shape[0], 1, train_mixed_data_maps.shape[1], train_mixed_data_maps.shape[2]))
        
        x_test = TEST_mixed_data_maps
        y_test = TEST_mixed_labels # Just grab last label, that's all we need for classifier
        x_train = train_mixed_data_maps
        y_train = train_mixed_labels
        # Create model
        classifier = RnnPredictor(encoding_size=6, hidden_size=8)
        encoder = CausalCNNEncoder(in_channels=6, channels=8, depth=2, reduced_size=30, encoding_size=6, kernel_size=3, device=device, window_size=window_size)
        
        inds = list(range(len(x_train)))
        random.shuffle(inds)
        x_train = x_train[inds]
        y_train = y_train[inds]
        n_train = int(0.2 * len(inds))
        
        trainset = torch.utils.data.TensorDataset(x_train[n_train:], y_train[n_train:])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        validset = torch.utils.data.TensorDataset(x_train[:n_train], y_train[:n_train])
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)
        testset = torch.utils.data.TensorDataset(x_test[:], y_test[:])
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
        
        params = list(classifier.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=0.0005)
        train_loss_trend, valid_loss_trend = [], []
        
        for epoch in range(1, 101):
            train_epoch_losses, train_pred_all, train_y_all = epoch_run(train_loader, classifier, encoder, window_size=window_size, optimizer=optimizer, train=True, num_pre_positive_encodings=None)
            valid_epoch_losses, valid_pred_all, valid_y_all = epoch_run(valid_loader, classifier, encoder, window_size=window_size, optimizer=optimizer, train=False, num_pre_positive_encodings=None)
            
            
            train_loss = np.mean(train_epoch_losses)
            valid_loss = np.mean(valid_epoch_losses)
            train_pred_all = torch.cat(train_pred_all)
            train_y_all = torch.cat(train_y_all)
            valid_pred_all = torch.cat(valid_pred_all)
            valid_y_all = torch.cat(valid_y_all)


            train_auc = roc_auc_score(train_y_all, train_pred_all)
            train_auprc = average_precision_score(train_y_all, train_pred_all)

            valid_auc = roc_auc_score(valid_y_all, valid_pred_all)
            valid_auprc = average_precision_score(valid_y_all, valid_pred_all)
            
            
            train_loss_trend.append(train_loss)
            valid_loss_trend.append(valid_loss)
            
            if epoch%10==0:
                print('***** Epoch %d *****' % epoch)
                print('Training Loss: %.3f \t Training AUROC: %.3f  \t Training AUPRC: %.3f'
                    '\t Valid Loss: %.3f \t Valid AUROC: %.3f \t Valid AUPRC: %.3f' % (train_loss, train_auc, train_auprc, valid_loss, valid_auc, valid_auprc))
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'loss': valid_loss,
                    'auc': valid_auc
                }

        test_epoch_losses, test_pred_all, test_y_all = epoch_run(test_loader, classifier, encoder, window_size=window_size, optimizer=optimizer, train=False, num_pre_positive_encodings=None)

        test_pred_all = torch.cat(test_pred_all)
        test_y_all = torch.cat(test_y_all)
        
        test_auc = roc_auc_score(test_y_all, test_pred_all)
        test_auprc = average_precision_score(test_y_all, test_pred_all)

        test_loss = np.mean(test_epoch_losses)
        overal_loss.append(test_loss)
        overal_auc.append(test_auc)
        overal_auprc.append(test_auprc)
        

        train_pred_all[train_pred_all > 0.5] = 1
        train_pred_all[train_pred_all <= 0.5] = 0
        
        valid_pred_all[valid_pred_all > 0.5] = 1
        valid_pred_all[valid_pred_all <= 0.5] = 0

        test_pred_all[test_pred_all > 0.5] = 1
        test_pred_all[test_pred_all <= 0.5] = 0
        
        print("Train classification report: ")
        print('train_y_all shape: ', train_y_all.shape, 'train_pred_all shape: ', train_pred_all.shape)
        print(classification_report(train_y_all.to('cpu'), train_pred_all, target_names=['normal', pos_sample_name]))
        print("Validation classification report: ")
        print(classification_report(valid_y_all.to('cpu'), valid_pred_all, target_names=['normal', pos_sample_name]))
        print()
        print("TEST classification report: ")
        print(classification_report(test_y_all.to('cpu'), test_pred_all, target_names=['normal', pos_sample_name]))
        print()
        
        print()
        
    print('Final Test performance:\t Loss: %.3f +- %.3f \t Test AUROC: %.3f +- %.3f \t Test AUPRC: %.3f +- %.3f' %
        (np.mean(overal_loss), np.std(overal_loss), np.mean(overal_auc), np.std(overal_auc), np.mean(overal_auprc),
         np.std(overal_auprc)))
