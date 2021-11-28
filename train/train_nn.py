import time, os
import math
import matplotlib.pyplot as plt
import torch
from torch import save, set_grad_enabled, sum, max
from torch import optim, cuda, load, device
from torch.nn.utils import clip_grad_norm_
from preprocessing.nn_dataset import get_data_set, TOTAL_BATCHES, TRAIN_BATCHES, N_TOKENS
from train.models import CrossEntropyTimeDistributedLoss
from train.models import TonicNet, Transformer_Model
from train.external import RAdam, Lookahead, OneCycleLR
import json
import numpy as np

"""
File containing functions which train various neural networks defined in train.models
"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub
    for k,v in obj.items():
        if isinstance(v, list):
            obj[k] = np.array(v)
            if obj[k].dtype==np.float32:
                obj[k] = obj[k].astype(np.float64)
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)

CV_PHASES = ['train', 'val']
TRAIN_ONLY_PHASES = ['train']


# MARK:- TonicNet
def TonicNet_lr_finder(train_emb_freq=3000, load_path=''):
    train_TonicNet(epochs=3,
                         save_model=False,
                         load_path=load_path,
                         shuffle_batches=True,
                         num_batches=TRAIN_BATCHES,
                         val=False,
                    train_emb_freq=train_emb_freq,
                         lr_range_test=True)


def TonicNet_sanity_test(num_batches=1, train_emb_freq=3000, load_path=''):
    train_TonicNet(epochs=200,
                         save_model=False,
                         load_path=load_path,
                         shuffle_batches=False,
                         num_batches=num_batches,
                         val=1,
                    train_emb_freq=train_emb_freq,
                         lr_range_test=False,
                    sanity_test=True)


def train_TonicNet(epochs,
                    save_model=True,
                    load_path='',
                    shuffle_batches=False,
                    num_batches=TOTAL_BATCHES,
                    val=True,
                    train_emb_freq=1,
                    lr_range_test=False,
                    sanity_test=False):

    model = TonicNet(nb_tags=N_TOKENS, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.3)

    if load_path != '':
        try:
            if cuda.is_available():
                model.load_state_dict(load(load_path)['model_state_dict'])
            else:
                model.load_state_dict(load(load_path, map_location=device('cpu'))['model_state_dict'])
            print("loded params from", load_path)
        except:
            raise ImportError(f'No file located at {load_path}, could not load parameters')
    print(model)

    if cuda.is_available():
        model.cuda()

    base_lr = 0.2
    max_lr = 0.2

    if lr_range_test:
        base_lr = 0.000003
        max_lr = 0.5

    step_size = 3 * min(TRAIN_BATCHES, num_batches)

    if sanity_test:
        base_optim = RAdam(model.parameters(), lr=base_lr)
        optimiser = Lookahead(base_optim, k=5, alpha=0.5)
    else:
        optimiser = optim.SGD(model.parameters(), base_lr)
    criterion = CrossEntropyTimeDistributedLoss()

    print(criterion)

    print(f"min lr: {base_lr}, max_lr: {max_lr}, stepsize: {step_size}")

    if not sanity_test and not lr_range_test:
        scheduler = OneCycleLR(optimiser, max_lr,
                                                  epochs=60, steps_per_epoch=TRAIN_BATCHES, pct_start=0.3,
                                                  anneal_strategy='cos', cycle_momentum=True, base_momentum=0.8,
                                                  max_momentum=0.95, div_factor=25.0, final_div_factor=1000.0,
                                                  last_epoch=-1)

    elif lr_range_test:
        lr_lambda = lambda x: math.exp(x * math.log(max_lr / base_lr) / (epochs * num_batches))
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    best_val_loss = 100.0

    if lr_range_test:
        lr_find_loss = []
        lr_find_lr = []

        itr = 0
        smoothing = 0.05

    if val:
        phases = CV_PHASES
    else:
        phases = TRAIN_ONLY_PHASES

    results = {
        'all_val_losses': [],
        'all_val_masked_losses': [],
        'all_val_tot_losses': [],
        'all_val_accs': [],
        'all_val_masked_accs': [],
        'all_val_tot_accs': []
    }

    for epoch in range(epochs):
        start = time.time()
        pr_interval = 50

        print(f'Beginning EPOCH {epoch + 1}')

        for phase in phases:

            count = 0
            batch_count = 0
            batch_count_masked = 0
            batch_count_tot = 0
            loss_epoch = 0
            loss_epoch_masked = 0
            loss_epoch_all = 0
            running_accuracy = 0.0
            running_masked_acc = 0.0 # eval
            running_tot_acc = 0.0 # eval
            running_batch_count = 0
            print_loss_batch = 0.0  # Reset on print
            print_acc_batch = 0.0  # Reset on print

            print(f'\n\tPHASE: {phase}')

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for x, y, psx, i, c, mask in get_data_set(phase, shuffle_batches=shuffle_batches, return_I=1):
                Y = y
                model.zero_grad() 

                if phase == 'train' and (epoch > -1 or load_path != ''):
                    if train_emb_freq < 1000:
                        train_emb = ((count % train_emb_freq) == 0)
                    else:
                        train_emb = False

                else:
                    train_emb = False

                with set_grad_enabled(phase == 'train'):
                    mask = mask.to(torch.int)
                    y_hat = model(x*mask, z=i*mask.squeeze(), train_embedding=train_emb)
                    
                    # training: use only voices that are NOT masked
                    mask = mask.to(torch.bool)
                    if phase=='train':
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss = criterion(masked_yhat, masked_y, )
                        _, preds = max(masked_yhat, 2)
                        running_accuracy += sum(preds == masked_y).item()
                        print_acc_batch += sum(preds == masked_y).item()
                        loss_epoch += loss.item()
                        print_loss_batch += loss.item()
                        batch_count += masked_yhat.shape[1]
                        running_batch_count += masked_yhat.shape[1]

                    else:
                        # evaluation: use only the voices that are NOT masked (used for validation selection)
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss = criterion(masked_yhat, masked_y, )
                        _, preds = max(masked_yhat, 2)
                        running_accuracy += sum(preds == masked_y).item()
                        print_acc_batch += sum(preds == masked_y).item()
                        loss_epoch += loss.item()
                        print_loss_batch += loss.item()
                        batch_count += masked_yhat.shape[1]
                        running_batch_count += masked_yhat.shape[1]
                        
                        # evaluation: use only the voices that ARE masked
                        mask = ~(mask.reshape(-1))
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss_masked = criterion(masked_yhat, masked_y)
                        _, preds = max(masked_yhat, 2)
                        running_masked_acc += sum(preds == masked_y).item()
                        loss_epoch_masked += loss_masked.item()
                        batch_count_masked += masked_yhat.shape[1]

                        # evaluation: use all voices
                        mask = torch.ones_like(mask)
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss_all = criterion(masked_yhat, masked_y)
                        _, preds = max(masked_yhat, 2)
                        running_tot_acc += sum(preds == masked_y).item()
                        loss_epoch_all += loss_all.item()
                        batch_count_tot += masked_yhat.shape[1]
                    
                count += 1
                if phase == 'train':
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 5)
                    optimiser.step()
                    if not sanity_test:
                        scheduler.step()

                if lr_range_test:
                    lr_step = optimiser.state_dict()["param_groups"][0]["lr"]
                    lr_find_lr.append(lr_step)

                    # smooth the loss
                    if itr == 0:
                        lr_find_loss.append(min(loss, 7))
                    else:
                        loss = smoothing * min(loss, 7) + (1 - smoothing) * lr_find_loss[-1]
                        lr_find_loss.append(loss)

                    itr += 1

                # print loss for recent set of batches
                if count % pr_interval == 0:
                    ave_loss = print_loss_batch/pr_interval
                    ave_acc = 100 * float(print_acc_batch)/running_batch_count
                    print_acc_batch = 0
                    running_batch_count = 0
                    print('\t\t[%d] loss: %.3f, acc: %.3f' % (count, ave_loss, ave_acc))
                    print_loss_batch = 0
                    # break

                if count == num_batches:
                    break

            # calculate loss and accuracy for phase
            if phase=='train':
                # calculate loss and accuracy for phase
                ave_loss_epoch = loss_epoch/count
                epoch_acc = 100 * float(running_accuracy) / batch_count
                print('\tfinished %s phase [%d] loss: %.3f, acc: %.3f' % (phase, epoch + 1, ave_loss_epoch, epoch_acc))
            else:
                # not masked
                val_loss = loss_epoch/count
                epoch_acc = 100 * float(running_accuracy) / batch_count

                # masked
                val_loss_masked = loss_epoch_masked/count
                epoch_acc_masked = 100 * float(running_masked_acc) / batch_count_masked

                # all
                val_loss_all = loss_epoch_all/count
                epoch_acc_all = 100 * float(running_tot_acc) / batch_count_tot

                results['all_val_losses'].append(val_loss)
                results['all_val_masked_losses'].append(val_loss_masked)
                results['all_val_tot_losses'].append(val_loss_all)
                results['all_val_accs'].append(epoch_acc)
                results['all_val_masked_accs'].append(epoch_acc_masked)
                results['all_val_tot_accs'].append(epoch_acc_all)

                print(f'\t Validation perf (on train subset of features): {val_loss:.3f}, {epoch_acc:.3f}')
                print(f'\t Validation perf (on masked subset): {val_loss_masked:.3f}, {epoch_acc_masked:.3f}')
                print(f'\t Validation perf (on total subset): {val_loss_all:.3f}, {epoch_acc_all:.3f}')

    save_json('tonicnet_results.json', results)


# MARK:- Transformer
def Transformer_lr_finder(load_path=''):
    train_Transformer(epochs=3,
                         save_model=False,
                         load_path=load_path,
                         shuffle_batches=True,
                         num_batches=TRAIN_BATCHES,
                         val=False,
                         lr_range_test=True)


def Transformer_sanity_test(num_batches=1, load_path=''):
    train_Transformer(epochs=1000,
                         save_model=0,
                         load_path=load_path,
                         shuffle_batches=False,
                         num_batches=num_batches,
                         val=1,
                         lr_range_test=False,
                    sanity_test=True)


def train_Transformer(epochs,
                    save_model=True,
                    load_path='',
                    shuffle_batches=False,
                    num_batches=TOTAL_BATCHES,
                    val=True,
                    lr_range_test=False,
                    sanity_test=False):

    model = Transformer_Model(nb_tags=N_TOKENS, nb_layers=5, emb_dim=256, dropout=0.1, pe_dim=256)

    if load_path != '':
        try:
            if cuda.is_available():
                model.load_state_dict(load(load_path)['model_state_dict'])
            else:
                model.load_state_dict(load(load_path, map_location=device('cpu'))['model_state_dict'])
            print("loded params from", load_path)
        except:
            raise ImportError(f'No file located at {load_path}, could not load parameters')
    print(model)

    if cuda.is_available():
        model.cuda()

    base_lr = 0.06
    max_lr = 0.06

    if lr_range_test:
        base_lr = 0.000003
        max_lr = 0.3

    step_size = 3 * min(TRAIN_BATCHES, num_batches)

    if sanity_test:
        base_optim = RAdam(model.parameters(), lr=base_lr/100)
        optimiser = Lookahead(base_optim, k=5, alpha=0.5)
    else:
        optimiser = optim.SGD(model.parameters(), base_lr)
    criterion = CrossEntropyTimeDistributedLoss()

    print(criterion)

    print(f"min lr: {base_lr}, max_lr: {max_lr}, stepsize: {step_size}")

    if not sanity_test and not lr_range_test:
        scheduler = OneCycleLR(optimiser, max_lr,
                                                  epochs=30, steps_per_epoch=TRAIN_BATCHES, pct_start=0.3,
                                                  anneal_strategy='cos', cycle_momentum=True, base_momentum=0.8,
                                                  max_momentum=0.95, div_factor=100.0, final_div_factor=1000.0,
                                                  last_epoch=-1)

    elif lr_range_test:
        lr_lambda = lambda x: math.exp(x * math.log(max_lr / base_lr) / (epochs * TRAIN_BATCHES))
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    best_val_loss = 100.0
    step = 0

    if lr_range_test:
        lr_find_loss = []
        lr_find_lr = []

        itr = 0
        smoothing = 0.05

    if val:
        phases = CV_PHASES
    else:
        phases = TRAIN_ONLY_PHASES

    results = {
        'all_val_losses': [],
        'all_val_masked_losses': [],
        'all_val_tot_losses': [],
        'all_val_accs': [],
        'all_val_masked_accs': [],
        'all_val_tot_accs': []
    }

    for epoch in range(epochs):
        start = time.time()
        pr_interval = 50

        print(f'Beginning EPOCH {epoch + 1}')

        for phase in phases:
            phase_loss = None
            model.zero_grad()
            count = 0
            batch_count = 0
            batch_count_masked = 0
            batch_count_tot = 0
            loss_epoch = 0
            loss_epoch_masked = 0
            loss_epoch_all = 0
            running_accuracy = 0.0
            running_masked_acc = 0.0 # eval
            running_tot_acc = 0.0 # eval
            running_batch_count = 0
            print_loss_batch = 0.0  # Reset on print
            print_acc_batch = 0.0  # Reset on print

            print(f'\n\tPHASE: {phase}')

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for x, y, psx, i, c, mask in get_data_set(phase, shuffle_batches=shuffle_batches, return_I=1):
                Y = y
                model.seq_len = x.shape[1]

                with set_grad_enabled(phase == 'train'):
                    # MASK input - turn into zeros where mask 
                    mask = mask.to(torch.int)
                    y_hat = model(x*mask, psx*mask.squeeze(-1))
                    y_hat = y_hat.view(1, -1, N_TOKENS)
                    
                    # training: use only voices that are NOT masked
                    mask = mask.to(torch.bool)
                    if phase=='train':
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss = criterion(masked_yhat, masked_y, )
                        _, preds = max(masked_yhat, 2)
                        running_accuracy += sum(preds == masked_y).item()
                        print_acc_batch += sum(preds == masked_y).item()
                        loss_epoch += loss.item()
                        print_loss_batch += loss.item()
                        batch_count += masked_yhat.shape[1]
                        running_batch_count += masked_yhat.shape[1]

                    else:
                        # evaluation: use only the voices that are NOT masked (used for validation selection)
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss = criterion(masked_yhat, masked_y, )
                        _, preds = max(masked_yhat, 2)
                        running_accuracy += sum(preds == masked_y).item()
                        print_acc_batch += sum(preds == masked_y).item()
                        loss_epoch += loss.item()
                        print_loss_batch += loss.item()
                        batch_count += masked_yhat.shape[1]
                        running_batch_count += masked_yhat.shape[1]
                        
                        # evaluation: use only the voices that ARE masked
                        mask = ~(mask.reshape(-1))
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss_masked = criterion(masked_yhat, masked_y)
                        _, preds = max(masked_yhat, 2)
                        running_masked_acc += sum(preds == masked_y).item()
                        loss_epoch_masked += loss_masked.item()
                        batch_count_masked += masked_yhat.shape[1]

                        # evaluation: use all voices
                        mask = torch.ones_like(mask)
                        masked_yhat = y_hat.squeeze().masked_select(mask.squeeze()[:,None].expand(-1, y_hat.shape[-1])).reshape(-1, y_hat.shape[-1])[None,:]
                        masked_y = Y.squeeze().masked_select(mask.squeeze())[None,:]
                        loss_all = criterion(masked_yhat, masked_y)
                        _, preds = max(masked_yhat, 2)
                        running_tot_acc += sum(preds == masked_y).item()
                        loss_epoch_all += loss_all.item()
                        batch_count_tot += masked_yhat.shape[1]

                    if phase_loss is None:
                        phase_loss = loss
                    else:
                        phase_loss += loss

                count += 1

                if count % 1 == 0:
                    if phase == 'train':
                        phase_loss.backward()
                        clip_grad_norm_(model.parameters(), 5)
                        optimiser.step()
                        step += 1
                        if not sanity_test:
                            scheduler.step()
                        phase_loss = None
                        model.zero_grad()

                if lr_range_test:
                    lr_step = optimiser.state_dict()["param_groups"][0]["lr"]
                    lr_find_lr.append(lr_step)

                    # smooth the loss
                    if itr == 0:
                        lr_find_loss.append(min(loss, 4))
                    else:
                        loss = smoothing * min(loss, 4) + (1 - smoothing) * lr_find_loss[-1]
                        lr_find_loss.append(loss)

                    itr += 1

                # print loss for recent set of batches
                if count % pr_interval == 0:
                    ave_loss = print_loss_batch/pr_interval
                    ave_acc = 100 * print_acc_batch/running_batch_count
                    print_acc_batch = 0
                    running_batch_count = 0
                    print('\t\t[%d] loss: %.3f, acc: %.3f' % (count, ave_loss, ave_acc))
                    print_loss_batch = 0
                    # break

                if count == num_batches:
                    break

            if phase=='train':
                # calculate loss and accuracy for phase
                ave_loss_epoch = loss_epoch/count
                epoch_acc = 100 * float(running_accuracy) / batch_count
                print('\tfinished %s phase [%d] loss: %.3f, acc: %.3f' % (phase, epoch + 1, ave_loss_epoch, epoch_acc))
            else:
                # not masked
                val_loss = loss_epoch/count
                epoch_acc = 100 * float(running_accuracy) / batch_count
                
                # masked
                val_loss_masked = loss_epoch_masked/count
                epoch_acc_masked = 100 * float(running_masked_acc) / batch_count_masked

                # all
                val_loss_all = loss_epoch_all/count
                epoch_acc_all = 100 * float(running_tot_acc) / batch_count_tot

                results['all_val_losses'].append(val_loss)
                results['all_val_masked_losses'].append(val_loss_masked)
                results['all_val_tot_losses'].append(val_loss_all)
                results['all_val_accs'].append(epoch_acc)
                results['all_val_masked_accs'].append(epoch_acc_masked)
                results['all_val_tot_accs'].append(epoch_acc_all)

                print(f'\t Validation perf (on train subset of features): {val_loss:.3f}, {epoch_acc:.3f}')
                print(f'\t Validation perf (on masked subset): {val_loss_masked:.3f}, {epoch_acc_masked:.3f}')
                print(f'\t Validation perf (on total subset): {val_loss_all:.3f}, {epoch_acc_all:.3f}')

        print('\n\ttime:', __time_since(start), '\n')


    save_json('transformer_results.json', results)


def __save_model(epoch, ave_loss_epoch, model, model_name, acc):

        test = os.listdir('eval')

        for item in test:
            if item.endswith(".pt"):
                os.remove(os.path.join('eval', item))

        path_loss = round(ave_loss_epoch, 3)
        path_acc = '%.3f' % acc
        path = f'eval/{model_name}_epoch-{epoch}_loss-{path_loss}_acc-{path_acc}.pt'
        save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': path_loss
        }, path)
        print("\tSAVED MODEL TO:", path)


def __time_since(t):
    now = time.time()
    s = now - t
    return '%s' % (__as_minutes(s))


def __as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
