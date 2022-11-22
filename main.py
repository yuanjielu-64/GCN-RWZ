import time
from configs.config import get_config
import torch.nn as nn
import argparse
import torch
from utils import util
from fastprogress import master_bar, progress_bar
from fastprogress import fastprogress
fastprogress.printing = lambda: True
from torch.optim import lr_scheduler
from model.GCNRWZ_NET import GCNRWZ_model

import numpy as np
import os.path as osp
import pandas as pd

def metrics_to_str(epoch, time, learning_rate, mae, rmse, mape, cmae, crmse, cmape, epoch_loss, conAccuracy, nonconAccuracy):
    result = ( 'epoch:{epoch:0>2d}\t'
               'time:{time:.3f}s\t'
               'lr:{learning_rate:.4e}\t'
               'loss:{loss:.4f}\t'
               'rmse:{rmse:.2f}\t'
               'mae:{mae:.2f}\t'
               'mape (%):{mape:.2f}\t'
               'Accuracy during clean (%):{noncon:.2f}\t'
               'con_rmse:{crmse:.2f}\t'
               'con_mae:{cmae:.2f}\t'
               'con_mape (%):{cmape:.2f}\t'
               'Accuracy during con (%):{con:.2f}\t'
               .format(
                   epoch= epoch + 1,
                   time=time,
                   learning_rate=learning_rate,
                   loss=epoch_loss,
                   rmse=rmse,
                   mae=mae,
                   mape=mape * 100,
                   noncon=nonconAccuracy,
                   crmse=crmse,
                   cmae=cmae,
                   cmape=cmape * 100,
                   con=conAccuracy
                   ))
    return result

def loss_function(criterion, out, y):

    label = y[:, :, :, 0]
    mask = y[:, :, :, -1]

    loss = criterion[2](out * mask.float(), label * mask.float())
    return loss

def train(net, train_loader, criterion, optimizer, config, epoch_bar, normalize_list):
    net.train()
    loss_iter = 0.0
    mae, rmse, mape, con_mae, con_rmse, con_mape = 0, 0, 0, 0, 0, 0
    label_iter = []
    pred_iter = []
    con_iter = []
    start = time.time()
    dataset = iter(train_loader)

    for batch_num in progress_bar(range(len(train_loader)), parent= epoch_bar):
        optimizer.zero_grad()
        try:
            x, y = next(dataset)
        except StopIteration:
            break

        out = net(x)

        loss = loss_function(criterion, out, y)

        pred_iter.append(out.detach().cpu().numpy())
        label_iter.append(y[:, :, :, 0].detach().cpu().numpy())
        con_iter.append(y[:, :, :, 1].detach().cpu().numpy())

        loss_iter += loss.item()

        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0 or batch_num == len(train_loader) - 1:
            mae, rmse, mape, con_mae, con_rmse, con_mape, conAccuracy, nonconAccuracy = util.metric(config, np.concatenate(pred_iter, axis = 0), np.concatenate(label_iter, axis = 0), np.concatenate(con_iter, axis = 0) , normalize_list)
            result = ('Train {} - Each 20 iter: loss:{loss:.4f}; rmse: {rmse:.2f} - {crmse:.2f}; mae: {mae:.2f} - {cmae:.2f}; mape (%): {mape:.2f} - {cmape:.2f}; conAccuracy (%): {con:.2f}; nonconAccuracy (%): {noncon:.2f}'.format(
                config.main_feature,
                loss= loss_iter / (batch_num + 1),
                rmse = rmse,
                crmse = con_rmse,
                mae = mae,
                cmae = con_mae,
                mape = mape * 100,
                cmape = con_mape * 100,
                con = conAccuracy,
                noncon = nonconAccuracy,
                ))
            epoch_bar.child.comment = result

    return time.time() - start, mae, rmse, mape, con_mae, con_rmse, con_mape, loss_iter / len(train_loader), loss_iter, conAccuracy, nonconAccuracy

def val_test(net, loader, criterion, config, epoch_bar, normalize_list, type = "Validation"):
    net.eval()
    loss_iter = 0.0
    label_iter = []
    pred_iter = []
    con_iter = []
    start = time.time()
    dataset = iter(loader)

    with torch.no_grad():
        for batch_num in progress_bar(range(len(loader)), parent=epoch_bar):
            try:
                x, y = next(dataset)
            except StopIteration:
                break

            out = net(x)
            # Change to real speed and write
            pred_iter.append(out.detach().cpu().numpy())
            label_iter.append(y[:, :, :, 0].detach().cpu().numpy())
            con_iter.append(y[:, :, :, 1].detach().cpu().numpy())

            loss = loss_function(criterion, out, y)

            loss_iter += loss.item()

            mae, rmse, mape, con_mae, con_rmse, con_mape, conAccuracy, nonconAccuracy = util.metric(config, np.concatenate(pred_iter, axis=0), np.concatenate(label_iter, axis=0), np.concatenate(con_iter, axis = 0),
                                          normalize_list)

            result = (
            '{} {} - Each 20 iter: loss:{loss:.4f}; rmse: {rmse:.2f} - {crmse:.2f}; mae: {mae:.2f} - {cmae:.2f}; mape (%): {mape:.2f} - {cmape:.2f}; conAccuracy (%): {con:.2f}; nonconAccuracy (%): {noncon:.2f}'.format(
                type,
                config.main_feature,
                loss= loss_iter / (batch_num + 1),
                rmse = rmse,
                crmse = con_rmse,
                mae = mae,
                cmae = con_mae,
                mape = mape * 100,
                cmape = con_mape * 100,
                con=conAccuracy,
                noncon=nonconAccuracy,
            ))
            epoch_bar.child.comment = result

    return time.time() - start, mae, rmse, mape, con_mae, con_rmse, con_mape, loss_iter / (len(loader) + 1), pred_iter, label_iter, loss_iter, conAccuracy, nonconAccuracy

def save_result(net, test_no_shuffle_loader, normalize_list, config):
    pred_iter = []
    label_iter = []
    net.eval()

    with torch.no_grad():
        dataset = iter(test_no_shuffle_loader)
        for i in range(len(test_no_shuffle_loader)):
            try:
                x, y = next(dataset)
            except StopIteration:
                break
            if len(x) < config.batch_size:
                break
            out = net(x)
            # Change to real speed and write
            pred_iter.append(out.detach().cpu().numpy())
            label_iter.append(y[:, :, :, 0].detach().cpu().numpy())

        pred = np.mean(np.concatenate(pred_iter, axis=0), 1)
        label = np.mean(np.concatenate(label_iter, axis=0), 1)
        normalize_feature = normalize_list[config.main_feature]
        pred = np.around(normalize_feature.inverse_transform(pred), 2)
        label = np.around(normalize_feature.inverse_transform(label), 2)

        zero = np.where(label == 0.0)
        for i in range(len(zero[0])):
            pred[zero[0][i], zero[1][i]] = 0
        lens = len(pred_iter) * config.batch_size
        output_time = pd.DataFrame(test_no_shuffle_loader.dataset.output_time[:lens])[0]
        pred_result = pd.DataFrame(pred, index=output_time)
        label_result = pd.DataFrame(label, index=output_time)

        pred_result.to_csv(config.expt_name + '_' + config.model_name + '_' + str(config.P) + "_pred.csv", index_label=None)
        label_result.to_csv(config.expt_name + '_' + config.model_name + '_' + str(config.P) + "_label.csv", index_label=None)

def main(config):
    # create log directory
    writer = util.writer(config)
    train_loader, val_loader, test_loader, test_no_shuffle_loader, dis, D, normalize_list = util.load_data(config)
    net = GCNRWZ_model(config, dis, D).to(config.device)
    total_params = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of parameters is: %d" % total_params)
    print("Total number of trainable parameters is: %d" % train_params)
    path = osp.join("save/", config.expt_name + '_' + config.model_name + '_' + str(config.P))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=0)

    if osp.exists(path) is not True:
        epoch = 0
        criterion = [nn.MSELoss().cuda(), nn.L1Loss().cuda(), nn.HuberLoss().cuda()]

        epoch_bar = master_bar(range(epoch, config.max_epochs))
    else:
        print("Loading the checkpoint...")
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        epoch = checkpoint['epoch']
        criterion = checkpoint['loss']
        epoch_bar = master_bar(range(epoch, config.max_epochs))
        print("The current epoch is: %d " % epoch)

    # Train + validate + test
    scheduler = lr_scheduler.StepLR(optimizer, step_size= config.learning_rate_step_size, gamma=0.8)
    stop = 0
    best_val_loss = np.inf

    epoch_set, train_loss_set, train_rmse_set = [], [], []
    val_loss_set, val_rmse_set, val_mae_set, val_mape_set, val_crmse_set, val_cmae_set, val_cmape_set, val_con_accuracy, val_noncon_accuracy = [], [], [], [], [], [], [], [], []
    test_rmse_set, test_mae_set, test_mape_set, test_crmse_set, test_cmae_set, test_cmape_set, test_con_accuracy, test_noncon_accuracy = [], [], [], [], [], [], [], []

    for epoch in epoch_bar:
        #record learning rate
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        scheduler.step()
        train_time, train_mae, train_rmse, train_mape, train_con_mae, train_con_rmse, train_con_mape, train_loss, total_loss, Train_conAccuracy, Train_nonconAccuracy = train(net, train_loader, criterion, optimizer, config, epoch_bar, normalize_list)
        epoch_bar.write(
            'train: ' + metrics_to_str(epoch, train_time, scheduler.get_last_lr()[0], train_mae, train_rmse, train_mape, train_con_mae, train_con_rmse, train_con_mape, train_loss, Train_conAccuracy, Train_nonconAccuracy))

        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('rmse/train_rmse', train_rmse, epoch)
        writer.add_scalar('mae/train_mae', train_mae, epoch)
        writer.add_scalar('mape/train_mape', train_mape, epoch)

        epoch_set.append(epoch)
        train_loss_set.append(total_loss)
        train_rmse_set.append(train_rmse)

        if epoch % config.val_every == 0:
            val_time, val_mae, val_rmse, val_mape, val_con_mae, val_con_rmse, val_con_mape, val_loss, pred_iter, label_iter, val_total_loss, Val_conAccuracy, Val_nonconAccuracy = val_test(net, val_loader, criterion, config, epoch_bar, normalize_list, "Validation")

            val_rmse_set.append(val_rmse)
            val_mae_set.append(val_mae)
            val_mape_set.append(val_mape)
            val_crmse_set.append(val_con_rmse)
            val_cmae_set.append(val_con_mae)
            val_cmape_set.append(val_con_mape)
            val_loss_set.append(val_total_loss)
            val_con_accuracy.append(Val_conAccuracy)
            val_noncon_accuracy.append(Val_nonconAccuracy)

            epoch_bar.write(
                'valid: ' + metrics_to_str(epoch, val_time, scheduler.get_last_lr()[0], val_mae, val_rmse,
                                           val_mape, val_con_mae, val_con_rmse, val_con_mape, val_loss, Val_conAccuracy, Val_nonconAccuracy))

            writer.add_scalar('loss/val_loss', val_loss, epoch)
            writer.add_scalar('rmse/val_rmse', val_rmse, epoch)
            writer.add_scalar('mae/val_mae', val_mae, epoch)
            writer.add_scalar('mape/val_mape', val_mape, epoch)

            #Save checkpoint
            if val_loss < best_val_loss:
                stop = 0
                best_val_loss = val_loss
                params_filename = osp.join("save/", config.expt_name + '_' + config.model_name + '_' + str(config.P))
                torch.save(
                    {'epoch': epoch,
                     'model_state_dict': net.state_dict(),
                     'opt_state_dict': optimizer.state_dict(),
                     'loss': criterion
                     }, params_filename)
            else:
                stop = stop + 1
                print("Stop: %d" % stop)

            if stop >= 5:
                print('No improvement after 5 epochs, we stop early!')

        if epoch % config.test_every == 0 or epoch == config.max_epochs - 1 or stop >= 5:
            best_params_filename = osp.join("save/", config.expt_name + '_' + config.model_name + '_' + str(config.P))
            net.load_state_dict(torch.load(best_params_filename)['model_state_dict'])

            test_time, test_mae, test_rmse, test_mape, test_con_mae, test_con_rmse, test_con_mape, test_loss, pred_iter, label_iter, Test_total_loss, Test_conAccuracy, Test_nonconAccuracy= val_test(net, test_loader, criterion, config,
                                                                  epoch_bar, normalize_list, "Test")

            test_rmse_set.append(test_rmse)
            test_mae_set.append(test_mae)
            test_mape_set.append(test_mape)
            test_crmse_set.append(test_con_rmse)
            test_cmae_set.append(test_con_mae)
            test_cmape_set.append(test_con_mape)
            test_con_accuracy.append(Test_conAccuracy)
            test_noncon_accuracy.append(Test_nonconAccuracy)

            epoch_bar.write(
                'test_: ' + metrics_to_str(epoch, test_time, scheduler.get_last_lr()[0], test_mae, test_rmse,
                                           test_mape, test_con_mae, test_con_rmse, test_con_mape, test_loss, Test_conAccuracy, Test_nonconAccuracy))

            writer.add_scalar('loss/test_loss', test_loss, epoch)
            writer.add_scalar('rmse/test_rmse', test_rmse, epoch)
            writer.add_scalar('mae/test_mae', test_mae, epoch)
            writer.add_scalar('mape/test_mape', test_mape, epoch)

        # Reviewing as map
        if epoch == config.max_epochs - 1  or stop >= 5:

            d = {'epoch': epoch_set, 'training_loss': train_loss_set,
                 'training_rmse': train_rmse_set,
                 'validation_loss': val_loss_set,
                 'validation_rmse': val_rmse_set,
                 'validation_mae': val_mae_set,
                 'validation_mape': val_mape_set,
                 'cvalidation_rmse': val_crmse_set,
                 'cvalidation_mae': val_cmae_set,
                 'cvalidation_mape': val_cmape_set,
                 'validation_conAccuracy': val_con_accuracy,
                 'validation_nonconAccuracy': val_noncon_accuracy,
                 'test_rmse': test_rmse_set,
                 'test_mae': test_mae_set,
                 'test_mape': test_mape_set,
                 'ctest_rmse': test_crmse_set,
                 'ctest_mae': test_cmae_set,
                 'ctest_mape': test_cmape_set,
                 'test_conAccuracy': test_con_accuracy,
                 'test_nonconAccuracy': test_noncon_accuracy,
                 }
            df = pd.DataFrame(data=d)
            df.to_csv('results/' + 'trainingInfo_' + config.model_name + '_' + str(config.P) + '.csv', index=False)

            save_result(net, test_no_shuffle_loader, normalize_list, config)
            break

    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gcn_tsp_parser')
    parser.add_argument('-c', '--config', type=str, default="configs/richmond_6.json")
    args = parser.parse_args()
    config = get_config(args.config)
    print("Loaded: {}".format(args.config))
    print("The number of Gpu is: {}".format(torch.cuda.device_count()))
    print("We mainly use {} as main feature, other features are auxiliary".format(config.main_feature))

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)

    net = main(config)
