from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import numpy as np
import os.path as osp
import torch
from tqdm import tqdm
from sklearn import preprocessing
import random

class mydataset(torch.utils.data.Dataset):
    def __init__(self, input, output, input_time, output_time, device):
        self.device = device
        self.input = input.tolist()
        self.output = output.tolist()
        self.input_time = input_time.tolist()
        self.output_time = output_time.tolist()

    def __getitem__(self, idx):
        return torch.tensor(self.input[idx]).to(self.device), torch.tensor(self.output[idx]).to(self.device)

    def __len__(self):
        return len(self.input)

def Radial_Basis_Function(x, config):
    np.seterr(divide='ignore', invalid='ignore')
    x = np.array(x)
    std = np.mean(x[x!=0])
    x[x > std] = 0
    x[x == 0] = float("inf")
    diag = np.diag_indices_from(x)
    x[diag[0], diag[1]] = 0
    dis = (x * x / (std * std))
    if config.expt_name == "richmond":
        x = np.exp(-1 * (x * x / (10 * 10)))
        x = np.where(x > 0.5, x, 0)
    else:
        x = np.exp(-1 * (x * x / (3 * 3)))
        x = np.where(x > 0.5, x, 0)
    x[diag[0],diag[1]] = 1
    return dis, x

def writer(config):
    # Create log directory
    log_dir = f"./logs/{config.expt_name}/"
    os.makedirs(log_dir, exist_ok=True)
    json.dump(config, open(f"{log_dir}/" + config.expt_name + '_' + config.model_name + '_' + str(config.P) + ".json", "w"), indent=4)
    writer = SummaryWriter(log_dir)  # Define Tensorboard writer
    return writer

def load_data(config):
    print("Loading the input file(s)")

    df = pd.read_csv(config.speed_filepath, index_col=0)
    df_timeofday = pd.read_csv(config.timeofday, index_col= 0)
    df_con = pd.read_csv(config.construction_filepath, index_col = 0)
    if config.expt_name == "tyson":
        # I don't have volumn data in tyson's dataset, this is just to keep the same dimension as richmond, please don't use volumn in tyson
        df_vol = pd.read_csv(config.construction_filepath, index_col=0)
    else:
        df_vol = pd.read_csv(config.volume_filepath, index_col=0)
    df_diff_speed = pd.read_csv(config.diff_filepath, index_col = 0)  # speed - History speed
    dis_matrix = pd.read_csv(config.dis_matrix, index_col = 0)
    A_matrix = pd.read_csv(config.A_matrix, index_col = 0)
    df_time = pd.read_csv(config.time_filepath, index_col= 0)
    df_mask = np.not_equal(df, 0.0)
    df_mask = df_mask.astype(np.float32)
    # Generate the history speed to avoid missing value
    df_history = df - df_diff_speed
    df_history = fix_zero(df_history)
    # Clean the missing value
    df = df + (1 - df_mask) * df_history

    # if the measured speed is greater than 60mph,
    # even if it is less than the historical speed, we consider it to be unaffected by the anomaly events
    if config.expt_name == "richmond":
        diff = np.minimum(np.array(df) - 60, 0)
        diff = np.where(diff < 0, 1, diff)
        df_diff_speed[df_diff_speed >= - 5] = 0
        df_diff_speed[df_diff_speed < 0] = 1
        df_diff_speed = df_diff_speed * diff
    else:
        df_diff_speed[df_diff_speed >= - 5] = 0
        df_diff_speed[df_diff_speed < 0] = 1

    # Scale the data
    df_nor = preprocessing.StandardScaler().fit(df)
    df_vol_nor = preprocessing.StandardScaler().fit(df_vol)
    df_history_nor = preprocessing.StandardScaler().fit(df_history)

    # normalize speed, volume, diff_speed
    normalize_list ={"Speed": df_nor,
                     "Volume":df_vol_nor,
                     "History_speed": df_history_nor,
    }

    time_ = df.index.values
    dis, D = Radial_Basis_Function(np.array(dis_matrix), config)

    # Generate the workzone information
    wz = wz_set(np.array(df_con), D)

    # speed, construction, volume, diff_speed
    dataframe = np.stack([df_nor.transform(df),
                          np.array(wz),
                          df_vol_nor.transform(df_vol),
                          df_history_nor.transform(df_history),
                          np.array(df_diff_speed),
                          np.array(df_time, dtype=int),
                          np.array(df_timeofday),
                          np.array(df_mask)])

    # including data shuffle
    data = slide_windows_generate(dataframe, time_, config)
    # already shuffle from mydataset

    train_shuffle = mydataset(data[0][0], data[0][1], data[0][2],data[0][3], config.device)
    val_shuffle = mydataset(data[1][0], data[1][1], data[1][2], data[1][3], config.device)
    test_shuffle = mydataset(data[2][0], data[2][1], data[2][2], data[2][3], config.device)
    # This is only for reviewing a map in the end of experiment
    test_no_shuffle = mydataset(data[3][0], data[3][1], data[3][2], data[3][3], config.device)

    train_loader = DataLoader(
        dataset = train_shuffle,
        batch_size = config.batch_size,
    )

    val_loader = DataLoader(
        dataset=val_shuffle,
        batch_size=config.batch_size,
    )

    test_loader = DataLoader(
        dataset=test_shuffle,
        batch_size=config.batch_size,
    )

    test_no_shuffle_loader = DataLoader(
        dataset = test_no_shuffle,
        batch_size= config.batch_size,
    )

    return train_loader, val_loader, test_loader, test_no_shuffle_loader, dis, D, normalize_list

def wz_set(x, adj):
    adj[adj > 0] = 1
    wz_set = np.zeros(shape=x.shape)

    for i in range(len(x)):
        a = (x[i] >= 1).nonzero()[0]
        for j in range(len(a)):
            wz_set[i] = np.maximum(wz_set[i], adj[a[j]])

    return wz_set

def wz_dis(x, adj, device):
    adj[adj > 0] = 1
    batchsize, channel, lens, num_of_vertices = x.shape
    x = torch.reshape(x, (batchsize * channel * lens, num_of_vertices))
    wz_set = torch.zeros(size = (batchsize * channel * lens, num_of_vertices), dtype=torch.float32).to(device)
    for i in range(lens):
        a = (x[i] >= 1).nonzero()
        for j in range(len(a)):
            e = adj[a[j]]
            wz_set[i] = torch.max(wz_set[i], e)
    return torch.reshape(wz_set,(batchsize, channel, lens, num_of_vertices))

def slide_windows_generate(dataset, time, config):
    save = config.expt_name + '_' + str(config.T) + '_' + str(config.P) + '.npy'
    path = osp.join('./data/processed',save)

    if osp.exists(path) is not True:
        dataset = dataset.transpose(1, 2, 0)
        num_input, num_node, dims = dataset.shape
        num_sample = num_input - config.T - config.P + 1

        index = pd.DataFrame(time, columns=['time'])
        index.time = pd.to_datetime(index.time)

        n = 0
        x_set = []
        y_set = []
        x_time_set = []
        y_time_set = []
        print("Creating the file")
        # num_sample
        for i in tqdm(range(num_sample)):
            flag = 0
            x = np.zeros(shape=(config.T, num_node, dims))
            y = np.zeros(shape=(config.P, num_node, dims))
            x_time = np.empty(shape=(config.T), dtype=object)
            y_time = np.empty(shape=(config.P), dtype=object)
            x[0] = dataset[i, :, :]
            x_time[0] = index['time'].iloc[i]
            for j in range(1, config.T + config.P):
                if j <= (config.T - 1):
                    if (index['time'].iloc[i + j] - index['time'].iloc[i + j - 1]).seconds == config.time_interval:
                        x[j] = dataset[i + j, :, :]
                        x_time[j] = index['time'].iloc[i + j]
                    else:
                        flag = 1
                        break
                else:
                    if (index['time'].iloc[i + j] - index['time'].iloc[i + j - 1]).seconds == config.time_interval:
                        y[j - config.T] = dataset[i + j, :, :]
                        y_time[j - config.T] = index['time'].iloc[i + j]
                    else:
                        flag = 1
                        break

            if flag == 0:
                n = n + 1
                x = x.astype(np.float32)
                y = y.astype(np.float32)
                x_set.append(x)
                y_set.append(y)
                x_time_set.append(x_time)
                y_time_set.append(y_time)

        data = pd.DataFrame([x_set, y_set, x_time_set, y_time_set]).T
        random.seed(1)
        test_data_sample = random.sample(list(data.T), config.observation_number)

        test_no_shuffle = list()
        for i in test_data_sample:
            sample_data = data[i: i + config.observation_length]
            data.drop(data[i: i + config.observation_length].index, inplace= True)
            test_no_shuffle.append(sample_data)

        if len(test_no_shuffle) != 0:
            test_no_shuffle = pd.concat(test_no_shuffle, axis = 0)
        else:
            test_no_shuffle = pd.DataFrame()
        data = data.sample(frac=1, random_state = 1234)

        train = data[:round((len(data) * 0.7 // config.batch_size) * config.batch_size)]
        val = data[round((len(data) * 0.7 // config.batch_size) * config.batch_size): round((len(data) * 0.8 // config.batch_size) * config.batch_size)]
        test = data[round((len(data) * 0.8 // config.batch_size) * config.batch_size): round((len(data) // config.batch_size) * config.batch_size)]

        np.save(path, np.array([train, val, test, test_no_shuffle], dtype= object))
        data = np.load(path, allow_pickle=True)
    else:
        data = np.load(path, allow_pickle=True)

    return data

def A_wave_function(args, A):
    with np.errstate(divide='ignore', invalid='ignore'):
        assert A.shape[0] == A.shape[1]

        D = np.diag(np.sum(A, axis=1))
        In = np.identity(A.shape[0])
        D_wave=  D + In
        A_wave = A + In
        D_wave_sqrt = D_wave ** -0.5
        D_wave_sqrt[D_wave_sqrt > 1] = 0
        D_wave_sqrt = np.mat(D_wave_sqrt)
        A_wave = np.mat(A_wave)
        A_hat = D_wave_sqrt * A_wave * D_wave_sqrt
        return torch.from_numpy(A_hat).type(torch.FloatTensor).to(args.device)

def metric(config, pred, label, con, normalize_list):
    batchsize, T, N = pred.shape[0] , pred.shape[1], pred.shape[2]
    normalize_feature = normalize_list[config.main_feature]
    pred = np.resize(pred, (pred.shape[0] * pred.shape[1], pred.shape[2]))
    label = np.resize(label, (label.shape[0] * label.shape[1], label.shape[2]))
    pred = np.around(normalize_feature.inverse_transform(pred), 2)
    label = np.around(normalize_feature.inverse_transform(label), 2)
    con = np.resize(con, (con.shape[0] * con.shape[1], con.shape[2]))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        if config.mask == "True":
            con[con >= 1] = 1
            noncon = 1 - con
            exist = (con != 0)
            exist0 = (con == 0)

            label[label <= 0] = 0.0
            mask = np.not_equal(label, 0.0)
            mask = mask.astype(np.float32)
            pred = pred * mask

            mask /= np.mean(mask)
            #mae
            mae = np.abs(np.subtract(pred, label).astype('float64'))
            mae = np.nan_to_num(mask * mae)
            con_mae = (mae * con).sum() / exist.sum()
            mae = (mae * noncon).sum() / exist0.sum()

            #rmse
            mse = ((pred- label)**2)
            mse = np.nan_to_num(mask * mse)

            con_rmse = np.sqrt((mse * con).sum() / exist.sum())
            rmse = np.sqrt((mse * noncon).sum() / exist0.sum())
            heatmap = pd.DataFrame(np.sqrt(np.mean(np.resize(mse, (batchsize, T, N)), axis= 0)))
            heatmap.to_csv("heatmap.csv")

            #mape
            mape = np.abs(np.divide(np.subtract(pred, label).astype('float64'),label))
            mape = np.nan_to_num(mask * mape)

            con_mape = (mape * con).sum() / exist.sum()
            mape = (mape * noncon).sum() / exist0.sum()

            # calculate the accuracy between pred and label
            setLabel = label.copy()
            setLabel[setLabel > 60] = 0
            setLabel[setLabel > 0] = 1
            con = setLabel * con
            noncon = setLabel * noncon

            diff = abs(label - pred) * mask
            condiff = (diff * con)
            condiff[condiff >= 5] = 0
            conAccuracy = (condiff > 0).sum() / (con == 1).sum() * 100

            noncondiff = (diff * noncon)
            noncondiff[noncondiff >= 5] = 0
            nonconAccuracy = (noncondiff > 0).sum() / (noncon == 1).sum() * 100

            return mae, rmse, mape, con_mae, con_rmse, con_mape, conAccuracy, nonconAccuracy
        else:
            mae = np.abs(np.subtract(pred, label)).astype(np.float64)
            rmse = np.square(mae)
            mape = np.divide(mae, label)
            mae = np.nan_to_num(mae)
            mae = np.mean(mae)
            rmse = np.nan_to_num(rmse)
            rmse = np.sqrt(np.mean(rmse))
            mape = np.nan_to_num(mape)
            mape_ = np.mean(mape, dtype=np.float64)
            return mae, rmse, mape_

def fix_zero(df):
    for i in df.columns:
        mean = df[i][df[i]!=0].mean()
        df[i] = df[i].replace(0, mean)

    return df
