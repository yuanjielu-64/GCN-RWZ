import numpy as np
import os.path as osp
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
from configs.config import get_config
from sklearn import preprocessing
from model.GCNRWZ_NET import GCNRWZ_model

class mydataset(torch.utils.data.Dataset):
    def __init__(self, input, output,input_time, output_time, device):
        self.device = device
        self.input = input.tolist()
        self.output = output.tolist()
        self.input_time = input_time.tolist()
        self.output_time = output_time.tolist()

    def __getitem__(self, idx):
        return torch.tensor(self.input[idx]).to(self.device), torch.tensor(self.output[idx]).to(self.device)

    def __len__(self):
        return len(self.input)


def slide_windows_generate():
    save = 'richmond_12_6.npy'
    path = osp.join('./data/processed',save)
    data = np.load(path, allow_pickle=True)

    return data

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

parser = argparse.ArgumentParser(description='gcn_tsp_parser')
parser.add_argument('-c', '--config', type=str, default="configs/richmond.json")
args = parser.parse_args()
config = get_config(args.config)

df = pd.read_csv(config.speed_filepath, index_col=0)
df_con = pd.read_csv(config.construction_filepath, index_col=0)
df_vol = pd.read_csv(config.volume_filepath, index_col=0)
df_diff_speed = pd.read_csv(config.diff_filepath, index_col=0)  # speed - History speed
dis_matrix = pd.read_csv(config.dis_matrix, index_col=0)
A_matrix = pd.read_csv(config.A_matrix, index_col=0)
df_history = df - df_diff_speed
df_mask = np.not_equal(df, 0.0)
df_mask = df_mask.astype(np.float32)

df_nor = preprocessing.StandardScaler().fit(df)
df_vol_nor = preprocessing.StandardScaler().fit(df_vol)
df_history_nor = preprocessing.StandardScaler().fit(df_history)

# normalize speed, volume, diff_speed
normalize_list = {"Speed": df_nor,
                  "Volume": df_vol_nor,
                  "History_speed": df_history_nor,
                  }

data = slide_windows_generate()
test_no_shuffle = mydataset(data[3][0], data[3][1], data[3][2], data[3][3], "cuda")

test_no_shuffle_loader = DataLoader(
    dataset=test_no_shuffle,
    batch_size= 16,
)

net = GCNRWZ_model(config, A_matrix, dis_matrix).to(config.device)
best_params_filename = osp.join("save/", config.expt_name + '_' + config.model_name + '_' + str(config.P))
net.load_state_dict(torch.load(best_params_filename)['model_state_dict'])
save_result(net, test_no_shuffle_loader, normalize_list, config)

print("")