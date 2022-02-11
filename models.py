# -*- coding: utf-8 _*_
# @Time : 20/1/2022 5:03 pm
# @Author: ZHA Mengyue
# @FileName: models.py
# @Software: MAEI
# @Blog: https://github.com/Dolores2333

import torch.nn as nn
import torch
from tqdm import tqdm
from einops import rearrange
from modules.utils import *
from modules.generation import *
#from modules.visualization import *
#from metrics.timegan_metrics import calculate_pred_disc


def mask_it(x, masks):
    # x(bs, ts_size, z_dim)
    b, l, f = x.shape
    #print("Shape of x is:", x.shape)
    #print("Shape of masks is:", masks.shape)
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, vis_size, z_dim)
    return x_visible


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.z_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.hidden_dim)

    def forward(self, x):
        x_enc, _ = self.rnn(x)
        x_enc = self.fc(x_enc)
        return x_enc


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.hidden_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.z_dim)

    def forward(self, x_enc):
        x_dec, _ = self.rnn(x_enc)
        x_dec = self.fc(x_dec)

        return x_dec


class Interpolator(nn.Module):
    def __init__(self, args):
        super(Interpolator, self).__init__()
        self.sequence_inter = nn.Linear(in_features=(args.ts_size - args.total_mask_size),
                                        out_features=args.ts_size)
        self.feature_inter = nn.Linear(in_features=args.hidden_dim,
                                       out_features=args.hidden_dim)

    def forward(self, x):
        # x(bs, vis_size, hidden_dim)
        x = rearrange(x, 'b l f -> b f l')  # x(bs, hidden_dim, vis_size)
        x = self.sequence_inter(x)  # x(bs, hidden_dim, ts_size)
        x = rearrange(x, 'b f l -> b l f')  # x(bs, ts_size, hidden_dim)
        x = self.feature_inter(x)  # x(bs, ts_size, hidden_dim)
        return x


class InterpoMAEUnit(nn.Module):
    def __init__(self, args):
        super(InterpoMAEUnit, self).__init__()
        self.args = args
        self.ts_size = args.ts_size
        self.mask_size = args.mask_size
        self.num_masks = args.num_masks
        self.total_mask_size = args.num_masks * args.mask_size
        args.total_mask_size = self.total_mask_size
        self.z_dim = args.z_dim
        self.encoder = Encoder(args)
        self.interpolator = Interpolator(args)
        self.decoder = Decoder(args)

    def forward(self, x, masks):
        """No mask tokens, using Interpolation in the latent space"""
        x_vis = mask_it(x, masks)  # (bs, vis_size, z_dim)
        x_enc = self.encoder(x_vis)  # (bs, vis_size, hidden_dim)
        x_inter = self.interpolator(x_enc)  # (bs, ts_size, hidden_dim)
        x_dec = self.decoder(x_inter)  # (bs, ts_size, z_dim)
        return x_inter, x_dec, masks


class InterpoMAE(nn.Module):
    def __init__(self, args, ori_data):
        super(InterpoMAE, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.model = InterpoMAEUnit(args).to(self.device)
        self.ori_data = ori_data
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.num_iteration = 0
        print(f'Successfully initialized {self.__class__.__name__}!')

    def train_recon(self):
        for t in tqdm(range(self.args.mae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            self.model.train()
            _, x_dec, masks = self.model(x_ori, random_masks)
            loss = self.criterion(x_dec, x_ori)

            self.num_iteration += 1

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def synthesize_cross_concate(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = cross_concat_generation(self.args, self.model, ori_data)

        # Save Re-normalized art_data
        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        print('Synthetic Data Generation by Cross Concate Finished.')
        return art_data
        

def run_mae(args, data):
    z = data.shape[-1]
    ori_data, min_val, max_val = min_max_scalar(data)
    # Write statistics
    args.min_val = min_val
    args.max_val = max_val
    #args.data_var = np.var(ori_data)
    #print(f'{args.data_name} data variance is {args.data_var}')
    slide_data = sliding_window(args, ori_data)

    # Initialize the Model
    model = InterpoMAE(args, slide_data)
    if args.training:
        print(f'Start Reconstruction Training! {args.mae_epochs} Epochs Needed.')
        model.train_recon()
    else:
        model = load_model(args, model)
        print(f'Successfully loaded the model!')

    print('Synthesizing')
    art_data = model.synthesize_cross_concate()
    art_data = art_data.reshape(-1, z)
    all_data = np.concatenate((art_data, ori_data), axis=0)
    return all_data

def pseudo_aug(args, data):
    return data

