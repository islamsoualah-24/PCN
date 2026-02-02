import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import time
import csv



if __name__ == '__main__':
    # 获取当前时间戳
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 记录代码运行时间
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='1', help='model id')
    parser.add_argument('--model', type=str, default='PCN',
                        help='model name, options: [PITS, Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='exchange_rate', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/exchange_rate/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file')


    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--c_in', type=int, default=8, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=8, help='input sequence length')
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=36, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # DLinear
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--shared_embedding', type=int, default=1, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=4,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimizationhhh
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPUbiejuanle
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


    # patchTST
    parser.add_argument('--e_layers', type=int, default=2, help='e_layers')
    parser.add_argument('--n_heads', type=int, default=4, help='n_heads')
    parser.add_argument('--d_ff', type=int, default=16, help='d_ff')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # args.data_path = args.data + '.csv'
    # if args.data == 'custom':
    #     args.data_path = 'weather.csv'

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}/{}_{}_P{}_S{}_share{}_ep{}_D{}_eb{}_dt{}_{}'.format(
                args.data,
                args.model_id,
                args.model,
                args.patch_len,
                args.stride,
                args.shared_embedding,
                args.train_epochs,
                args.d_model,
                args.embed,
                args.distil,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))


            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}/{}_{}_P{}_S{}_share{}_ep{}_D{}_eb{}_dt{}_{}'.format(
            args.data,
            args.model_id,
            args.model,
            args.patch_len,
            args.stride,
            args.shared_embedding,
            args.train_epochs,
            args.d_model,
            args.embed,
            args.distil,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 将运行时间写入 CSV 文件
    with open("costtime.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, setting, elapsed_time])  # 写入时间戳、代码名称和运行时间

    print(f"代码运行时间：{elapsed_time} 秒")
