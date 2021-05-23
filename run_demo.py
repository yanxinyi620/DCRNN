import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))


parser = argparse.ArgumentParser()
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.use_cpu_only = True
args.config_filename = 'data/model/pretrained/METR-LA/config.yaml'
args.output_filename = 'data/dcrnn_predictions.npz'

run_dcrnn(args)


'''
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
'''


''' results:
2021-05-23 14:05:45,717 - INFO - Horizon 01, MAE: 2.17, MAPE: 0.0517, RMSE: 3.76
2021-05-23 14:05:45,806 - INFO - Horizon 02, MAE: 2.47, MAPE: 0.0612, RMSE: 4.59
2021-05-23 14:05:45,893 - INFO - Horizon 03, MAE: 2.66, MAPE: 0.0683, RMSE: 5.16
2021-05-23 14:05:45,978 - INFO - Horizon 04, MAE: 2.82, MAPE: 0.0742, RMSE: 5.61
2021-05-23 14:05:46,068 - INFO - Horizon 05, MAE: 2.95, MAPE: 0.0792, RMSE: 5.98
2021-05-23 14:05:46,160 - INFO - Horizon 06, MAE: 3.06, MAPE: 0.0836, RMSE: 6.28
2021-05-23 14:05:46,253 - INFO - Horizon 07, MAE: 3.16, MAPE: 0.0875, RMSE: 6.54
2021-05-23 14:05:46,345 - INFO - Horizon 08, MAE: 3.25, MAPE: 0.0910, RMSE: 6.77
2021-05-23 14:05:46,444 - INFO - Horizon 09, MAE: 3.33, MAPE: 0.0941, RMSE: 6.97
2021-05-23 14:05:46,536 - INFO - Horizon 10, MAE: 3.40, MAPE: 0.0970, RMSE: 7.15
2021-05-23 14:05:46,629 - INFO - Horizon 11, MAE: 3.47, MAPE: 0.0996, RMSE: 7.31
2021-05-23 14:05:46,719 - INFO - Horizon 12, MAE: 3.54, MAPE: 0.1024, RMSE: 7.48
Predictions saved as data/dcrnn_predictions.npz.
'''
