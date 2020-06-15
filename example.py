from Kitsune import Kitsune
import numpy as np
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import argparse


def read_ds_info(fpath):
    with open(fpath, 'r') as fin:
        data = fin.readlines()

    info_dict = {}
    for row in data:
        row = row.strip()
        tag, start, end = row.split(',')
        start, end = int(start), int(end)
        if tag == 'TRAIN_SET':
            info_dict['TRAIN_SET'] = (start, end)
        else:
            tag, label = '_'.join(tag.split('_')[:-1]), tag.split('_')[-1]
            if tag not in info_dict:
                info_dict[tag] = {}
            info_dict[tag][label] = (start, end)
    return info_dict


parser = argparse.ArgumentParser(description='Run Kitsune.')
parser.add_argument('--traffic-trace', type=str, help='pcap/tsv file to read.')
parser.add_argument('--n-train-FM', type=int)
args = parser.parse_args()

info_dict = read_ds_info(args.traffic_trace + '.info')

# File location
path = args.traffic_trace  # the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf  # the number of packets to process

# KitNET params:
maxAE = 10  # maximum size for any autoencoder in the ensemble layer
# the number of instances taken to learn the feature mapping (the ensemble's architecture)
FMgrace = args.n_train_FM
# the number of instances used to train the anomaly detector (ensemble itself)
ADgrace = info_dict['TRAIN_SET'][1] - info_dict['TRAIN_SET'][0] + 1 - FMgrace
del info_dict['TRAIN_SET']

# Build Kitsune
K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i += 1
    if i % 5000 == 0:
        print(i)
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: " + str(stop - start))

for tag, label in info_dict.items():
    benign_start, benign_end = label['benign']
    attack_start, attack_end = label['attack']
    benign_rmses = RMSEs[benign_start:benign_end]
    attack_rmses = RMSEs[attack_start:attack_end]
    if len(benign_rmses) == 0 or len(attack_rmses) == 0:
        print("AUC ROC score for [%s]: ERROR_ZERO_COUNT" % tag)
        continue
    rmses = np.array(benign_rmses + attack_rmses)
    y = np.array([0] * len(benign_rmses) + [1] * len(attack_rmses))
    fpr, tpr, thresholds = metrics.roc_curve(y, rmses, pos_label=1)
    tpr_lst = {}
    for fpr_th in [0.01, 0.05]:
        for i in range(1, len(fpr)):
            curr_fpr = fpr[i]
            prev_fpr = fpr[i-1]
            if prev_fpr <= fpr_th and fpr_th <= curr_fpr:
                tpr_lst[fpr_th] = tpr[i-1]
    eer_score = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    score = metrics.roc_auc_score(y, rmses)
    print(','.join([tag, str(score), str(
        tpr_lst[0.01]), str(tpr_lst[0.05]), str(eer_score)]))
