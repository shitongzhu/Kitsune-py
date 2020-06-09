from Kitsune import Kitsune
import numpy as np
import time
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser(description='Run Kitsune.')
parser.add_argument('--traffic-trace', type=str, help='pcap/tsv file to read.')
parser.add_argument('--n-train-FM', type=int)
parser.add_argument('--n-train-AE', type=int)
parser.add_argument('--attack-start', type=int, default=100000000)
parser.add_argument('--cutoff', type=int, default=100000000)
args = parser.parse_args()

##############################################################################
# Kitsune a lightweight online network intrusion detection system based on an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 3.6.3   #######################

# File location
path = args.traffic_trace #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = args.n_train_FM #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = args.n_train_AE #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i+=1
    if i % 5000 == 0:
        print(i)
    rmse = K.proc_next_packet()
    if rmse == -1 or i > args.cutoff:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

train_cutoff = FMgrace + ADgrace + 1
benign_rmses = RMSEs[train_cutoff:args.attack_start]
attack_rmses = RMSEs[args.attack_start+1:]

rmses = np.array(benign_rmses + attack_rmses)
y = np.array([0] * len(benign_rmses) + [1] * len(attack_rmses))
score = metrics.roc_auc_score(y, rmses)
print("AUC ROC score: %f" % score)
