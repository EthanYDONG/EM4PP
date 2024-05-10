import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.sparse import csr_matrix
from hksimulation.Simulation_Branch_HP import Simulation_Branch_HP


np.random.seed(0)


# N 是每个cluster有多少条seq
# Tmax 是最大的时间
###################simulation hks data#######################
options = {
    'N': 100, 'Nmax': 100, 'Tmax': 50, 'tstep': 0.001,
    'dt': [0.001], 'M': 0, 'GenerationNum': 10
}
D = 20  # event type个数 也就是维度
K = 2  # cluster个数
nTest = 5
nSeg = 5
nNum = options['N'] / nSeg
mucenter = np.random.rand(D) / D
mudelta = 0.1   # 两个cluster的base intensity的差距

# First01 cluster: Hawkes process with exponential kernel

print('01 Simple exponential kernel')
para1 = {'kernel': 'exp', 'landmark': [0]}
para1['mu'] = mucenter
L = len(para1['landmark'])
para1['A'] = np.zeros((D, D, L))
for l in range(1, L + 1):
    para1['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
eigvals_list = []
eigvecs_list = []
for l in range(L):
    eigvals, eigvecs = np.linalg.eigh(para1['A'][:, :, l])
    eigvals_list.append(eigvals)
    eigvecs_list.append(eigvecs)
all_eigvals = np.concatenate(eigvals_list)
max_eigval = np.max(all_eigvals)
para1['A'] = 0.05 * para1['A'] / max_eigval
para1['w'] = 0.5
Seqs1 = Simulation_Branch_HP(para1, options)
for sequence in Seqs1:
    sequence['seq_type'] = 0

print(Seqs1)


def convert_to_database(Seqs):
    database = {
        'event_features': None,
        'type2idx': {},
        'idx2type': {},
        'seq2idx': {},
        'idx2seq': {},
        'sequences': []
    }

    for idx, seq in enumerate(Seqs):
        times = seq['Time']
        events = seq['Mark']
        start = seq['Start']
        stop = seq['Stop']
        seq_type = seq['seq_type']
        seq_feature = seq['Feature']

        database['sequences'].append({
            'times': times,
            'events': events,
            'seq_feature': seq_feature,
            't_start': start,
            't_stop': stop,
            'label': seq_type
        })

        seq_name = str(idx)
        database['seq2idx'][seq_name] = idx
        database['idx2seq'][idx] = seq_name

        # 将事件类型映射为整数，并建立索引
        for event in events:
            if event not in database['type2idx']:
                event_index = len(database['type2idx'])
                database['type2idx'][event] = event_index
                database['idx2type'][event_index] = event

    return database


# 转换为database格式
database = convert_to_database(Seqs1)
print(database)



# First02 cluster: Hawkes process with exponential kernel
'''
print('02 Simple exponential kernel')
para2 = {'kernel': 'exp', 'landmark': [0]}
para2['mu'] = mucenter+mudelta  
L = len(para2['landmark'])
# para2['A'] = np.zeros((D, D, L))
# for l in range(1, L + 1):
#     para2['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
# eigvals_list = []
# eigvecs_list = []
# for l in range(L):
#     eigvals, eigvecs = np.linalg.eigh(para2['A'][:, :, l])
#     eigvals_list.append(eigvals)
#     eigvecs_list.append(eigvecs)
# all_eigvals = np.concatenate(eigvals_list)
# max_eigval = np.max(all_eigvals)
#para2['A'] = 0.5 * para2['A'] / max_eigval
para2['A'] = para1['A']
para2['w'] = para1['w']
Seqs2 = Simulation_Branch_HP(para2, options)
for sequence in Seqs2:
    sequence['seq_type'] = 1

SeqsMix = Seqs1 + Seqs2

# print(type(SeqsMix))
# print("Total sequences in the list:", len(SeqsMix))
# if len(SeqsMix) > 0:
#     first_seq = SeqsMix[0]
#     print("Type of the first sequence:", type(first_seq))
#     print("Contents of the first sequence:", first_seq)


# # 再保存成需要的格式即可
# # with open('/home/wangqingmei/kdd24TPPre/hkstools/data_hks2/SeqsMix.pkl', 'wb') as f:
# #      pickle.dump(SeqsMix, f)
     
# # 加载SeqsMix数据
# with open('/home/wangqingmei/kdd24TPPre/hkstools/data_hks2/SeqsMix.pkl', 'rb') as f:
#     SeqsMix = pickle.load(f)

# 确定数据集大小
total_size = len(SeqsMix)
train_size = int(0.8 * total_size)
dev_size = int(0.1 * total_size)
test_size = total_size - train_size - dev_size  # 剩余的10%

# 确保数据集是随机的
np.random.shuffle(SeqsMix)

# 划分数据集
train_data = SeqsMix[:train_size]
dev_data = SeqsMix[train_size:train_size+dev_size]
test_data = SeqsMix[train_size+dev_size:]

filenametrain = f'SeqsN{options["N"]}_Dim{D}_K{K}_mudelta{mudelta}_train.pkl'

filenamedev = f'SeqsN{options["N"]}_Dim{D}_K{K}_mudelta{mudelta}_dev.pkl'

filenametest = f'SeqsN{options["N"]}_Dim{D}_K{K}_mudelta{mudelta}_test.pkl'


# 将划分后的数据集保存为pickle文件
with open(f'/home/wangqingmei/kdd24TPPre/hkstools/hksdata0418delta01/{filenametrain}', 'wb') as f:
    pickle.dump(train_data, f)

with open(f'/home/wangqingmei/kdd24TPPre/hkstools/hksdata0418delta01/{filenamedev}', 'wb') as f:
    pickle.dump(dev_data, f)

with open(f'/home/wangqingmei/kdd24TPPre/hkstools/hksdata0418delta01/{filenametest}', 'wb') as f:
    pickle.dump(test_data, f)
'''