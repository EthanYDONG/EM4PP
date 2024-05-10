"""
An example of traditional linear Hawkes process model without features
"""
import sys
from hksimulation.Simulation_Branch_HP import Simulation_Branch_HP


import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import dev.util as util
from model.HawkesProcess import HawkesProcessModel,HawkesProcessModel_seq
from preprocess.DataIO import load_sequences_csv
from preprocess.DataOperation import data_info, EventSampler,SequenceSampler, enumerate_all_events

###################simulation hks data#######################
options = {
    'N': 50, 'Nmax': 100, 'Tmax': 50, 'tstep': 0.001,
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

# print(Seqs1)

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
        seq_feature = None

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

        for event in events:
            if event not in database['type2idx']:
                event_index = len(database['type2idx'])
                database['type2idx'][event] = event_index
                database['idx2type'][event_index] = event

    return database






if __name__ == '__main__':
    # hyper-parameters
    memory_size = None
    batch_size = 1
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()
    seed = 2
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    epochs = 200

    # For real world data, load event sequences from csv file
    '''
    domain_names = {'seq_id': 'id',
                    'time': 'time',
                    'event': 'event'}
    database = load_sequences_csv('{}/{}/xxxxxxx.csv'.format(util.EM4PP_PATH, util.DATA_DIR),
                                  domain_names=domain_names)
    '''
    database = convert_to_database(Seqs1)
    data_info(database)
    #Seqdataset = SequenceSampler(database=database, memorysize=memory_size)
    # sample batches from database
    trainloader = DataLoader(SequenceSampler(database=database, memorysize=memory_size),
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)
    validloader = DataLoader(SequenceSampler(database=database, memorysize=memory_size),
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)

    # 
    # initialize model
    num_type = len(database['type2idx'])
    mu_dict = {'model_name': 'NaiveExogenousIntensity_seq',
               'parameter_set': {'activation': 'identity'}
               }
    alpha_dict = {'model_name': 'NaiveEndogenousImpact_seq',
                  'parameter_set': {'activation': 'identity'}
                  }
    # kernel_para = np.ones((2, 1))
    kernel_para = 2*np.random.rand(2, 3)
    kernel_para[1, 0] = 2
    kernel_para = torch.from_numpy(kernel_para)
    kernel_para = kernel_para.type(torch.FloatTensor)
    kernel_dict = {'model_name': 'MultiGaussKernel',
                   'parameter_set': kernel_para}
    loss_type = 'mle'
    hawkes_model = HawkesProcessModel_seq(num_type=num_type,
                                      mu_dict=mu_dict,
                                      alpha_dict=alpha_dict,
                                      kernel_dict=kernel_dict,
                                      activation='identity',
                                      loss_type=loss_type,
                                      use_cuda=use_cuda)

    # initialize optimizer
    optimizer = optim.Adam(hawkes_model.lambda_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # train model
    # hawkes_model.fit(trainloader, optimizer, epochs, scheduler=scheduler,
    #                  sparsity=0.01, nonnegative=0, use_cuda=use_cuda, validation_set=validloader)
    hawkes_model.fit_em(trainloader, optimizer, epochs, scheduler=scheduler,
                     sparsity=None, nonnegative=0, use_cuda=use_cuda, validation_set=None)    
    # save model
    hawkes_model.save_model('{}/{}/full.pt'.format(util.EM4PP_PATH, util.OUTPUT_DIR), mode='entire')
    hawkes_model.save_model('{}/{}/para.pt'.format(util.EM4PP_PATH, util.OUTPUT_DIR), mode='parameter')

    # load model
    hawkes_model.load_model('{}/{}/full.pt'.format(util.EM4PP_PATH , util.OUTPUT_DIR), mode='entire')



    # plot exogenous intensity
    all_events = enumerate_all_events(database, seq_id=1, use_cuda=use_cuda)
    hawkes_model.plot_exogenous(all_events,
                                output_name='{}/{}/exogenous_linearHawkes.png'.format(util.EM4PP_PATH, util.OUTPUT_DIR))

    # plot endogenous Granger causality
    hawkes_model.plot_causality(all_events,
                                output_name='{}/{}/causality_linearHawkes.png'.format(util.EM4PP_PATH, util.OUTPUT_DIR))


'''
    # simulate new data based on trained model
    new_data, counts = hawkes_model.simulate(history=database,
                                             memory_size=memory_size,
                                             time_window=5,
                                             interval=1.0,
                                             max_number=10,
                                             use_cuda=use_cuda)
'''