import bittensor
import torch
import numpy as np
import pandas as pd
from loguru import logger
from termcolor import colored
from typing import List
from bittensor import Session

np.set_printoptions(precision=2, suppress=True, linewidth=500, sign=' ')
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def log_all( session: Session, history: List[bittensor.synapse.SynapseOutput] ):
    log_outputs(history)
    log_batch_weights(session, history)
    log_row_weights(session)
    log_col_weights
    log_incentive(session)
    log_ranks(session)
    log_request_sizes(session, history)
    log_return_codes(session, history)
    log_dendrite_success_times( session )


def _calculate_request_sizes_sum(session: Session, history: List[bittensor.synapse.SynapseOutput]):
    request_size_list = []
    request_sizes_sum = torch.zeros(session.metagraph.n)
    for output in history:
        request_sizes_sum += output.request_sizes
        request_size_list.append(output.request_sizes)
    request_sizes_sum = request_sizes_sum.tolist()

    return request_sizes_sum, request_size_list

def log_dendrite_success_times(session: Session):
    print ('Avg Success time: \n ')
    remotes = session.dendrite.remotes
    neurons = [remote.neuron for remote in remotes]
    uids = session.metagraph.neurons_to_uids(neurons)
    succees_time = [remote.stats.avg_success_time for remote in remotes]
    df = pd.DataFrame([succees_time], columns=uids.tolist())
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    df.rename_axis("[uid]", axis=1)
    print (df)
    print('\n')

def log_ranks(session: Session):
    print ('Ranks: \n ')
    if session.metagraph.uids != None and session.metagraph.weights != None:
        uids = session.metagraph.uids.tolist()
        ranks = session.metagraph.ranks.tolist()
        ranks, uids  = zip(*sorted(zip(ranks, uids), reverse=True))
        df = pd.DataFrame([ranks], columns=uids)
        df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
        max_val = df.max(numeric_only=True, axis=1)
        min_val = df.min(numeric_only=True, axis=1)
        total = df.sum(numeric_only=True, axis=1)
        df.loc[:,'Min'] = min_val
        df.loc[:,'Max'] = max_val
        df.loc[:,'Total'] = total
        print (df)
    print('\n')


def log_incentive(session: Session):
    print ('Incentive: \n ')
    if session.metagraph.uids != None and session.metagraph.weights != None:
        uids = session.metagraph.uids.tolist()
        incentive = session.metagraph.incentive.tolist()
        incentive, uids  = zip(*sorted(zip(incentive, uids), reverse=True))
        df = pd.DataFrame([incentive], columns=uids)
        df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
        max_val = df.max(numeric_only=True, axis=1)
        min_val = df.min(numeric_only=True, axis=1)
        total = df.sum(numeric_only=True, axis=1)
        df.loc[:,'Min'] = min_val
        df.loc[:,'Max'] = max_val
        df.loc[:,'Total'] = total
        print (df)
    print('\n')

def log_return_codes(session: Session, history: List[bittensor.synapse.SynapseOutput]):
    print('Return Codes: \n ')
    request_size_list = []
    request_sizes_sum = torch.zeros(session.metagraph.n)
    for output in history:
        request_sizes_sum += output.request_sizes
        request_size_list.append(output.request_sizes)
    request_sizes_sum = request_sizes_sum.tolist()

    return request_sizes_sum, request_size_list

def log_return_codes(session: Session, history: List[bittensor.synapse.SynapseOutput]):
    print('Return Codes: \n ')
    request_sizes_sum, _ = _calculate_request_sizes_sum(session, history)

    rows = []
    for output in history:
        _, retcodes  = zip(*sorted(zip(request_sizes_sum, output.return_codes.tolist()), reverse=True))
        rows.append(retcodes)
    _, uids  = zip(*sorted(zip(request_sizes_sum, session.metagraph.uids.tolist()), reverse=True))
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df = pd.DataFrame(rows, columns=uids)
    df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
    print (df)
    print('\n')

def log_request_sizes(session: Session, history: List[bittensor.synapse.SynapseOutput]):
    print('Request Sizes: \n ')
    request_sizes_sum, request_size_list = _calculate_request_sizes_sum(session, history)

    rows = []
    for rs in request_size_list:
        _, rs  = zip(*sorted(zip(request_sizes_sum, rs.tolist()), reverse=True))
        rows.append(rs)
    _, uids  = zip(*sorted(zip(request_sizes_sum, session.metagraph.uids.tolist()), reverse=True))
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df = pd.DataFrame(rows, columns=uids)
    total_row = df.sum(numeric_only=True, axis=1)
    total_col = df.sum(numeric_only=True, axis=0)
    min_row = df.min(numeric_only=True, axis=1)
    min_col = df.min(numeric_only=True, axis=0)
    max_row = df.max(numeric_only=True, axis=1)
    max_col = df.max(numeric_only=True, axis=0)
    mean_row = df.mean(numeric_only=True, axis=1)
    mean_col = df.mean(numeric_only=True, axis=0)
    df.loc[:,'Min'] = min_row
    df.loc[:,'Max'] = max_row
    df.loc[:,'Mean'] = mean_row
    df.loc[:,'Total'] = total_row
    df.loc['Min'] = min_col
    df.loc['Max'] = max_col
    df.loc['Mean'] = mean_col
    df.loc['Total'] = total_col
    df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
    print (df)
    print('\n')

def log_row_weights(session: Session):
    print ('Row Weights: \n ')
    if session.metagraph.uids != None and session.metagraph.weights != None:
        uids = session.metagraph.uids.tolist()
        weights = session.metagraph.W[0,:].tolist()
        weights, uids  = zip(*sorted(zip(weights, uids), reverse=True))
        df = pd.DataFrame([weights], columns=uids)
        df.rename_axis("[uid]", axis=1)
        max_val = df.max(numeric_only=True, axis=1)
        min_val = df.min(numeric_only=True, axis=1)
        total = df.sum(numeric_only=True, axis=1)
        df.loc[:,'Min'] = min_val
        df.loc[:,'Max'] = max_val
        df.loc[:,'Total'] = total
        print (df)
    print('\n')

def log_col_weights(session: Session):
    print ('Col Weights: \n ')
    if session.metagraph.uids != None and session.metagraph.weights != None:
        uids = session.metagraph.uids.tolist()
        weights = session.metagraph.W[:,0].tolist()
        weights, uids  = zip(*sorted(zip(weights, uids), reverse=True))
        df = pd.DataFrame([weights], columns=uids)
        df.rename_axis("[uid]", axis=1)
        max_val = df.max(numeric_only=True, axis=1)
        min_val = df.min(numeric_only=True, axis=1)
        total = df.sum(numeric_only=True, axis=1)
        df.loc[:,'Min'] = min_val
        df.loc[:,'Max'] = max_val
        df.loc[:,'Total'] = total
        print (df)
    print('\n')

def log_batch_weights(session: Session, history: List[bittensor.synapse.SynapseOutput]):
    print ('Batch Weights: \n ')
    weights_sum = history[0].weights
    for output in history[1:]:
        weights_sum += torch.mean(output.weights, axis=0)
    weights_sum = weights_sum.tolist()

    uids = session.metagraph.uids.tolist()
    _, sorted_uids  = zip(*sorted(zip(weights_sum, uids), reverse=True))

    rows = []
    for output in history:
        batch_weights = torch.mean(output.weights, axis=0).tolist()
        _, sorted_batch_weights  = zip(*sorted( zip(weights_sum, batch_weights), reverse=True))
        rows.append(sorted_batch_weights)
    df = pd.DataFrame(rows, columns = sorted_uids)
    min_row = df.min(numeric_only=True, axis=1)
    min_col = df.min(numeric_only=True, axis=0)
    max_row = df.max(numeric_only=True, axis=1)
    max_col = df.max(numeric_only=True, axis=0)
    mean_row = df.mean(numeric_only=True, axis=1)
    mean_col = df.mean(numeric_only=True, axis=0)
    df.loc[:,'Min'] = min_row
    df.loc[:,'Max'] = max_row
    df.loc[:,'Mean'] = mean_row
    df.loc['Min'] = min_col
    df.loc['Max'] = max_col
    df.loc['Mean'] = mean_col
    df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
    print (df)
    print('\n')
        
def log_outputs(history: List[bittensor.synapse.SynapseOutput]):
    print ('Training Outputs: \n ')
    cols = ['Batch Size', 'Total Loss', 'Local Loss', 'Remote Loss', 'Distillation Loss'] + list(history[0].metadata.keys())
    rows = []
    for output in history:
        row = []

        if output.local_hidden != None:
            row.append(output.local_hidden.shape[0])
        else:
            row.append('')

        if output.loss != None:
            row.append(output.loss.item())
        else:
            row.append('')

        if output.local_target_loss != None:
            row.append(output.local_target_loss.item())
        else:
            row.append('')

        if output.remote_target_loss != None:
            row.append(output.remote_target_loss.item())
        else:
            row.append('')

        if output.distillation_loss != None:
            row.append(output.distillation_loss.item())
        else:
            row.append('')

        for key in output.metadata.keys():
            row.append(output.metadata[key].item())
        
        rows.append(row)
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame(rows, columns=cols)
    min_val = df.min(numeric_only=True, axis=0)
    max_val = df.max(numeric_only=True, axis=0)
    mean_val = df.mean(numeric_only=True, axis=0)
    df.loc['Min'] = min_val
    df.loc['Max'] = max_val
    df.loc['Mean'] = mean_val
    print (df)
    print('\n')


def log_training_output_history(session, epoch, batch_idx, batch_size, total_examples, history):
   
    # Colorize outputs and log.
    processed = ((batch_idx + 1) * batch_size)
    progress = (100. * processed) / total_examples
    total_str = colored('{}'.format(total_examples), 'red')
    processed_str = colored('{}'.format(processed), 'green')
    progress_str = colored('{:.2f}%'.format(progress), 'green')
    logger.info('Epoch: {} [{}/{} ({})]', epoch, processed_str, total_str, progress_str)
    print('\n')

    print ('Outputs: \n ')   
    try:
        rows = []
        cols = ['Batch Size', 'Total Loss', 'Local Loss', 'Remote Loss', 'Distillation Loss'] + list(history[0].metadata.keys())
        for output in history:
            row = []
            row.append(output.local_hidden.shape[0])
            row.append(output.loss.item())
            row.append(output.local_target_loss.item())
            row.append(output.remote_target_loss.item())
            row.append(output.distillation_loss.item())
            for key in output.metadata.keys():
                row.append(output.metadata[key].item())
            rows.append(row)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        df = pd.DataFrame(rows, columns=cols)
        min_val = df.min(numeric_only=True, axis=0)
        max_val = df.max(numeric_only=True, axis=0)
        mean_val = df.mean(numeric_only=True, axis=0)
        df.loc['Min'] = min_val
        df.loc['Max'] = max_val
        df.loc['Mean'] = mean_val
        print (df)
    except Exception as e:
        print ('Not set. Exception occured: {}'.format(e))
    print('\n')

    # Log chain weights
    print ('Chain weights: \n ')
    if session.metagraph.state.uids != None and session.metagraph.chain_weights != None:
        uids = session.metagraph.state.uids.tolist()
        weights = session.metagraph.chain_weights().tolist()
        weights, uids  = zip(*sorted(zip(weights, uids), reverse=True))
        df = pd.DataFrame([weights], columns=uids)
        df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
        max_val = df.max(numeric_only=True, axis=1)
        min_val = df.min(numeric_only=True, axis=1)
        total = df.sum(numeric_only=True, axis=1)
        df.loc[:,'Min'] = min_val
        df.loc[:,'Max'] = max_val
        df.loc[:,'Total'] = total
        print (df)
    print('\n')

    # Log batch weights
    print ('Batch Weights: \n ')
    if history[0].weights != None:
        batch_weights_list = []
        for output in history:
            batch_weights = torch.mean(output.weights, axis=0).tolist()
            _, batch_weights  = zip(*sorted( zip(weights, batch_weights), reverse=True))
            batch_weights_list.append(batch_weights)
        df = pd.DataFrame(batch_weights_list, columns=uids)
        min_row = df.min(numeric_only=True, axis=1)
        min_col = df.min(numeric_only=True, axis=0)
        max_row = df.max(numeric_only=True, axis=1)
        max_col = df.max(numeric_only=True, axis=0)
        mean_row = df.mean(numeric_only=True, axis=1)
        mean_col = df.mean(numeric_only=True, axis=0)
        df.loc[:,'Min'] = min_row
        df.loc[:,'Max'] = max_row
        df.loc[:,'Mean'] = mean_row
        df.loc['Min'] = min_col
        df.loc['Max'] = max_col
        df.loc['Mean'] = mean_col
        df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
        print (df)
    print('\n')
    
    # Log return ops.
    print('Requests: \n ')
    if history[0].request_sizes != None:
        sizes = []
        for output in history:
            batch_sizes = output.request_sizes.tolist()
            _, batch_sizes  = zip(*sorted(zip(weights, batch_sizes), reverse=True))
            sizes.append(batch_sizes)
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        df = pd.DataFrame(sizes, columns=uids)
        total_row = df.sum(numeric_only=True, axis=1)
        total_col = df.sum(numeric_only=True, axis=0)
        min_row = df.min(numeric_only=True, axis=1)
        min_col = df.min(numeric_only=True, axis=0)
        max_row = df.max(numeric_only=True, axis=1)
        max_col = df.max(numeric_only=True, axis=0)
        mean_row = df.mean(numeric_only=True, axis=1)
        mean_col = df.mean(numeric_only=True, axis=1)
        df.loc[:,'Min'] = min_row
        df.loc[:,'Max'] = max_row
        df.loc[:,'Mean'] = mean_row
        df.loc[:,'Total'] = total_row
        df.loc['Min'] = min_col
        df.loc['Max'] = max_col
        df.loc['Mean'] = mean_col
        df.loc['Total'] = total_col
        df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
        print (df)
    print('\n')