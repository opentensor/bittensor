import torch
import numpy as np
import pandas as pd
from loguru import logger
from termcolor import colored

np.set_printoptions(precision=2, suppress=True, linewidth=500, sign=' ')
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

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
    except:
        print ('Not set.')
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