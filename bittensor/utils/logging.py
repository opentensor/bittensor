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
    processed = ((batch_idx + 1) * total_examples)
    progress = (100. * processed) / total_examples
    total_str = colored('{}'.format(total_examples), 'red')
    processed_str = colored('{}'.format(processed), 'green')
    progress_str = colored('{:.2f}%'.format(progress), 'green')
    logger.info('Epoch: {} [{}/{} ({})]', epoch, processed_str, total_str, progress_str)

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
    print (df)

    # Log chain weights
    uids = session.metagraph.state.uids.tolist()
    weights = session.metagraph.state.weights.tolist()
    weights, uids  = zip(*sorted(zip(weights, uids)))
    df = pd.DataFrame(weights, columns=uids)
    df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
    print (df)

    # Log batch weights
    batch_weights_list = []
    for output in history:
        batch_weights = output.weights
        batch_weights = torch.mean(batch_weights, axis=0)
        _, batch_weights  = zip(*sorted(zip(weights, batch_weights)))
        batch_weights_list.append(batch_weights)
    df = pd.DataFrame(batch_weights_list, columns=uids)
    df.rename_axis('[batch]').rename_axis("[uid]", axis=1)
    print (df)

    # Log return ops.
    retops = []
    for output in history:
        rtops = output.retops
        _, rtops  = zip(*sorted(zip(weights, rtops)))
        retops.append(rtops)
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df = pd.DataFrame(retops_sorted, columns=uids, name = 'batch return ops.')
    df.rename_axis('[batch]').rename_axis("[uid]", axis=1)