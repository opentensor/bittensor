# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import math
import time
import torch
from rich import print as rich_print
from rich.table import Table
from rich.errors import MarkupError
from rich.style import Style
from typing import List, Tuple, Callable, Dict, Any, Union, Set
import datetime
from prometheus_client import Counter, Gauge, Histogram, Summary, Info
import bittensor
import bittensor.utils.networking as net

class ValidatorLogger:
    r"""
    Logger object for handling all logging function specific to validator.
    Including console log styling, console table print and prometheus. 
    
    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
    """
    def __init__(self, config = None):
        # Neuron stats recorded by validator neuron/nucleus
        #   [Column_name, key_name, format_string, rich_style]  # description
        self.config = config
        self.neuron_stats_columns = [
            ['UID', 'uid', '{:.0f}', 'cyan'],  # neuron UID
            ['Upd!', 'updates!', '{}', 'bright_yellow'],  # number of exponential moving average updates with zeroing on
            ['nUpd', 'updates_shapley_values_nxt', '{}', 'bright_yellow'],  # number of exponential moving average updates to nShap
            ['mUpd', 'updates_shapley_values_min', '{}', 'bright_yellow'],  # number of exponential moving average updates to mShap
            ['nTime', 'response_time_nxt', '{:.2f}', 'yellow'],  # response time to TextCausalLMNext forward requests [TextCausalLMNext]
            ['sTime', 'response_time', '{:.2f}', 'yellow'],  # response time to TextCausalLM forward requests
            ['Route', 'routing_score', '{:.3f}', 'grey30'],  # validator routing score (higher preferred)
            ['Weight', 'weight', '{:.5f}', 'green'],  # weight set on substrate (each epoch)
            ['nShap!', 'shapley_values_nxt!', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for phrase validation (zeroing) [TextCausalLMNext]
            ['nShap', 'shapley_values_nxt', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for phrase validation [TextCausalLMNext]
            ['mShap!', 'shapley_values_min!', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley (zeroing)
            ['mShap', 'shapley_values_min', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley
            ['sLoss', 'loss', '{:.2f}', 'bright_cyan'],  # next token prediction loss average over sequence
            ['vLoss', 'loss_val', '{:.2f}', 'bright_cyan'],  # next token prediction loss for validation task
            ['nvLoss', 'loss_val_nxt', '{:.2f}', 'bright_cyan'],  # next token prediction loss for validation task [TextCausalLMNext]
            ['nLoss', 'loss_nxt', '{:.2f}', 'bright_cyan'],  # next token phrase prediction loss for phrase validation task [TextCausalLMNext]
            ['RLoss', 'routing_loss', '{:.3f}', 'grey30'],  # MSE between routing_score and conditioned loss
            ['nRLoss', 'routing_loss_nxt', '{:.3f}', 'grey30'],  # MSE between routing_score_nxt and conditioned loss [TextCausalLMNext]
            ['sShap', 'shapley_values', '{:.0f}', 'magenta'],  # Shapley value (=Base+Syn) over sequence
            ['vShap', 'shapley_values_val', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for validation
            ['sBase', 'base_params', '{:.0f}', ''],  # parameter count estimate via adjusted scaling law
            ['vBase', 'base_params_val', '{:.0f}', ''],  # square root parameter count estimate for validation task
            ['nBase', 'base_params_nxt', '{:.0f}', ''],  # square root parameter count estimate for phrase validation task [TextCausalLMNext]
            ['nParam~', 'est_params_nxt', '{:.2g}', 'magenta'],  # parameter count estimate for phrase validation task [TextCausalLMNext]
            ['nDiv', 'logits_divergence_nxt', '{:.2g}', ''],  # logits divergence avg compared to network prob dist [TextCausalLMNext]
            ['nExc', 'logits_excess_nxt', '{:.2f}', ''],  # logits divergence excess avg above network avg + std [TextCausalLMNext]
            ['sSyn', 'synergy', '{:.0f}', 'white'],  # Shapley pairwise synergy over sequence loss (parameter count estimate)
            ['vSyn', 'synergy_val', '{:.0f}', 'white'],  # Shapley pairwise synergy over validation loss (count estimate)
            ['nSyn', 'synergy_nxt', '{:.0f}', 'white'],  # Shapley pairwise synergy over phrase validation loss (count estimate) [TextCausalLMNext]
            ['sSynD', 'synergy_loss_diff', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over sequence loss (loss difference)
            ['vSynD', 'synergy_loss_diff_val', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over validation loss (loss difference)
            ['nSynD', 'synergy_loss_diff_nxt', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over phrase validation loss (loss difference) [TextCausalLMNext]
        ]
        # console_width (:obj:`int`, `required`):
        #     Config console width for table print.
        self.console_width = self.config.get('width', None) if self.config else None
        self.prometheus = ValidatorPrometheus(config)

    def print_response_table(
        self, 
        batch_predictions: List, 
        stats: Dict, 
        sort_col: str, 
        task_repeat: int = 4, 
        tasks_per_server: int = 3
    ):
        r""" 
        Prints the query response table: top prediction probabilities and texts for batch tasks.
        
            Args:
                batch_predictions (:obj:`List[Union[str, Dict{torch.Tensor, str}]]`, `required`):
                    Predictions in string per task per uid. In the format of [(task, {uid, "prob: phrase" })] of length batch size.
                stats (:obj:`Dict{Dict}`, `required`):
                    Statistics per endpoint for this batch. In the format of {uid, {statistics}}.
                sort_col (:type:`str`, `required`):
                    Column name used for sorting. Options from self.neuron_stats_columns[:, 1].
                task_repeat (:type:`int`, `required`):
                    The number of servers to compare against under the same set of task.
                tasks_per_server (:type:`int`, `required`):
                    How many tasks to show for each server.
        """
        # === Batch permutation ===
        batch_size = len(batch_predictions)
        if batch_size == 0:
            return
        batch_perm = torch.randperm(batch_size)  # avoid restricting observation to predictable subsets

        # === Column selection ===
        columns = [c[:] for c in self.neuron_stats_columns if c[1] in ['uid', sort_col, 'loss_nxt', 'synergy_nxt', 'logits_excess_nxt']]
        col_keys = [c[1] for c in columns]

        # === Sort rows ===
        sort = sorted([(uid, s[sort_col]) for uid, s in stats.items() if sort_col in s],
                    reverse='loss' not in sort_col, key=lambda _row: _row[1])
        if sort_col in col_keys:
            sort_idx = col_keys.index(sort_col)  # sort column with key of sort_col
            columns[sort_idx][0] += '\u2193'  # ↓ downwards arrow (sort)

        for i, (uid, val) in enumerate(sort):
            # === New table section ===
            if i % task_repeat == 0:
                table = Table(width=self.console_width, box=None)
                if i == 0:
                    table.title = f"[white bold] Query responses [/white bold] | " \
                                f"[white]context[/white][bold]continuation[/bold] | .prob: 'prediction'"

                for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
                    table.add_column(col, style=stl, justify='right')

            # === Last table section ===
            if i == len(sort) - 1:
                table.caption = f'[bold]{len(sort)}[/bold]/{len(stats)} (respond/topk) | ' \
                                f'[bold]{tasks_per_server}[/bold] tasks per server | ' \
                                f'repeat tasks over [bold]{task_repeat}[/bold] servers ' \
                                f'[white]\[{math.ceil(1. * len(sort) / task_repeat) * tasks_per_server}/' \
                                f'{batch_size} batch tasks][/white]'

            # === Row addition ===
            row = [txt.format(stats[uid][key]) for _, key, txt, _ in columns]
            for j in range(tasks_per_server):
                batch_item = ((i // task_repeat) * tasks_per_server + j) % batch_size  # repeat task over servers, do not exceed batch_size
                task, predictions = batch_predictions[batch_perm[batch_item]]
                row += [predictions[uid]]

                if i % task_repeat == 0:
                    table.add_column(task, header_style='not bold', style='', justify='left')

            table.add_row(*row)

            # === Table print ===
            if (i == len(sort) - 1) or (i % task_repeat == task_repeat - 1):
                try:
                    rich_print(table)
                except MarkupError as e:
                    print(e)
                else:
                    if i == len(sort) - 1:
                        print()

    def print_synergy_table(
        self, 
        stats: Dict, 
        syn_loss_diff: Dict ,
        sort_col: str, 
    ):
        r""" 
        Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal).
            
            Args:
                stats (:obj:`Dict{Dict}`, `required`):
                    Statistics per endpoint for this batch. In the format of {uid, {statistics}}.
                syn_loss_diff (:obj:`Dict`, `required`):
                    Dictionary table of pairwise synergies as loss reductions, with direct loss on diagonal.
                sort_col (:type:`str`, `required`):
                    Column name used for sorting. Options from self.neuron_stats_columns[:, 1]. 
        """
        sort = sorted([(uid, s[sort_col]) for uid, s in stats.items() if sort_col in s],
                    reverse='loss' not in sort_col, key=lambda _row: _row[1])
        uid_col = self.neuron_stats_columns[0]  # [Column_name, key_name, format_string, rich_style]
        columns = [uid_col] + [[f'{s[0]}', '', '{:.2f}', ''] for s in sort]
        rows = [[uid_col[2].format(s[0])] +
                [('[bright_cyan]{:.2f}[/bright_cyan]' if t == s else
                '[magenta]{:.3f}[/magenta]' if syn_loss_diff[s[0]][t[0]] > 0 else
                '[dim]{:.0f}[/dim]').format(syn_loss_diff[s[0]][t[0]]).replace('0.', '.') for t in sort] for s in sort]

        # === Synergy table ===
        table = Table(width=self.console_width, box=None)
        table.title = f'[white] Synergy table [/white] | Pairwise synergy'
        table.caption = f'loss decrease'

        for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
            table.add_column(col, style=stl, justify='right')
        for row in rows:
            table.add_row(*row)

        if len(rows):
            rich_print(table)
            print()

    def print_stats_table(
        self, 
        stats: Dict, 
        sort_col: str, 
        title: str, 
        caption: str, 
        mark_uids=None
    ):
        r""" 
        Gathers data and constructs neuron statistics table and prints it.

            Args: 
                stats (:obj:`Dict{Dict}`, `required`):
                    Statistics per endpoint for this batch. In the format of {uid, {statistics}}.
                sort_col (:type:`str`, `required`):
                    Column name used for sorting. Options from self.neuron_stats_columns[:, 1].
                title (:type:`str`, `required`):
                    Title of the table.
                caption (:type:`str`, `required`):
                    Caption shown at the end of table.
        """
        # === Gather columns and rows ===
        if mark_uids is None:
            mark_uids = list()
        stats_keys = [set(k for k in stat)
                    for stat in stats.values() if sort_col in stat]  # all available stats keys with sort_col

        if len(stats_keys) == 0:
            return  # nothing to print

        stats_keys = set.union(*stats_keys)
        columns = [c[:] for c in self.neuron_stats_columns if c[1] in stats_keys]  # available columns intersecting with stats_keys
        rows = [[('', 0) if key not in stat
                else (('* ' if key == 'uid' and mark_uids and uid in mark_uids else '') + txt.format(stat[key]), stat[key])
                for _, key, txt, _ in columns]
                for uid, stat in stats.items() if sort_col in stat]  # only keep rows with at least one non-empty cell

        if len(columns) == 0 or len(rows) == 0:
            return  # nothing to print

        # === Sort rows ===
        col_keys = [c[1] for c in columns]
        if sort_col in col_keys:
            sort_idx = col_keys.index(sort_col)  # sort column with key of sort_col
            columns[sort_idx][0] += '\u2193'  # ↓ downwards arrow (sort)
            rows = sorted(rows, reverse='loss' not in sort_col, key=lambda _row: _row[sort_idx][1])  # sort according to sortcol

        # === Instantiate stats table ===
        table = Table(width=self.console_width, box=None, row_styles=[Style(bgcolor='grey15'), ""])
        table.title = title
        table.caption = caption

        for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
            table.add_column(col, style=stl, justify='right')
        for row in rows:
            table.add_row(*[txt for txt, val in row])

        # === Print table ===
        rich_print(table)

    def print_synapse_table(
        self, 
        name: str, 
        stats: Dict, 
        sort_col: str, 
        time: time.time
    ):
        r""" 
        Prints the evaluation of the neuron responses to the validator request
            
            Args: 
                stats (:obj:`Dict{Dict}`, `required`):
                    Statistics per endpoint for this batch. In the format of {uid, {statistics}}.
                sort_col (:type:`str`, `required`):
                    Column name used for sorting. Options from self.neuron_stats_columns[:, 1].
                name (:obj:`str`, `required`):
                    Name of synapse for the title of the table.
                start_time (:obj:`time.time`, `required`):
                    Starting time for shapley calculation.

        """
        self.print_stats_table(stats, sort_col,
                    f'[white] \[{name}] responses [/white] | Validator forward',  # title
                    f'[bold]{len([s for s in stats.values() if len(s) and sort_col in s])}[/bold]/'
                    f'{len(stats)} (respond/topk) | '
                    f'[bold]Synapse[/bold] | [white]\[{time:.3g}s][/white]'  # caption
                    )

    def print_weights_table(
            self,
            min_allowed_weights: int,
            max_weight_limit: int,
            neuron_stats: Dict,
            title: str,
            metagraph_n: int, 
            sample_uids: torch.Tensor, 
            sample_weights: torch.Tensor, 
            include_uids: List = None, 
            num_rows: int = None
        ):
        r""" 
        Prints weights table given sample_uids and sample_weights.
        
            Args:
                min_allowed_weights (:type:`int`, `required`):
                    subtensor minimum allowed weight to set.
                max_weight_limit (:type:`int`, `required`):
                    subtensor maximum allowed weight to set.
                neuron_stats (:obj:`Dict{Dict}`, `required`):
                    Statistics per endpoint for this batch. In the format of {uid, {statistics}}.
                title (:type:`str`, `required`):
                    Title of the table.
                metagraph_n (:type:`int`, `required`):
                    Total number of uids in the metagraph.
                sample_uids (:obj:`torch.Tensor`, `required`):
                    Uids to set weight for. 
                sample_weights (:obj:`torch.Tensor`, `required`):
                    Weights to set uids for. 
                include_uids (:type:`list`, `optional`): 
                    Set of uids to inculde in the table.
                num_rows (:type:`int`, `optional`): 
                    Total number of uids to print in total.
        """
        # === Weight table ===
        # Prints exponential moving average statistics of valid neurons and latest weights
        _neuron_stats = {}
        uid_weights = []  # (uid, weight) tuples for sorting to find top/bottom weights
        unvalidated = []

        if len(sample_weights) == 0:
            return 

        for uid, weight in zip(sample_uids.tolist(), sample_weights.tolist()):
            if uid in neuron_stats:
                _neuron_stats[uid] = {k: v for k, v in neuron_stats[uid].items()}
                _neuron_stats[uid]['weight'] = weight
                uid_weights += [(uid, weight)]
            else:
                unvalidated += [uid]

        if include_uids is not None and num_rows is not None:
            sorted_uids = sorted(uid_weights, key=lambda tup: tup[1])
            top_bottom_uids = [_uid for _uid, _ in sorted_uids[:5] + sorted_uids[-10:]]
            _include_uids = set(include_uids) | set(top_bottom_uids)
            avail_include_uids = list(set(_neuron_stats.keys()) & _include_uids)  # exclude include_uids with no stats
            if len(_neuron_stats) > num_rows:  # limit table to included_uids and remaining sample up to num_rows
                remaining_uids = set(_neuron_stats.keys()) - _include_uids  # find sample remaining, loses sample ordering
                remaining_uids = [uid for uid in _neuron_stats if uid in remaining_uids]  # recover sample ordering
                limited_uids = avail_include_uids + remaining_uids[:num_rows - len(_include_uids)]
                _neuron_stats = {uid: stats for uid, stats in _neuron_stats.items() if uid in limited_uids}

        print()
        self.print_stats_table(_neuron_stats, 'weight',
                    f'[white] Neuron weights [/white] | ' + title,  # title
                    f'Validated {min_allowed_weights}/'
                    f'[bold]{len(neuron_stats)}[/bold]/{metagraph_n} (min/[bold]valid[/bold]/total) | '
                    f'sum:{sample_weights.sum().item():.2g} '
                    f'[white] max:[bold]{sample_weights.max().item():.4g}[/bold] / '
                    f'min:[bold]{sample_weights.min().item():.4g}[/bold] [/white] '
                    f'\[{max_weight_limit:.4g} allowed]',  # caption
                    mark_uids=include_uids)

    def print_console_validator_identifier(
        self, 
        uid: int, 
        wallet: 'bittensor.Wallet', 
        external_ip: str,
    ):  
        r""" Console print for validator identifier.
        """

        # validator identifier status console message (every 25 validation steps)
        rich_print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
        f"{f'[bright_white]core_validator[/bright_white]'.center(16 + len('[bright_white][/bright_white]'))} | "
        f"UID [cyan]{uid}[/cyan] "
        f"[dim white not bold][{external_ip}][/dim white not bold] "
        f"[white not bold]cold:[bold]{wallet.name}[/bold]:"
        f"[bright_white not bold]{wallet.coldkeypub.ss58_address}[/bright_white not bold] "
        f"[dim white]/[/dim white] "
        f"hot:[bold]{wallet.hotkey_str}[/bold]:"
        f"[bright_white not bold]{wallet.hotkey.ss58_address}[/bright_white not bold][/white not bold]")

    def print_console_metagraph_status(
        self, 
        uid: int, 
        metagraph: 'bittensor.Metagraph', 
        current_block: int, 
        start_block: int, 
        network: str,
        netuid: int
    ):
        r""" Console print for current validator's metagraph status. 
        """
        # validator update status console message
        rich_print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
        f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
        f'Updated [yellow]{current_block - metagraph.last_update[uid]}[/yellow] [dim]blocks ago[/dim] | '
        f'Dividends [green not bold]{metagraph.dividends[uid]:.5f}[/green not bold] | '
        f'Stake \u03C4[magenta not bold]{metagraph.total_stake[uid]:.5f}[/magenta not bold] '
        f'[dim](retrieved [yellow]{current_block - start_block}[/yellow] blocks ago from {network})[/dim]')

    def print_console_query_summary(
        self, 
        current_block: int, 
        start_block: int,
        blocks_per_epoch: int, 
        epoch_steps: int, 
        epoch: int,
        responsive_uids: List, 
        queried_uids: List, 
        step_time: float, 
        epoch_responsive_uids: Set, 
        epoch_queried_uids: Set
    ):
        r""" Console print for query summary.
        """
        rich_print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
        f"{f'[magenta dim not bold]#{current_block}[/magenta dim not bold]'.center(16 + len('[magenta dim not bold][/magenta dim not bold]'))} | "
        f'[green not bold]{current_block - start_block}[/green not bold]/'
        f'[white not bold]{blocks_per_epoch}[/white not bold] [dim]blocks/epoch[/dim] | '
        f'[white not bold]Step {epoch_steps}[white not bold] '
        f'[dim] Epoch {epoch}[/dim] | '
        f'[bright_green not bold]{len(responsive_uids)}[/bright_green not bold]/'
        f'[white]{len(queried_uids)}[/white] '
        f'[[yellow]{step_time:.3g}[/yellow]s] '
        f'[dim white not bold][green]{len(epoch_responsive_uids)}[/green]/'
        f'{len(epoch_queried_uids)}[/dim white not bold]')

    def print_console_subtensor_weight(
        self,
        sample_weights: torch.Tensor,
        epoch_responsive_uids: Set, 
        epoch_queried_uids: Set, 
        max_weight_limit: float, 
        epoch_start_time: time.time,
    ):
        r""" Console print for weight setting to subtensor.
        """

        rich_print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
        f"{f'[bright_white]Set weights[/bright_white]'.center(16 + len('[bright_white][/bright_white]'))} | "
        f'[bright_green not bold]{len(sample_weights)}[/bright_green not bold] [dim]weights set[/dim] | '
        f'[bright_green not bold]{len(epoch_responsive_uids)}[/bright_green not bold]/'
        f'[white]{len(epoch_queried_uids)}[/white] '
        f'[dim white not bold][green]responsive[/green]/queried[/dim white not bold] '
        f'[[yellow]{time.time() - epoch_start_time:.0f}[/yellow]s] | '
        f'[dim]weights[/dim] sum:{sample_weights.sum().item():.2g} '
        f'[white] max:[bold]{sample_weights.max().item():.4g}[/bold] / '
        f'min:[bold]{sample_weights.min().item():.4g}[/bold] [/white] '
        f'\[{max_weight_limit:.4g} allowed]')

    def format_predictions(
        uids: torch.Tensor, 
        query_responses: List[List[torch.FloatTensor]],
        return_ops: List[torch.LongTensor], 
        inputs: torch.FloatTensor,
        validation_len: int, 
        index_s: int = 0, 
        number_of_predictions: int = 3
    ) -> List:
        r""" Format batch task topk predictions for rich table print of query responses.
        """
        batch_predictions = []
        std_tokenizer = bittensor.tokenizer()

        # === Batch iteration ===
        for batch_item in range(inputs.shape[0]):
            # === Task formatting ===
            context = inputs[batch_item][:-validation_len]
            answer = inputs[batch_item][-validation_len:]

            context = repr(std_tokenizer.decode(context))[1:-1][-30:]  # strip '' and truncate
            answer = repr(std_tokenizer.decode(answer))[1:-1][:15]  # strip '' and truncate

            task = f"[reverse]{context}[/reverse][bold]{answer}[/bold]"

            # === Prediction formatting ===
            predictions = {}
            for index, uid in enumerate(uids.tolist()):
                if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
                    topk_tensor = query_responses[index][index_s]  # [batch_size, (topk + 1), max_len] (prob_k) + floor_prob
                    topk_tokens = topk_tensor[batch_item, :-1, 1:].int()  # [batch_size, topk, max_len - 1] Phrase tokens with ignore_index token for padding.
                    topk_probs = topk_tensor[batch_item, :-1, 0]  # [batch_size, topk] Probabilities for each phrase in topk

                    # === Topk iteration ===
                    topk_predictions = ''
                    for i in range(number_of_predictions):
                        phrase = topk_tokens[i]
                        phrase = phrase[phrase >= 0]  # strip negative ignore_index = -100
                        phrase_str = repr(std_tokenizer.decode(phrase))[:15]  # decode, escape and truncate
                        prob = f'{topk_probs[i]:.3f}'.lstrip('0').replace('1.000', '1.00')
                        topk_predictions += f"[green]{prob}[/green]: {phrase_str} "

                    predictions[uid] = topk_predictions[:-1]  # strip trailing space

            batch_predictions += [(task, predictions)]

        return batch_predictions

    def step_log(
            self, 
            uid, 
            wallet,
            metagraph,
            subtensor,
            netuid,
            neuron_stats,
            epoch_status, 
            epoch_params, 
            step_status, 
            stats,
            debug,
            synapse_keys,
        ): 

        # === Synergy table ===
        # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
        self.vlogger.print_synergy_table(stats, step_status.syn_loss_diff, 'loss_nxt')

        # === Neuron responses (table) ===
        # Prints the evaluation of the neuron responses to the validator request
        self.vlogger.print_synapse_table( 'Stats table', stats, 'loss_nxt', step_status.shapley_time)

        # === ALL logging for validation step (including console message, console tables, prometheus, wandb) ===
        if epoch_status.step % 25 == 1:
            # console message - validator identifier status (every 25 validation steps)
            self.vlogger.print_console_validator_identifier(uid, wallet, net.get_external_ip())
            # console message - validator update status (every 25 validation steps)
            self.vlogger.print_console_metagraph_status(uid, metagraph, step_status.current_block, epoch_params.start_block, subtensor.network, netuid)

        # console message - query summary (every validation step)
        self.vlogger.print_console_query_summary(
            current_block = step_status.current_block, 
            start_block = epoch_params.start_block,
            blocks_per_epoch = epoch_params.blocks_per_epoch, 
            epoch_steps = epoch_params.epoch_steps, 
            epoch = epoch_status.step, 
            responsive_uids = step_status.responsive_uids, 
            queried_uids = step_status.queried_uids, 
            step_time = step_status.step_time, 
            epoch_responsive_uids = epoch_status.responsive_uids, 
            epoch_queried_uids = epoch_status.queried_uids
        )

        if debug:
            # console table - stats table (every validation step)
            # Prints exponential moving average statistics of valid neurons from latest validator forward 
            self.vlogger.print_stats_table({uid: neuron_stats[uid]
                            for uid, stat in stats.items() if len(set(stat.keys()) & set(synapse_keys))},
                        self.weight_key,
                        f'[white] Stats update [/white] | ' + str(self),  # title
                        f'#{step_status.current_block}: '
                        f'[bold]{step_status.current_block - epoch_params.start_block}[/bold]/{epoch_params.blocks_per_epoch} (blocks/epoch) | '
                        f'Epoch {self.epoch} | '
                        f'[white] Step {epoch_status.step} ({self.global_step} global) \[{step_status.step_time:.3g}s] [/white]')  # caption

            # console table - weight table (every validation step)
            sample_uids, sample_weights = self.calculate_weights()
            self.vlogger.print_weights_table(
                min_allowed_weights = self.subtensor.min_allowed_weights(netuid=self.config.netuid) if self.config.subtensor.network == 'finney' else self.subtensor.min_allowed_weights,
                max_weight_limit = self.subtensor.max_weight_limit(netuid=self.config.netuid)  if self.config.subtensor.network == 'finney' else self.subtensor.max_weight_limit,
                neuron_stats = self.neuron_stats,
                title = str(self),
                metagraph_n = self.metagraph.n, 
                sample_uids = sample_uids, 
                sample_weights = sample_weights,
                include_uids=list(stats.keys()), 
                num_rows=len(stats) + 25
            )

    def epoch_log(self, debug, sample_uids, sample_weights, epoch_status, subtensor, metagraph, netuid, neuron_stats):

        # === ALL end of epoch logging (including console message, console table, prometheus, wandb)===
        if debug:
                # console table - weight table (every end of epoch)
            self.vlogger.print_weights_table(
                min_allowed_weights = subtensor.min_allowed_weights(netuid=netuid),
                max_weight_limit = subtensor.max_weight_limit(netuid=netuid),
                neuron_stats = neuron_stats,
                title = str(self),
                metagraph_n = metagraph.n, 
                sample_uids = sample_uids, 
                sample_weights = sample_weights,
            )  

        # console message - subtensor weight (every end of epoch)
        self.vlogger.print_console_subtensor_weight(
            sample_weights = sample_weights, 
            epoch_responsive_uids = epoch_status.responsive_uids, 
            epoch_queried_uids = epoch_status.queried_uids, 
            max_weight_limit = subtensor.max_weight_limit(netuid=netuid), 
            epoch_start_time = epoch_status.start_time
        )
class ValidatorPrometheus:
    r"""
    Prometheis logging object for validator.
        Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
    """ 
    def __init__(self, config):
        self.config = config
        self.info = Info("neuron_info", "Info sumamries for the running server-miner.")
        self.gauges = Gauge('validator_gauges', 'Gauges for the running validator.', ['validator_gauges_name'])
        self.counters = Counter('validator_counters', 'Counters for the running validator.', ['validator_counters_name'])
        self.step_time = Histogram('validator_step_time', 'Validator step time histogram.', buckets=list(range(0, 2 * bittensor.__blocktime__, 1)))

    def log_run_info(
        self, 
        parameters: torch.nn.parameter.Parameter, 
        uid: int, 
        network: str, 
        wallet: 'bittensor.Wallet'
    ):
        r""" Set up prometheus running info. 
        """

        self.gauges.labels( "model_size_params" ).set( sum(p.numel() for p in parameters) )
        self.gauges.labels( "model_size_bytes" ).set( sum(p.element_size() * p.nelement() for p in parameters) )
        self.info.info({
            'type': "core_validator",
            'uid': str(uid),
            'network': network,
            'coldkey': str(wallet.coldkeypub.ss58_address),
            'hotkey': str(wallet.hotkey.ss58_address),
        })

    def log_epoch_start(
        self, 
        current_block: int, 
        batch_size: int, 
        sequence_length: int, 
        validation_len: int, 
        min_allowed_weights: int, 
        blocks_per_epoch: int, 
        epochs_until_reset: int
    ):
        r""" All prometheus logging at the start of epoch. 
        """
        self.gauges.labels("current_block").set( current_block )
        self.gauges.labels("batch_size").set( batch_size )
        self.gauges.labels("sequence_length").set( sequence_length )
        self.gauges.labels("validation_len").set( validation_len )
        self.gauges.labels("min_allowed_weights").set( min_allowed_weights )
        self.gauges.labels("blocks_per_epoch").set( blocks_per_epoch )
        self.gauges.labels("epochs_until_reset").set( epochs_until_reset )
        self.gauges.labels("scaling_law_power").set( self.config.nucleus.scaling_law_power )
        self.gauges.labels("synergy_scaling_law_power").set( self.config.nucleus.synergy_scaling_law_power )
        self.gauges.labels("epoch_steps").set(0)
    
    def log_step(
        self,
        current_block: int,
        last_update: int,
        step_time: int,
        loss: int
    ):
        r""" All prometheus logging at the each validation step. 
        """
        self.gauges.labels("global_step").inc()
        self.gauges.labels("epoch_steps").inc()
        self.gauges.labels("current_block").set(current_block)
        self.gauges.labels("last_updated").set( current_block - last_update )
        self.step_time.observe( step_time )
        self.gauges.labels('step_time').set( step_time )
        self.gauges.labels("loss").set( loss )

    def log_epoch_end(
        self,
        uid: int,
        metagraph: 'bittensor.Metagraph'
    ):
        r""" All prometheus logging at the end of epoch. 
        """
        self.gauges.labels("epoch").inc()
        self.gauges.labels("set_weights").inc()
        self.gauges.labels("stake").set( metagraph.total_stake[uid] )
        self.gauges.labels("rank").set( metagraph.ranks[uid] )
        self.gauges.labels("trust").set( metagraph.trust[uid] )
        self.gauges.labels("incentive").set( metagraph.incentive[uid] )
        self.gauges.labels("dividends").set( metagraph.dividends[uid] )
        self.gauges.labels("emission").set( metagraph.emission[uid] )

