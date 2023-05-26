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

import os
import time
import math
import copy
import queue
import torch
import random
import bittensor
import argparse
import bittensor as bt
import traceback

from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from reward import RewardModel
from gating import GatingModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from datetime import datetime

__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific. Do not include the answer in the question.
'''

__default_base_prompt__ = '''
You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
'''

__default_follow_up_prompt__ = '''
Ask a follow up question.
'''
class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser( full_path )
        config.neuron.reward_path = os.path.expanduser( config.neuron.reward_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path, exist_ok = True)
        if not os.path.exists( config.neuron.reward_path + '/hf_ckpt.pt' ):
            os.makedirs( config.neuron.reward_path, exist_ok = True )
            os.system(
                f"wget -O { config.neuron.reward_path + '/hf_ckpt.pt'} \
                https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt"
            )
        if not config.neuron.dont_save_events:
            # Add custom event logger for the events.
            logger.level("EVENTS", no=38, icon="📝")
            logger.add( 
                config.neuron.full_path + "/" + "completions.log", 
                rotation=config.neuron.events_retention_size, serialize=True, enqueue=True, backtrace=False, diagnose=False, level="EVENTS", 
                format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message} | {extra[prompt]} {extra[completion]} {extra[uids]} {extra[all_uids]} {extra[rewards]}{extra[all_completions]} {extra[block]}"
            )

    def record_event( self, event: SimpleNamespace ):
        self.history.put( event )
        if not self.config.neuron.dont_save_events:
            logger.log(
                "EVENTS", 
                "events", 
                prompt = event.message,
                completion = event.completion,
                uids = event.uids.tolist(),
                all_uids = event.all_uids.tolist(),
                rewards = event.rewards.tolist(),
                all_completions = event.all_completions,
                block = event.block.item(),
            )

    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 1 )
        parser.add_argument( '--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument( '--neuron.base_prompt', type=str, help = 'Prompt injected before a question is completed by miners on the network', default = __default_base_prompt__ )
        parser.add_argument( '--neuron.follow_up_prompt', type=str, help = 'Follow up prompt that is completed by miners on the network.', default = __default_follow_up_prompt__ )
        parser.add_argument( '--neuron.reset_bootstrap_prompt_frequency', type=int, help = 'How frequent to use the base follow up question.', default = 3 )
        parser.add_argument( '--neuron.question_prompt', type=str, help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = __default_question_prompt__ )
        parser.add_argument( '--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument( '--neuron.length_timeout_multiplier', type = int, help = 'Base timeout for all requests.', default = 0.01 )
        parser.add_argument( '--neuron.inference_topk', type = int, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument( '--neuron.training_topk', type = int, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 50 )
        parser.add_argument( '--neuron.training_timeout', type = int, help = 'Query timeout during training', default = 4 )
        parser.add_argument( '--neuron.inference_timeout', type = int, help = 'Query timeout during inference', default = 10 )
        parser.add_argument( '--neuron.inference_only', action = 'store_true', help = 'If set, training off and only inference will be served via axon.', default = False )
        parser.add_argument( '--neuron.axon_off', action = 'store_true', help = 'If set, the axon will be turned off.', default = False )
        parser.add_argument( '--neuron.reward_path', type = str, help = 'Path to reward model.', default = '~/.bittensor/reward_models' )
        parser.add_argument( '--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 100000 )
        parser.add_argument( '--neuron.device', type = str, help = 'Device to run the validator on.', default = "cuda" if torch.cuda.is_available() else "cpu" )
        parser.add_argument( '--neuron.epoch_length_override', type = int, help = 'Override the default timeout', default = -1 )
        parser.add_argument( '--neuron.dont_save_events', action = 'store_true', help = 'If set, we dont save events to a log file.', default = False )
        parser.add_argument( '--neuron.events_retention_size',  type = str,  help = 'Events retention size.', default = "2 GB" )
        parser.add_argument( '--neuron.no_reward_model', action = 'store_true', help = 'If set, we dont load the reward model instead use just the scores.', default = False )
        parser.add_argument( '--neuron.question_random_sample_uids', action = 'store_true', help = 'If set, random sample uids to get question.', default = False )
        parser.add_argument( '--neuron.reward_shift', type = int, help = 'The value to shift rewards for calculation.', default = 3 )
        parser.add_argument( '--neuron.no_nsfw_filter', action = 'store_true', help = 'If set, allow handling of not-safe-for-work messages.', default = False )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        bt.axon.add_args( parser )
        GatingModel.add_args( parser )
        cls.add_args( parser )
        return bt.config( parser )
    
    def __init__( self ):      
        self.config = neuron.config()
        self.check_config( self.config )
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        print( self.config )
        
        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )
        self.tokenizer = AutoTokenizer.from_pretrained( 'EleutherAI/gpt-j-6b' )

        # check if invoking iter() is indeed necessary
        self.dataset = iter(load_dataset('squad_v2', split='train', streaming=True).shuffle(buffer_size=10000))

        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to( self.device )
        self.alpha = 0.99
        self.hotkeys = self.metagraph.hotkeys
        # Reward model
        if not self.config.neuron.no_reward_model:
            bittensor.logging.info('Loading reward model')
            self.reward_model = RewardModel( model_path = 'EleutherAI/gpt-j-6b', device = self.config.neuron.device )
            for fpath in os.listdir( self.config.neuron.reward_path ):
                if fpath.endswith(".pt") or fpath.endswith(".bin"):
                    checkpoint = os.path.join( self.config.neuron.reward_path, fpath )
                    break
            ckpt_state = torch.load( checkpoint )
            self.reward_model.load_state_dict( ckpt_state )
            self.reward_model.eval()
            self.reward_model.half()
            self.reward_model.requires_grad_( False )
            self.reward_model.to( self.device )
            bittensor.logging.info('done loading reward model')

        # Init the gating model which learns which miners to select for each query.
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to( self.device )
        # Denddrite pool for querying the network.
        self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        # History of forward events.
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )
        # Get a list of peers delegating to me
        delegated = self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )
        self.my_nominators = { nomin[0]: nomin[1] for nomin in delegated[0][0].nominators } if len(delegated) else {}

        self.load()
        self.check_weights()

        # set up filter model
        filter_model_path = 'facebook/roberta-hate-speech-dynabench-r4-target'
        self.filter_model = AutoModelForSequenceClassification.from_pretrained(filter_model_path).to(self.device)
        self.filter_tokenizer = AutoTokenizer.from_pretrained(filter_model_path)
        self.filter_tokenizer.pad_token = self.filter_tokenizer.eos_token
        self.filter_message_count = 0

        # Axon set and served for inference requests, unless --neuron.axon_off flag is set.
        if not self.config.neuron.axon_off:
            # Build synapse entrypoint.
            class Synapse( bittensor.TextPromptingSynapse ):
                def priority( _, forward_call: "bittensor.TextPromptingForwardCall" ) -> float:
                    if forward_call.src_hotkey == self.wallet.hotkey.ss58_address: return math.inf # myself.
                    elif forward_call.src_hotkey in self.my_nominators: return self.my_nominators[ forward_call.src_hotkey ].tao # Delegates.
                    else: return 0.0 # Everyone else.

                def blacklist( _, forward_call: "bittensor.TextPromptingForwardCall" ) -> bool:
                    if forward_call.src_hotkey == self.wallet.hotkey.ss58_address: 
                        return True

                    elif forward_call.src_hotkey in self.metagraph.hotkeys:
                        uid =  self.metagraph.hotkeys.index(forward_call.src_hotkey)
                        if self.metagraph.validator_permit[uid]:
                            return True         
                        return False # Non Validator miners
                    
                    elif forward_call.src_hotkey in self.my_nominators:
                        return False # Delegates, dont blacklist.
                    else: 
                        return False # Everyone else, dont blacklist.

                def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass
                
                def forward( _, messages: List[Dict[str, str]] ) -> str:
                    return self.inference(
                        messages = messages,
                        timeout = self.config.neuron.inference_timeout
                    )
                
                def multi_forward( _, messages: List[Dict[str, str]] ) -> str:
                    return self.inference(
                        messages = messages,
                        timeout = self.config.neuron.inference_timeout,
                        return_all = True
                    )

            # Serve axon.
            self.axon = bittensor.axon(
                wallet = self.wallet,
                metagraph = self.metagraph,
                config = self.config,
            )
            self.synapse = Synapse( axon = self.axon )
            self.axon.start()
            self.subtensor.serve_axon( self.config.netuid, self.axon )

    def filter_message(
            self,
            message 
    ) -> bool:
        """ Check if the message is related to any sexual content. 

        Args: 
            message (str):
                The message that we check if we should filter out.
        Returns: 
            result (bool):
                True indicates we should filter out the result, false indicates the result is safe.
        """
        # If no filter needed, then just return false withough checking.
        if self.config.neuron.no_nsfw_filter: 
            return False
        
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        tokenized = self.filter_tokenizer(message)
        input_ids = tokenized['input_ids']
        bound_score1 = 0.5
        bound_score2 = 0.5
        
        while len(input_ids) > 0:
            _input_ids = input_ids[:512]

            with torch.no_grad():
                output = self.filter_model(torch.tensor([_input_ids]).to(self.device))
            
            filter_out = output.logits[0, 0] < bound_score1 or output.logits[0, 1] > bound_score2

            if filter_out:
                bittensor.logging.debug( 'filtered message', message )
                break
            else:
                bittensor.logging.debug( 'safe message', message )
            
            input_ids = input_ids[512:]

        self.filter_message_count += 1
        return filter_out

    def forward(
            self, 
            roles: List[ str ],
            messages: List[ str ],
            topk: Optional[int] = None,
            random_sample_uids: Optional[ bool ] = False,
            train_gating_model: Optional[ bool ] = False,
            train_network: Optional[ bool ] = False,
            timeout: float = None,
            question: bool =  False,
        ) -> SimpleNamespace:
        """
        Queries the network for a response to the passed message using a gating model to select the best uids.
        Trains the gating model based on the rewards calculated for the successful completions and passes rewards
        backward for potential PPO.

        Args:
            roles ( List[ str ] ): 
                roles associated with messages.
            message ( List[ str ] ): 
                messages content for each role. 
            topk (Optional[int]): 
                The number of uids to consider for the query. If None or -1, all uids will be considered.
                If provided, selects the top k uids based on the gating model scores.
            random_sample_uids( bool, default = False ):
                If True, randomly samples the uids to query rather than using topk.
            train_gating_model ( bool, default = False ):
                If True, trains the gating model based on the rewards calculated for the successful completions.
            train_network ( bool, default = False ):
                If True, sends backward messages to the network.
        Returns:
            result (SimpleNamespace): 
                A namespace containing the completion with the highest reward, message, uids,
                rewards, scores, and all completions.
        """
        bittensor.logging.info( 'forward()' )
        bittensor.logging.debug( 'roles', roles )
        bittensor.logging.debug( 'message', messages )

        # Format the messages for the query.
        unravelled_message = ''
        for role, message in list(zip( roles, messages )):
            if role == 'system': unravelled_message += 'system: ' + message + '\n'
            if role== 'assistant': unravelled_message += 'assistant: ' + message + '\n'
            if role == 'user': unravelled_message += 'user: ' + message + '\n'

        # Set `topk` to the number of items in `self.metagraph.n` if `topk` is not provided or is -1.
        # Find the available `uids` that are currently serving.
        # If `topk` is larger than the number of available `uids`, set `topk` to the number of available `uids`.
        available_uids = torch.tensor( [ uid for uid, ax in enumerate( self.metagraph.axons ) if (ax.is_serving) and (not self.metagraph.validator_permit[uid]) ], dtype = torch.int64 ).to( self.device )
        if topk is None or topk == -1: topk = self.metagraph.n.item()
        if topk > len( available_uids ): topk = len( available_uids )
        if len( available_uids ) == 0: bittensor.logging.error('no available uids'); return None
        bittensor.logging.trace( 'available_uids', available_uids )
        bittensor.logging.trace( 'topk', topk)

        # We run the gating network here to get the best uids
        # Use the gating model to generate scores for each `uid`.
        scores = self.gating_model( unravelled_message ).to( self.device )
        bittensor.logging.trace( 'scores', scores )

        # Select the top `topk` `uids` based on the highest `scores`.
        # Use the selected `uids` to query the dendrite pool.
        # Print the `completions`.
        if random_sample_uids:
            topk_uids = torch.tensor( random.sample( available_uids.tolist(), topk ), dtype = torch.int64 ).to( self.device )
        else:
            topk_uids = available_uids[ scores[ available_uids ].sort()[ 1 ][ -topk: ]]
        forward_calls = self.dendrite_pool( 
            roles = roles, 
            messages = messages, 
            uids = topk_uids, 
            timeout = timeout,
        )
        bittensor.logging.trace( 'topk_uids', topk_uids )

        # Filter out any `None` `completions`.
        successful_uids = torch.tensor([uid for uid, call in list(zip(topk_uids, forward_calls)) if call is not None and call.completion is not None and len(call.completion)>10], dtype=torch.int64).to(self.device)
        successful_completions = [call.completion for call in forward_calls if call is not None and call.completion is not None and len(call.completion)>10]
        unsuccessful_uids = torch.tensor([uid for uid in topk_uids if uid not in successful_uids])
        bittensor.logging.debug( 'successful_uids', successful_uids )
        if len( successful_completions ) == 0: bittensor.logging.error('no successful completions'); return None

        # Calculate the rewards for the successful `completions` using the reward model.
        # Print the rewards for all `uids`.`
        if not self.config.neuron.no_reward_model:
            flattened_message_for_reward = ''
            for role_i, message_i in list(zip(roles, messages)):
                if role_i != 'system': flattened_message_for_reward += message_i.strip() + '\n'
            full_completions_for_reward = [ 'Question: ' + flattened_message_for_reward + 'Answer: ' + comp.strip() for comp in successful_completions ]
            completions_for_reward = [comp.strip() for comp in successful_completions] 
            rewards = self.reward_model.reward( full_completions_for_reward, completions_for_reward, difference = True, shift = self.config.neuron.reward_shift).detach().to( self.device )
            bittensor.logging.trace( 'rewards', rewards )
        else:
            rewards = scores[ successful_uids ]

        # Train the gating model using the scores and rewards of the successful `completions`.
        if train_gating_model:
            self.gating_model.backward( scores = scores[ successful_uids ], rewards = rewards )
            bittensor.logging.trace( 'Apply backward to gating model' )

        # Pass rewards backward for potential PPO.
        if train_network:
            self.dendrite_pool.backward( 
                forward_calls = forward_calls,
                rewards = rewards,
                timeout = timeout,
            )
            bittensor.logging.trace( 'Applied backward to network.' )

        best_idx = rewards.detach().argmax()
        bittensor.logging.trace( 'rewards', rewards )
        bittensor.logging.trace('successful_completions', len(successful_completions))
        bittensor.logging.trace('best_idx', best_idx)
        best_completion = successful_completions[best_idx]
        

        # Save the query history in a `result` object.
        # Return the `completion` with the highest reward.
        event = SimpleNamespace( 
            completion = successful_completions[ rewards.argmax( dim = 0 ) ],
            message = message,  
            uids = successful_uids,
            rewards = rewards,
            all_uids = topk_uids,
            all_completions = successful_completions,
            block = self.metagraph.block,
            is_question = message == self.config.neuron.question_prompt,
            best_completion = best_completion
        )
        self.record_event( event ) 

        # First we normalize the rewards with a softmax.
        normalized_rewards = torch.nn.functional.softmax( event.rewards.to( self.device ), dim=0 )

        # We scatter the normalized onto the moving scores (updating them but not changing the source)
        scattered_rewards = self.moving_averaged_scores.scatter(0, event.uids.to( self.device ), normalized_rewards.to( self.device ) )
        scattered_rewards = scattered_rewards.scatter(0, unsuccessful_uids.to( self.device ) , torch.zeros_like(unsuccessful_uids, dtype=torch.float).to( self.device ) )

        # We now perform a moving average of the scattered rewards.
        self.moving_averaged_scores = self.alpha * self.moving_averaged_scores + ( 1 - self.alpha ) * scattered_rewards
        bittensor.logging.trace( 'normalized_rewards', normalized_rewards )
        bittensor.logging.trace( 'scattered_rewards', scattered_rewards )
        bittensor.logging.trace( 'moving_averaged_scores', self.moving_averaged_scores )    
        print("===== Best Completion =====")
        print(f"\n===== {successful_uids[best_idx], rewards[best_idx]} =====\n")

        print('flattened_message_for_reward:\n', flattened_message_for_reward) 
        print('completion:\n', best_completion.strip())

        return event

    def inference( 
            self, 
            messages: List[Dict[str, str]],
            timeout: float,
            dont_use_reward_model: bool = True,
            return_all = False
        ) -> str:
        bittensor.logging.info( 'inference()')

        # Pre-process messages.
        roles = []; contents = []; unravelled_message = ''; user_message = None
        for message_dict in messages:
            roles.append( message_dict['role'] )
            contents.append( message_dict['content'] )
            if message_dict['role'] == 'system': unravelled_message += 'system: ' + message_dict['content'] + '\n'
            if message_dict['role'] == 'assistant': unravelled_message += 'assistant: ' + message_dict['content'] + '\n'
            if message_dict['role'] == 'user': 
                unravelled_message += 'user: ' + message_dict['content'] + '\n'
                user_message = message_dict['content']

        bittensor.logging.info( 'inference message', str(unravelled_message) )

        if user_message and self.filter_message(user_message):
            if return_all:
                return ['Received possible explicit content.']
            else:
                return 'Received possible explicit content.'

        # Get scores for query.
        scores = self.gating_model( unravelled_message ).to( self.device )
        bittensor.logging.info( 'inference scores', str(scores) )

        # Get uids for query.
        uids = scores.sort()[ 1 ][ -self.config.neuron.inference_topk: ]
        bittensor.logging.info( 'inference uids', str(uids) )

        # Query using dendrite pool
        forward_start = time.time()
        bittensor.logging.trace( 'applying dendrite forward' )
        forward_calls = self.inference_pool( 
            roles = roles, 
            messages = contents, 
            uids = uids, 
            timeout = timeout,
        )
        bittensor.logging.trace( 'finished dendrite forward ', time.time() - forward_start )

        # Return longest completion.
        if dont_use_reward_model or self.config.neuron.no_reward_model:
            bittensor.logging.info('not applying the reward model taking the best completed response')
            # Return first best from scores.
            forward_calls.reverse()
            
            if return_all:
                completions = []
                for call in forward_calls:
                    if len( call.completion ) > 0 and not self.filter_message(call.completion):
                        completions.append(call.completion)
                if len(completions) > 0:
                    return completions
            
            else:
                for call in forward_calls:
                    if len( call.completion ) > 0 and not self.filter_message(call.completion):
                        bittensor.logging.info( 'best completion', call.completion )
                        return call.completion

            if return_all:
                return ['no valid completions']
            
            else:
                return 'no valid completions'
            

        else:
            # Format messages for reward model.
            flattened_message_for_reward = ''
            for role_i, message_i in list(zip(roles, messages)):
                if role_i != 'system': flattened_message_for_reward += message_i.strip() + '\n\n'
            completions = [ call.completion for call in forward_calls if len(call.completion) > 0 and not self.filter_message(call.completion) ] 
            flattened_completions_for_reward = [ flattened_message_for_reward + comp.strip() for comp in completions ] 

            # Return best via reward model.
            reward_model_start = time.time()
            completions_for_reward = [comp.strip() for comp in completions] 
            rewards = self.reward_model.reward( flattened_completions_for_reward, completions_for_reward, difference =False ).to( self.device )
            best_completion = completions[ rewards.argmax( dim = 0 ) ]
            bittensor.logging.info('finished applying the reward model ', time.time() - reward_model_start )
            
            if return_all: 
                return completions
            else:
                return best_completion

    def get_question(self, uids, bootstrap_prompt, reset_bootstrap_prompt = False, random_sample_uids = False):
        
        def _get_question(uids, bootstrap_prompt, reset_bootstrap_prompt = False):
            # retrieve the answer
            # sample = next(self.dataset)
            # google_ai_dataset_place_holder = sample['answers']['text'][0]

            if reset_bootstrap_prompt:
                bootstrap_prompt = next(self.dataset)['context'] # google_ai_dataset_place_holder
                self.base_prompt = bootstrap_prompt
                with open('prompt_history.txt', 'a') as file:
                    file.write("============== reset ==================" + '\n')
                    file.write(f"bootstrap prompt: {bootstrap_prompt}" + '\n')
                        
            else:
                bootstrap_prompt = bootstrap_prompt.replace('As an AI language model, ', '') 
            
            question_prompt = f"{bootstrap_prompt}\n\n{self.config.neuron.follow_up_prompt}"
            
            questions = self.dendrite_pool(
                roles = ['user'], 
                messages = [ question_prompt ], 
                uids = uids, 
                timeout = 12,
            )
            
            successful_questions = [question.completion for question in questions if question is not None and question.completion is not None and len(question.completion) > 10 and not self.filter_message(question.completion) ]
            full_completions_for_reward = [ 'Question: ' + bootstrap_prompt + 'Answer: ' + comp.strip() for comp in successful_questions ]
            completions_for_reward = [comp.strip() for comp in successful_questions] 
            reward_diffs = self.reward_model.reward( full_completions_for_reward, completions_for_reward, difference = True, shift = self.config.neuron.reward_shift).to( self.device )
            
            for question, reward_diff in zip(successful_questions, reward_diffs.tolist()):
                print(f"\n=== Question score: {reward_diff}===\n")
                print(question)
                if reward_diff > 0 :
                    return question, reward_diff

            return None, None
        
        def _get_random_uids():
            available_uids = torch.tensor( [ uid for uid, ax in enumerate( self.metagraph.axons ) if ax.is_serving ], dtype = torch.int64 )
            uids = torch.tensor( random.sample( available_uids.tolist(), self.config.neuron.training_topk ), dtype = torch.int64 )
            return uids 
        
        question = None

        if random_sample_uids:
            uids = _get_random_uids()

        while question is None:
            question, reward_diff = _get_question(uids, bootstrap_prompt, reset_bootstrap_prompt)
            reset_bootstrap_prompt = True
            uids = _get_random_uids()

        return question, reward_diff
    
    def train( self ):
        """ Training 
            The function uses an infinite loop to repeatedly generate a random question, 
            ask the network to complete the question, and train the gating network using 
            the question and the resulting completions.
        """
        # Store the current epoch block number for comparison later.
        last_epoch_block = self.subtensor.block
        steps = 0
        
        # grab the question from the current sample
        prompt = next(self.dataset)['context']
        self.base_prompt = self.config.neuron.base_prompt
        reward_diff = 0
        self.last_sync = self.subtensor.block
        
        # Start an infinite loop for training.
        try:
            while True:
                # Ask the network to complete the random question, training the gating network.
                with open('prompt_history.txt', 'a') as file:
                    file.write(f"{steps} | Q score({round(reward_diff , 4)}): {prompt}" + '\n')
                
                forward_result = self.forward( 
                    roles = ['system', 'user' ],
                    messages = [ self.base_prompt, prompt ],
                    topk = self.config.neuron.training_topk,
                    random_sample_uids = True,
                    train_gating_model = True,
                    timeout = self.config.neuron.inference_timeout,
                    question = False
                )
                
                with open('prompt_history.txt', 'a') as file:
                    file.write(f"{steps} | A score({round(forward_result.rewards.sort(descending = True)[0][0].item(), 4)}): {forward_result.best_completion}" + '\n')

                if forward_result is not None:
                    idx_reward_sorted = forward_result.rewards.sort(descending = True)[1]
                    prompt, reward_diff = self.get_question(
                        uids = forward_result.uids[idx_reward_sorted],
                        bootstrap_prompt = forward_result.best_completion, 
                        reset_bootstrap_prompt = (steps % self.config.neuron.reset_bootstrap_prompt_frequency == 0),
                        random_sample_uids = self.config.neuron.question_random_sample_uids
                    )

                # Resync metagraph before returning. (sync every 15 min or ~75 blocks)
                if self.subtensor.block - self.last_sync > 100:
                    self.metagraph.sync()
                    self.last_sync = self.subtensor.block
                    self.save()
                    delegates = self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )

                    # Recreate pools here to ensure sizing is correct.
                    self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
                    self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )

                    self.my_nominators = { nomin[0]: nomin[1] for nomin in delegates[0][0].nominators } if len(delegates) else {}
                    self.check_weights()

                    if self.metagraph.n > self.gating_model.num_uids:
                        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to( self.device )

                # Check if enough epoch blocks have elapsed since the last epoch.
                epoch_length = self.subtensor.validator_epoch_length(self.config.netuid) if self.config.neuron.epoch_length_override == -1 else self.config.neuron.epoch_length_override
                blocks_until_epoch = epoch_length - ( self.subtensor.block - last_epoch_block )
                bittensor.logging.debug( 'blocks_until_epoch', blocks_until_epoch )
                if blocks_until_epoch <= 0: 
                    bittensor.logging.trace( 'epoch()' )
                    bittensor.logging.info( 'block', self.subtensor.block )

                    # Update the last epoch block to the current epoch block.
                    last_epoch_block = self.subtensor.block
                    
                    # Computes the average reward for each uid across non-zero values 
                    # using the rewards history stored in the self.history list.
                    uids, weights = self.compute_weights()
                    bittensor.logging.info( 'weights', weights )

                    # Set the weights on chain via our subtensor connection.
                    self.subtensor.set_weights(
                        wallet = self.wallet,
                        netuid = self.config.netuid,
                        uids = uids,
                        weights = weights,
                        wait_for_finalization = False,
                    )
                steps += 1 

        except Exception as e:
            bittensor.logging.info( 'Error in training loop', str( e    ) )
            print(traceback.format_exc())
    
    def compute_weights( self ) -> Tuple[ torch.LongTensor, torch.FloatTensor ]:
        """
            Computes the average reward for each uid across non-zero values 
            using the rewards history stored in the self.history list.

            Returns:
                uids ( torch.LongTensor, shape = (n) ): 
                    Uid to set weights on.
                weights ( torch.FloatTensor, shape = (n) ): 
                    The weights for each uid.
        """
        bittensor.logging.info( 'compute_weights()' )

        # Return zeros weights if there is no history.
        if self.history.qsize() == 0: 
            bittensor.logging.warning( 'No history to compute weights returning all ones.' )
            return torch.ones((self.metagraph.n)) / self.metagraph.n

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize( self.moving_averaged_scores, p=1, dim=0 )
        bittensor.logging.trace( 'raw_weights', raw_weights )
        bittensor.logging.trace( 'top10 values', raw_weights.sort()[0] )
        bittensor.logging.trace( 'top10 uids', raw_weights.sort()[1] )
     
        # Process the raw weights to final_weights via subtensor limitations.
        processed_weight_uids, processed_weights = bittensor.utils.weight_utils.process_weights_for_netuid(
            uids = self.metagraph.uids.to( "cpu" ),
            weights = raw_weights.to( "cpu" ),
            netuid = self.config.netuid,
            subtensor = self.subtensor,
            metagraph = self.metagraph
        )
        bittensor.logging.trace( 'processed_weights', processed_weights )
        bittensor.logging.trace( 'processed_weight_uids', processed_weight_uids )
        return processed_weight_uids, processed_weights

    def run(self):
        if self.config.neuron.inference_only:
            # Start an infinite loop, allows axon to service inference requests.
            last_sync = self.subtensor.block
            while True:
                time.sleep(12)
                if self.subtensor.block -last_sync > 100:
                    self.metagraph.sync()
                    self.last_sync = self.subtensor.block
                    self.load(inference_only = True)

        else:
            # Normal validator train operation for validation.
            self.train()

    def save(self, path=None):
        r""" Save hotkeys and moving average scores to filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path
            state_dict = {
                'neuron_weights': self.moving_averaged_scores,
                'neuron_hotkeys': self.hotkeys
            }

            torch.save(state_dict, f'{path}/model.torch')
            bittensor.logging.success(prefix='Saved model', sufix=f'<blue>{path}/model.torch</blue>')

            gating_state_dict = {
                'model_state_dict':self.gating_model.state_dict(),
                'num_hotkeys': self.gating_model.num_uids
            }
            torch.save(gating_state_dict, f'{path}/gating.torch')
            bittensor.logging.success(prefix='Saved gating model', sufix=f'<blue>{path}/gating.torch</blue>')
        except Exception as e:
            logger.warning(f'Failed to save model with error: {e}')

    def load(self, path=None, inference_only=False):
        r""" Load hotkeys and moving average scores from filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path
            state_dict = torch.load(f'{path}/model.torch')
            self.moving_averaged_scores = state_dict['neuron_weights'].clone().detach()
            self.hotkeys = state_dict['neuron_hotkeys']
            bittensor.logging.success(prefix='Reloaded model', sufix=f'<blue>{path}/model.torch</blue>')

            gating_state_dict = torch.load(f'{path}/gating.torch')
            if self.gating_model.num_uids == gating_state_dict['num_hotkeys']:
                self.gating_model.load_state_dict(gating_state_dict['model_state_dict'], strict=False)
                bittensor.logging.success(prefix='Reloaded Gating model', sufix=f'<blue>{path}/gating.torch</blue>')

            elif inference_only:
                self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config, num_uids=gating_state_dict['num_hotkeys']).to( self.device )
                self.gating_model.load_state_dict(gating_state_dict['model_state_dict'], strict=False)
                bittensor.logging.success(prefix='Reloaded Gating model', sufix=f'<blue>{path}/gating.torch</blue>')

        except Exception as e:
            logger.warning(f'Failed to load model with error: {e}')

    def check_weights(self):
        """ Checks current hotkeys with the current version of the metagraph """
        for uid, hotkey in enumerate( self.hotkeys ):
            if hotkey != self.metagraph.hotkeys[ uid ]:
                self.moving_averaged_scores[ uid ] = 0 #hotkey has been replaced
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            new_moving_average  = torch.zeros((self.metagraph.n)).to( self.device )
            new_moving_average[:len(self.hotkeys)] = self.moving_averaged_scores
            self.moving_averaged_scores = new_moving_average
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)


if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().run()
