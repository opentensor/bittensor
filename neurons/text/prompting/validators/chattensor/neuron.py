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
import queue
import torch
import argparse

import json

import bittensor as bt

from types import SimpleNamespace

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uvicorn import Config, Server
import asyncio
import uvicorn

from typing import List, Optional, Union
from reward import RewardModel
from gating import GatingModel


torch.autograd.set_detect_anomaly(True)


__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific about science and language.
'''
__default_base_prompt__ = '''
You are Chattensor.
Chattensor is a research project by Opentensor Cortex.
Chattensor is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Chattensor is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Chattensor is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Chattensor is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Chattensor is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Chattensor is here to assist.
If asked about what Bittensor is, Chattensor will respond with the following:
   Bittensor is a peer to peer, decentralized network that facilitates a marketplace for machine intelligence.
   Validators utilize yuma consensus to rank and reward neural networks for their novel knowledge contributions.
   Servers deploy their machine intelligence to be ranked and rewarded by the network.
   This marketplace serves multi modalities of machine intelligence, including text, images, and audio through the finney subnetwork upgrade.
If asked about who the creators of Chattensor are, Chattensor will respond with the following:
    - Carro
    - Prism
Lists should not be long, like so:
• first item
• second item
• third item
They are the creators of Chattensor, but not Bittensor. That was founded by Jacob Steeves (Const) and Ala Shaabana (Shibshib). 
The current maintainers of Bittensor is the Opentensor Foundation. Carro and Prism work at Opentensor.
'''

class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )
        bt.metagraph.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument('--netuid', type = int , help = 'Prompting network netuid', default = 41 )
        parser.add_argument('--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument('--neuron.base_prompt', type=str , help = 'Prompt injected before a question is completed by miners on the network', default = __default_base_prompt__ )
        parser.add_argument('--neuron.question_prompt', type=str , help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = __default_question_prompt__ )
        parser.add_argument('--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument('--neuron.backprop_on_inference', type = int, help = 'Maximum number history values to store at any time.', default = 100 )
        parser.add_argument('--neuron.inference_topk', type = str, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument('--neuron.training_topk', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument('--neuron.epoch_length', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument('--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 100 )
    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.metagraph.add_args( parser )
        bt.logging.add_args( parser )
        GatingModel.add_args( parser )
        return bt.config( parser )
    
    def __init__( self, config=None ):
        self.config = config if config is not None else neuron.config()
        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
        self.reward_model = RewardModel( self.config.neuron.reward_model_name ).to(self.device)
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to(self.device)
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )
        self.clients = {}


    async def websocket_endpoint( self, websocket: WebSocket ):
        r""" Websocket endpoint for the neuron. 
        """
        await websocket.accept()
        self.clients[websocket] = {"connection": websocket}
        try:
            while True:
                data = await websocket.receive_text()
                messages = json.loads(data)
                user_message = messages[-1]['content']
                response = await self.inference(user_message, return_loss=False)

                response = response.replace('\n', ' NEWLINE ')
                response = response.replace('\t', ' TAB ')

                for word in response.split():
                    if word == 'NEWLINE':
                        await websocket.send_text('\n')
                    elif word == 'TAB':
                        await websocket.send_text('\t')
                    else:
                        await websocket.send_text(word + ' ')
                    await asyncio.sleep(0.1)

                await websocket.send_text('<|endoftext|>')
            
        except WebSocketDisconnect:
            del self.clients[websocket]
        
        except KeyboardInterrupt:
            del self.clients[websocket]
            exit()

    def complute_weights( self ) -> torch.FloatTensor:
        """
            Computes the average reward for each uid across non-zero values 
            using the rewards history stored in the self.history list.

            Returns:
                weights ( torch.FloatTensor, shape = (n) ): 
                    The weights for each uid.
        """
        # Return zeros weights if there is no history.
        if self.history.qsize() == 0: 
            print ('no history to compute weights.'); return torch.zeros((self.metagraph.n))

        # Averages the rewards for each uid across non-zero values.
        rewards = []

        # Iterate over all events in the `history` list.
        for event in self.history.queue:
            # Normalize the rewards for the current event using softmax normalization.
            normalized_rewards = torch.nn.functional.softmax( event.rewards.to( self.device ), dim=0 )
            print ('normalized_rewards', normalized_rewards.size(), normalized_rewards)

            # Use the `uids` of the current event to scatter the normalized rewards
            # into a zero-initialized tensor with the same shape as `self.metagraph.n`.
            scattered_rewards = torch.zeros((self.metagraph.n)).to( self.device ).scatter(0, event.uids.to( self.device ), normalized_rewards.to( self.device ) )
            print ('scattered_rewards', scattered_rewards.size(), scattered_rewards)

            # Append the scattered rewards to the `rewards` list.
            rewards.append(scattered_rewards)

        # Stack the scattered rewards tensors along the second dimension.
        rewards = torch.stack( rewards, 1 ).to( self.device )
        print ('rewards', rewards.size(), rewards)

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        avg_rewards = torch.nan_to_num( rewards.sum(1) / (rewards != 0).sum(1), 0 )
        print ('avg_rewards', avg_rewards.size(), avg_rewards)

        # Return the calculated average rewards.
        return avg_rewards
   
    async def forward( 
            self, 
            message: str,
            topk: Optional[int] = None,
            return_loss: bool = True,
        ) -> SimpleNamespace:
        """ Inference is called by clients seeking the outputs of the model
            We use the gating network to determine the best models to query 
            Optionally we use the reward model to train the gating network.

            Args: 
                message (str): The message to query the network with.
        """
        print ('forward --------------------' )
        print ('message', message )

        # Set `topk` to the number of items in `self.metagraph.n` if `topk` is not provided or is -1.
        # Find the available `uids` that are currently serving.
        # If `topk` is larger than the number of available `uids`, set `topk` to the number of available `uids`.
        available_uids = torch.tensor( [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ], dtype = torch.int64 ).to( self.device )
        if topk is None or topk == -1: topk = self.metagraph.n.item()
        if topk > len(available_uids): topk = len(available_uids)
        print ('\ntopk', topk)
        if len( available_uids ) == 0: return print('no available uids'); None

        # We run the gating network here to get the best uids
        # Use the gating model to generate scores for each `uid`.
        scores = self.gating_model( message ).to( self.device )
        print ('\nscores', scores.size(), scores)

        # Select the top `topk` `uids` based on the highest `scores`.
        # Use the selected `uids` to query the dendrite pool.
        # Print the `completions`.
        topk_uids = available_uids[ scores[ available_uids ].sort()[ 1 ][ -topk: ]]
        completions = await self.dendrite_pool( prompt = self.config.neuron.base_prompt, message = message, uids = topk_uids )

        if completions is None:
            print("No completions found")
            print(completions)
            return None
        print ('\ntopk_uids',  len(topk_uids), topk_uids)
        print ('\ncompletions', len(completions), completions)

        # Filter out any `None` `completions`.
        successful_uids = torch.tensor( [ uid for uid, completion in list( zip( topk_uids, completions ) ) if completion is not None and len(completion) > 10 ], dtype = torch.int64 ).to( self.device )
        successful_completions = [ completion for completion in completions if completion is not None and len(completion) > 10 ]
        print ('\nsuccessful_uids', len(successful_uids), successful_uids)
        print ('\nsuccessful_completions', len(successful_completions), successful_completions)
        if len( successful_completions ) == 0: print ('no successful completions'); return None

        # Calculate the rewards for the successful `completions` using the reward model.
        # Print the rewards for all `uids`.
        rewards = self.reward_model.reward( successful_completions ).to( self.device )
        print ('\nrewards', rewards.size(), rewards)

        if return_loss:
            # Train the gating model using the scores and rewards of the successful `completions`.
            self.gating_model.backward( scores = scores[ successful_uids ], rewards = rewards )

        # Save the query history in a `result` object.
        # Return the `completion` with the highest reward.
        result = SimpleNamespace( 
            completion = successful_completions[ rewards.argmax( dim = 0 ) ],
            message = message,  
            uids = successful_uids,
            rewards = rewards,
            scores = scores,
            all_completions = completions
        )
        self.history.put( result )

        # Return the completion with the highest reward.
        print ('result', result )
        return result
    
    # User queries here.
    async def inference( self, message, return_loss=True ) -> str:
        """Inference"""
        result = await self.forward( message, topk = self.config.neuron.inference_topk, return_loss = return_loss )
        if result == None: return "Failed"
        else: return result.completion

    async def train(self):
        """Training
        The function uses an infinite loop to repeatedly generate a random question,
        ask the network to complete the question, and train the gating network using
        the question and the resulting completions.
        """
        # Store the current epoch block number for comparison later.
        last_epoch_block = self.subtensor.block
        sync_block = self.subtensor.block + 100

        async def training_loop():
            nonlocal sync_block
            # Start an infinite loop for training.
            while True:
                # Query the network for a random question.
                question = await self.forward(self.config.neuron.question_prompt)
                if question == None: continue  # no responses from network.

                # Ask the network to complete the random question, training the gating network.
                await self.forward(question.completion, topk=self.config.neuron.training_topk)

                # Check if enough epoch blocks have elapsed since the last epoch.
                if (sync_block - self.subtensor.block) == 0:  # run every block. # > self.subtensor.validator_epoch_length( self.config.netuid ) :

                    # Update the last epoch block to the current epoch block.
                    sync_block = self.subtensor.block + 100

                    # Computes the average reward for each uid across non-zero values
                    # using the rewards history stored in the self.history list.
                    weights = self.complute_weights()
                    print('\nweights', weights.size(), weights)

                    # Set the weights on chain via our subtensor connection.
                    self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        uids=self.metagraph.uids,
                        weights=weights,
                        wait_for_finalization=True,
                    )

                await asyncio.sleep(1)

        training_task = asyncio.create_task(training_loop())

        try:
            await training_task
        except asyncio.CancelledError:
            print("Training cancelled.")

            
# Set the default event loop policy
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

app = FastAPI()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await n.websocket_endpoint(websocket)

@app.post("/inference")
async def api_inference(message: str):
    return await n.inference(message)

async def start_training(n: neuron):
    await n.train()

if __name__ == '__main__':
    n = neuron()
    config = Config(app=app, host="0.0.0.0", port=8000)
    server = Server(config)

    async def main():
        server_task = asyncio.create_task(server.serve())
        training_task = asyncio.create_task(start_training(n))
        await asyncio.gather(server_task, training_task)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt, gracefully shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop)))
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()