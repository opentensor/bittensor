


class Neuron:

    def __init__( config:Munch, **kwargs ):
        if config == None:
            config = Neuron.default_config( );       
        bittensor.Config.update_with_kwargs(config.miner, kwargs) 
        Neuron.check_config(config)
        self.config = config

        # ---- Wallet ----
        self.wallet = bittensor.Wallet( self.config )
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )
        
        # ---- Bittensor components ----
        self.axon = bittensor.Axon( config = self.config, wallet = self.wallet )
        self.dendrite = bittensor.Dendrite( config = self.config, wallet = self.wallet )
        self.subtensor = bittensor.Subtensor( config = self.config, wallet = self.wallet )
        self.metagraph = bittensor.Metagraph( subtensor = self.subtensor )

        # ---- Forward and Backward serving threads ----
        self.quit_forward = mp.Event()
        self.forward_thread = threading.Thread( target = self.forward_loop, name = 'forward', daemon=True )
        self.quit_backward = mp.Event()
        self.backward_thread = threading.Thread( target = self.backward_loop, name = 'backward', daemon=True )

    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod
    def check_config(config: Munch):
        full_path = '{}/{}/{}'.format(config.neuron.root_dir, config.neuron.name, config.neuron.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.clip_gradients', default=0.8, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        parser.add_argument('--miner.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
        parser.add_argument('--miner.log_interval', default=10, type=int, help='Batches before we log miner info.')
        parser.add_argument('--miner.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
        parser.add_argument('--miner.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str,  help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.name', default='bert-nsp', type=str, help='Trials for this miner go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.uid')
        parser.add_argument('--miner.record_log', default=False, help='Record all logs when running this miner')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        BertMLMNucleus.add_args(parser)
        bittensor.Axon.add_args(parser)
        bittensor.Dendrite.add_args(parser)
        bittensor.Subtensor.add_args(parser)

    # ---- Starts miner ----
    def start(self):   
        # --- Connect to the chain. ---
        if not self.subtensor.connect():
            raise RuntimeError('Failed to connect miner to network: {}'.format(self.subtensor.config.network))

        # --- Subscribe our endpoint ----
        if not self.subtensor.subscribe(
                self.config.axon.external_ip, 
                self.config.axon.external_port,
                bittensor.proto.Modality.TEXT,
                self.wallet.coldkeypub,
                wait_for_finalization = True,
                timeout = 4 * bittensor.__blocktime__):
            raise RuntimeError('Failed to subscribe miner on network: {}'.format(self.subtensor.config.network))

        # --- Start the forward endpoint ----
        self.axon.start()

        # --- Start the forward/backward threads ----
        self.forward_thread.start()
        self.backwar_thread.start()

        # --- Sync metagraph  ----
        self.metagraph.sync()
        self.weights = torch.rand([self.metagraph.n()])

        # --- Runs the miner main loop.
        self.run()

    # --- Main loop ----
    def run (self):

        # --- Init running state. ---
        self.global_step = 0
        self.best_train_loss = math.inf

        # --- Loop for epochs. ---
        for self.epoch in range(self.config.miner.n_epochs):
            try:
                # ---- Train Model. ----
                self.train()
                self.scheduler.step()
                
                # ---- Catch NaNs. ----
                # If model has borked for some reason, we need to make sure it doesn't emit weights
                # Instead, reload into previous version of model
                if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                    self.model, self.optimizer = self.model_toolbox.load_model( self.config )
                    self.model.router.set_dendrite( self.dendrite )
                    self.model.router.set_metagraph( self.metagraph )
                    continue

                # ---- Set weights on chain. ----
                self.subtensor.set_weights (
                    uids = self.metagraph.uids(),
                    weights = self.weights,
                    wait_for_inclusion = False
                )

                # ---- Sync metagraph. ----
                self.metagraph.sync() # Pulls latest chain info.
                self.weights = torch.nn.functional.pad(self.weights, pad = [0, self.metagraph.n() - self.weights.numel() ]) # Pads weights to the correct size.

                # --- Epoch logs. ----
                print(self.axon)
                print(self.dendrite)
                print(self.metagraph)

                # ---- Update Tensorboard. ----
                self.axon.__to_tensorboard__(self.tensorboard, self.global_step)
                self.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
                self.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
            
                # ---- Save best loss and model. ----
                if self.training_loss and self.epoch % 10 == 0 and self.training_loss < self.best_train_loss:
                    self.best_train_loss = self.training_loss / 10 # update best train loss
                    self.model_toolbox.save_model(
                        self.config.miner.full_path,
                        {
                            'epoch': self.epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'loss': self.best_train_loss,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    )
                    self.tensorboard.add_scalar('Miner/Train_loss', self.training_loss, self.global_step)
                

            # --- Catch Errors ----
            except Exception as e:
                logger.error('Exception in training script with error: {}, {}', e, traceback.format_exc())
                logger.info('Continuing to train.')

    # ---- Train Epoch ----
    def train(self):
        self.training_loss = 0.0
        for local_step in range(self.config.miner.epoch_length):
            # ---- Forward pass ----
            inputs, targets = mlm_batch(self.dataset, self.config.miner.batch_size_train, self.tokenizer, self.data_collator)
            output = self.model.remote_forward (
                    bittensor.neuron,
                    inputs = inputs.to(self.model.device), 
                    targets = targets.to(self.model.device)
            )

            # ---- Backward pass ----
            loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            loss.backward() # Accumulates gradients on the model.
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation

            # ---- Train row weights ----
            batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
            self.weights = (1 - 0.03) * self.weights + 0.03 * batch_weights # Moving avg update.
            self.weights = F.normalize( self.weights, p = 1, dim = 0 ) # Ensure normalization.

            # ---- Step logs ----
            logger.info('GS: {} LS: {} Epoch: {}\tLocal Target Loss: {}\tRemote Target Loss: {}\tDistillation Loss: {}\tAxon: {}\tDendrite: {}',
                    colored('{}'.format(self.global_step), 'red'),
                    colored('{}'.format(local_step), 'blue'),
                    colored('{}'.format(self.epoch), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.remote_target_loss.item()), 'blue'),
                    colored('{:.4f}'.format(output.distillation_loss.item()), 'red'),
                    self.axon,
                    self.dendrite)
            logger.info('Codes: {}', output.router.return_codes.tolist())
            
            self.tensorboard.add_scalar('Miner/Rloss', output.remote_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Miner/Lloss', output.local_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Miner/Dloss', output.distillation_loss.item(), self.global_step)

            # ---- Step increments ----
            self.global_step += 1
            self.training_loss += output.local_target_loss.item()

            # --- Memory clean up ----
            torch.cuda.empty_cache()
            del output

    # ---- Forward loop -----
    def forward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.info('Serving thread started: ')
        while not self.quit_forward.is_set():
            try:

                # ---- Pull request ----
                logger.info('Axon:{}, waiting for forward query ... ', self.axon)
                pong, pubkey, inputs, modality = self.axon.next_forward_item( timeout = 10.0 )

                # ---- Process request ----
                if None not in [ pong, pubkey, inputs, modality]:
                    logger.info('Recieved Query: from:{}, inputs.shape:{}', pubkey, inputs.shape)
                    outputs = self.model.local_forward( inputs, training = False ).local_hidden
                    pong.send( outputs.detach() )
                    logger.info('Sent response: to:{}, outputs.shape:{}', pubkey, outputs.shape)

            except Exception as e:
                logger.exception('Error in forward thread with error {}', e)
                continue
            
    # ---- Backward loop -----
    def backward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.info('Backward thread started: ')
        while not self.quit_forward.is_set():
            try:

                # ---- Pull request ----
                logger.info('Axon:{}, waiting for backward query ... ', self.axon)
                pong, pubkey, inputs_x, grads_dy, modality = self.axon.next_backward_item( timeout = 10.0 )
                # TODO(anyone): apply gradients from peers to maximize profit.
                pong.send( None )

            except Exception as e:
                logger.exception('Error in backward thread with error {}', e)
                continue

    def __del__(self):
        if self.forward_thread != None:
            self.quit_forward.set()
            self.forward_thread.join( timeout = 12.0 )
            if self.forward_thread.is_alive():
                logger.error('Failed to join forward thread.')

        if self.backward_thread != None:
            self.quit_backward.set()
            self.backward_thread.join( timeout = 12.0 )
            if self.backward_thread.is_alive():
                logger.error('Failed to join backward thread.')


    
