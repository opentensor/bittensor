import replicate
import torch
from loguru import logger

class ReplicateUtility():
    def __init__(self, config):
        """

        Args:
            config (Bittensor.Config (Munch)): Bittensor config object
        """
        default_neuron_config = vars(config.neuron)
        neuron_config = {}
        for key in default_neuron_config.keys():
            neuron_config["neuron.{}".format(key)] = default_neuron_config[key]

        self.config = config
        self.experiment = replicate.init(
            path=self.config.neuron.datapath,
            params={
                **self.append_class_prefix(config.neuron, "neuron"),
                **self.append_class_prefix(config.synapse, "synapse"), 
                **self.append_class_prefix(config.axon, "axon"), 
                **self.append_class_prefix(config.dendrite, "dendrite"),
                **self.append_class_prefix(config.metagraph, "metagraph")
            }
        )
    
    def append_class_prefix(self, config, prefix):
        """Used primarily to set up parameters of replicate.ai experiments. Forces a dot notation 
           of the params since replicate.ai ignores it. 
           
           For example, batch_size_train becomes neuron.batch_size_train since batch_size_train is a neuron config.

        Args:
            config (Bittensor.Config): Neuron, Synapse, Axon, Dendrite, etc. config objects.
            prefix (str): word to prefix to the key.

        Returns:
            dict : same config, but with keys prefixed with dot notation.
        """
        default_config = vars(config)
        prefixed_config = {}
        for key in default_config.keys():
            prefixed_config["{}.{}".format(prefix, key)] = default_config[key]
        
        return prefixed_config

    def checkout_experiment(self, model, best=True):
        """ Checks out best (or latest) experiment using Replicate API and returns it for training.

        Args:
            model (Torch.nn.Module): Torch model to be loaded
            best (bool, optional): Define whether to check out the best 
                                    performing version of the model, or the latest version. Defaults to True.

        Raises:
            Exception: Generic exception if something fails when checking out.

        Returns:
            model (Torch.nn.Module): Torch model to be loaded
        """
        try:
            experiment = replicate.experiments.get(self.config.session.checkout_experiment)
            
            latest_experiment = experiment.best()
            if not best:
                latest_experiment = experiment.latest()
            
            logger.info("Checking out experiment {} to {}".format(
                self.config.session.checkout_experiment, 
                self.config.neuron.datapath + self.config.neuron.neuron_name))
            
            model_file = latest_experiment.open(self.config.neuron.datapath + self.config.neuron.neuron_name + "/model.torch")
            checkpt = torch.load(model_file)
            model.load_state_dict(checkpt['model'])
        except Exception as e:
            raise Exception

        return model

    def checkpoint_experiment(self, epoch, **experiment_metrics):
        """ Creates a checkpoint for the current experiment object

        Args:
            epoch (integer): Current epoch at which we are checkpointing the experiment.
        """
        # Create a checkpoint within the experiment.
        # This saves the metrics at that point, and makes a copy of the file
        # or directory given, which could weights and any other artifacts.
        self.experiment.checkpoint(
            path=self.config.neuron.datapath + self.config.neuron.neuron_name + "/model.torch",
            step=epoch,
            metrics=experiment_metrics,
            primary_metric=("loss", "minimize"),
        )    
