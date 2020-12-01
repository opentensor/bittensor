import replicate
import torch
from loguru import logger

class ReplicateUtility():
    def __init__(self, config):
        """

        Args:
            config (Bittensor.Config (Munch)): Bittensor config object
        """
        self.config = config
        self.experiment = replicate.init(
            path=self.config.neuron.datapath,
            params={
                **vars(config.neuron), 
                **vars(config.synapse), 
                **vars(config.axon), 
                **vars(config.dendrite),
                **vars(config.metagraph),
                **vars(config.session)
            }
        )
    
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
