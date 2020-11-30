import replicate
import torch
from loguru import logger

class ReplicateUtility():
    def __init__(self, config):
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

    
