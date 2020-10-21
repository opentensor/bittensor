import torch
import time
from pathlib import Path

class ModelToolbox:
    """
        Utility class to load, save, and modify existing models. 
    """

    def __init__(self, datapath, dataset_name):
        # Log/data/model paths.
        self.trial_id = dataset_name + '-' + str(time.time()).split('.')[0]
        self.data_path = datapath + "datasets/"
        self.log_dir = datapath + self.trial_id + '/logs/'
        self.model_path = datapath + self.trial_id + '/model.torch'
        self.trial_path = datapath + self.trial_id

    def save_model(self, model, epoch, optimizer, test_loss):
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss
            }, self.model_path)

    def load_model(self, present_model, saved_model_path, optimizer):
        checkpoint = torch.load(saved_model_path)
        present_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        best_test_loss = checkpoint['test_loss']

        return present_model, optimizer, epoch, best_test_loss
    
    def setup_model_directory(self):
        # Create model directory for saving if it does not exist.
        Path(self.trial_path).mkdir(parents=True, exist_ok=True)
