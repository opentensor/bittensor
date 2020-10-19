class DPNConfig:
    r"""
    This is the configuration class to store the configuration of a :class:`~DPNSynapse`.
    It is used to instantiate a Dual Path model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a 
    "shallow" DPN-26 configuration. 

    For deeper network configurations, it is possible to set the num_blocks parameter to (3, 4, 20, 3) for a
    DPN-92. 
    
    For DPN-98 set the following:
    in_planes: (160, 320, 640, 1280)
    out_planes: (256, 512, 1024, 2048)
    num_blocks: (3, 6, 20, 3)
    dense_depth: (16, 32, 32, 128)


    Args:
        in_planes (:obj:`tuple`, `required`, defaults to (96,192,384,768)):
            The inputs of convolutional layers 2, 3, 4, and 5. 
        out_planes (:obj:`tuple`, `required`, defaults to (256,512,1024,2048)):
            Output planes of convolutional layers 2, 3, 4, and 5.
        num_blocks (:obj:`tuple`, `required`, defaults to (2,2,2,2)):
            How many blocks of layers to create for layers 2, 3, 4, and 5
        dense_depth (:obj:`tuple`, `required`, defaults to (16,32,24,128):
            Width increment of the densely connected path.
      

    Examples::

        >>> from bittensor.synapses.dpn.dpn_configuration import DPNConfig

        >>> # Initializing a DPN configuration
        >>> configuration = DPNConfig()

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    block_config = None
    
    def __init__(
        self,
        in_planes=(96,192,384,768),
        out_planes=(256,512,1024,2048),
        block_config=(2,2,2,2),
        dense_depth=(16,32,24,128)
    ):

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.block_config = block_config
        self.dense_depth = dense_depth

    def DPN(self):
        cfg = {
            'in_planes': self.in_planes,
            'out_planes': self.out_planes,
            'num_blocks': self.block_config,
            'dense_depth': self.dense_depth,
        }
        return cfg