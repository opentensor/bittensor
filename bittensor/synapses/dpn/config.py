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
        target_size (:obj:`int`, `required`, defaults to (10)):
            The number of logit heads used by the target layer.      
        in_planes (:obj:`tuple`, `required`, defaults to (96,192,384,768)):
            The inputs of convolutional layers 2, 3, 4, and 5. 
        out_planes (:obj:`tuple`, `required`, defaults to (256,512,1024,2048)):
            Output planes of convolutional layers 2, 3, 4, and 5.
        num_blocks (:obj:`tuple`, `required`, defaults to (2,2,2,2)):
            How many blocks of layers to create for layers 2, 3, 4, and 5
        dense_depth (:obj:`tuple`, `required`, defaults to (16,32,24,128):
            Width increment of the densely connected path.
      

    Examples::

        >>> from bittensor.synapses.dpn.config import DPNConfig

        >>> # Initializing a DPN configuration
        >>> configuration = DPNConfig()

        >>> # Initializing a DNP Synapse
        >>> model = DPNSynapse ( configuration )
    """

    __default_target_size__ = 10
    __default_in_planes__ = (96,192,384,768)
    __default_out_planes__ = (256,512,1024,2048)
    __default_block_config__ = (2,2,2,2)
    __default_dense_depth__ = (16,32,24,128)
    
    def __init__(self, **kwargs):
        self.target_size = kwargs.pop("target_size",
                                         self.__default_target_size__)
        self.in_planes = kwargs.pop("in_planes",
                                         self.__default_in_planes__)
        self.out_planes = kwargs.pop("out_planes",
                                         self.__default_out_planes__)
        self.block_config = kwargs.pop("block_config",
                                         self.__default_block_config__)
        self.dense_depth = kwargs.pop("dense_depth",
                                         self.__default_dense_depth__)

        self.run_type_checks()
    
    def run_type_checks(self):
        assert isinstance(self.target_size, int)
        assert isinstance(self.in_planes, tuple)
        assert isinstance(self.out_planes, tuple)
        assert isinstance(self.block_config, tuple)
        assert isinstance(self.dense_depth, tuple)
    