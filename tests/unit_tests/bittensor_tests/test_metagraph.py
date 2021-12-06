import bittensor
import torch
import unittest

class TestMetagraph(unittest.TestCase):

    def setUp (self):
        self.metagraph = bittensor.metagraph(network = 'nobunaga')
        assert True

    def test_print_empty(self):
        print (self.metagraph)

    def test_forward(self):
        row = torch.ones( (self.metagraph.n), dtype = torch.float32 )
        for i in range( self.metagraph.n ):
            self.metagraph(i, row)
        self.metagraph.sync()
        row = torch.ones( (self.metagraph.n), dtype = torch.float32 )
        for i in range( self.metagraph.n ):
            self.metagraph(i, row)

    def test_load_sync_save(self):
        self.metagraph.sync()
        self.metagraph.save()
        self.metagraph.load()
        self.metagraph.save()

    def test_factory(self):
        graph = self.metagraph.load().sync().save()

    def test_state_dict(self):
        self.metagraph.load()
        state = self.metagraph.state_dict()
        assert 'uids' in state
        assert 'stake' in state
        assert 'last_update' in state
        assert 'block' in state
        assert 'tau' in state
        assert 'weights' in state
        assert 'endpoints' in state









   
