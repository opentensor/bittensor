from bittensor.dataloaders.text_dataloader import GenesisTextDataloader

def test_text_dataloader():
    batch_size = 20
    block_size = 128
    epoch_length = 10
    dataset = GenesisTextDataloader(batch_size, block_size)
    dataloader = dataset.dataloader(epoch_length)

    assert len(dataloader) == epoch_length
    assert len(dataloader) != len(dataset)
    assert len(dataset[0]) == block_size
    assert len(dataloader.dataset) == batch_size * epoch_length