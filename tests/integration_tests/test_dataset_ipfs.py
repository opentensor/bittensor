import bittensor

def test_text_dataset():
    batch_size = 20
    block_size = 128
    num_batches = 10
    epoch_length = 10

    dataset = bittensor.dataset (
        batch_size = batch_size, 
        block_size = block_size,
        num_batches = num_batches
    )
    
    dataloader = dataset.dataloader(epoch_length)

    assert len(dataloader) == epoch_length
    assert len(dataloader) != len(dataset)
    assert len(dataset[0]) == block_size
    assert len(dataloader.dataset) == batch_size * epoch_length
    
    dataset.close()