import bittensor

def test_text_dataset():
    batch_size = 20
    block_size = 128
    epoch_length = 10
    dataset = bittensor.dataset (
        batch_size = batch_size, 
        block_size = block_size
    )
    dataset = dataset.dataset(epoch_length)

    assert len(dataset) == epoch_length
    assert len(dataset) != len(dataset)
    assert len(dataset[0]) == block_size
    assert len(dataset.dataset) == batch_size * epoch_length