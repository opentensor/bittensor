import bittensor

def test_construct_text_corpus():

    # text corpus for the train set
    dataset = bittensor.dataset(max_corpus_size = 10000, save_dataset = True)
    dataset.construct_text_corpus()


    # text corpus for the test set
    dataset.dataset_name = 'test'
    dataset.construct_text_corpus()

    # text corpus for the validation set
    dataset.dataset_name = 'validation'
    dataset.construct_text_corpus()

def test_next():
    dataset = bittensor.dataset(max_corpus_size = 1000)
    next(dataset)
    next(dataset)
    next(dataset)
