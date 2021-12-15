import bittensor

logging = bittensor.logging()

def test_construct_text_corpus():
    # text corpus for the train set
    dataset = bittensor.dataset(max_corpus_size = 10000, save_dataset = True)
    dataset.construct_text_corpus()
    dataset.close()

def test_next():
    dataset = bittensor.dataset(max_corpus_size = 1000)
    next(dataset)
    next(dataset)
    next(dataset)
    dataset.close()

if __name__ == "__main__":
    test_construct_text_corpus()