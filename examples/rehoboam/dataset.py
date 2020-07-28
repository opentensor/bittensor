import torch
import torchtext
from torchtext.data.utils import get_tokenizer

class Dataset():

    def __init__(self, bptt: int):
        self.TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                    init_token='<sos>',
                                    eos_token='<eos>',
                                    lower=True)
        self.train_txt, self.val_txt, self.test_txt = torchtext.datasets.WikiText2.splits(self.TEXT)
        self.TEXT.build_vocab(self.train_txt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bptt = bptt

    def batchify(self, data, bsz):
        data = self.TEXT.numericalize([data.examples[0].text])
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

       
