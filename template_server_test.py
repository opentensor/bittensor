import bittensor
import torch
bittensor.logging(debug=True)
import torch.nn.functional as F
import cProfile, pstats, io
from pstats import SortKey

wallet = bittensor.wallet(name='test',hotkey='test')

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
model_config = AutoConfig.from_pretrained('gpt2')
model = AutoModel.from_pretrained('gpt2')

import cProfile, pstats, io
from pstats import SortKey

def forward_hidden_state(inputs_x, synapse):
        print('forward_hidden_state')
        output = model(inputs_x.to(model.device)) # .hidden_states[-1]
        print('forward_hidden_state end')

        # padding_r = (1024-output.size(2))
        # encoded_hidden = F.pad(output, (0, padding_r),  "constant", 0)
        return torch.rand(10, inputs_x.size(1), 1024)# encoded_hidden


pr = cProfile.Profile()
pr.enable()

forward_hidden_state(torch.rand(10, 64).to(torch.long), None)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(.1)
print(s.getvalue())