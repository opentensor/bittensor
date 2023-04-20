import bittensor as bt


llm = bt.prompting(wallet_name="prompt")



print(llm("what is bittensor?").completion)