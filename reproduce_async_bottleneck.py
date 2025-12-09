import bittensor
import time
import asyncio
import threading
import psutil
import os

class MySynapse(bittensor.Synapse):
    input: int = 1
    output: int = None

def forward(synapse: MySynapse) -> MySynapse:
    synapse.output = synapse.input + 1
    return synapse

async def main():
    with open("reproduce_result.txt", "w") as f:
        f.write("Creating Axon...\n")
        wallet = bittensor.Wallet(name="mock_wallet")
        port = 8099
        
        axon = bittensor.Axon(wallet=wallet, port=port)
        axon.attach(forward_fn=forward)
        
        f.write("Starting Axon...\n")
        start_time = time.time()
        axon.start()
        startup_time = time.time() - start_time
        f.write(f"Axon started in {startup_time:.4f} seconds\n")
        
        process = psutil.Process(os.getpid())
        f.write(f"Initial CPU Usage: {process.cpu_percent(interval=1.0)}%\n")
        
        f.write("Sending request...\n")
        dendrite = bittensor.Dendrite(wallet=wallet)
        synapse = MySynapse(input=10)
        
        req_start = time.time()
        response = await dendrite(axon, synapse)
        req_time = time.time() - req_start
        f.write(f"Request completed in {req_time:.4f} seconds. Response: {response.output}\n")
        
        f.write("Sleeping for 2 seconds to measure idle CPU usage...\n")
        cpu_usage = process.cpu_percent(interval=2.0)
        f.write(f"Idle CPU Usage: {cpu_usage}%\n")
        
        f.write("Stopping Axon...\n")
        stop_start = time.time()
        axon.stop()
        stop_time = time.time() - stop_start
        f.write(f"Axon stopped in {stop_time:.4f} seconds\n")

if __name__ == "__main__":
    asyncio.run(main())
