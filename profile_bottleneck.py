import bittensor
import time
import asyncio
import threading
import psutil
import os

class MySynapse(bittensor.Synapse):
    input: int = 1
    output: int = None

# Simulate a blocking blocking sync function
def forward(synapse: MySynapse) -> MySynapse:
    time.sleep(2) # Blocks for 2 seconds
    synapse.output = synapse.input + 1
    return synapse

async def monitor_loop():
    """Monitor loop responsiveness."""
    print("Monitor: Starting")
    try:
        while True:
            start = time.time()
            await asyncio.sleep(0.1)
            diff = time.time() - start
            if diff > 0.2:
                print(f"Monitor: Loop blocked for {diff:.4f}s")
            # else:
            #    print(f"Monitor: Tick {diff:.4f}s")
    except asyncio.CancelledError:
        print("Monitor: Stopped")

async def main():
    print("Creating Axon...")
    wallet = bittensor.Wallet(name="mock_wallet")
    port = 8099
    
    axon = bittensor.Axon(wallet=wallet, port=port)
    axon.attach(forward_fn=forward)
    axon.start()
    
    # Start monitor
    monitor_task = asyncio.create_task(monitor_loop())
    
    print("Sending request...")
    dendrite = bittensor.Dendrite(wallet=wallet)
    synapse = MySynapse(input=10)
    
    req_start = time.time()
    # This should take ~2 seconds.
    # If blocking, monitor will print "Loop blocked for ~2.0s"
    # If non-blocking, monitor will continue ticking (no blockage msg)
    response = await dendrite(axon, synapse)
    req_time = time.time() - req_start
    
    with open("verification_result.txt", "w") as f:
        f.write(f"Request completed in {req_time:.4f} seconds. Response: {response.output}\n")
    
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    
    axon.stop()

if __name__ == "__main__":
    asyncio.run(main())
