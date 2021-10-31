import binascii
import struct
import hashlib
import math
import bittensor
import rich
import time

def hex_bytes_to_u8_list( hex_bytes: bytes ):
    hex_chunks = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
    return hex_chunks

def u8_list_to_hex( values: list ):
    total = 0
    for val in reversed(values):
        total = (total << 8) + val
    return total 

def create_seal_hash( block_hash:bytes, nonce:int ) -> bytes:
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
    block_bytes = block_hash.encode('utf-8')[2:]
    pre_seal = nonce_bytes + block_bytes
    seal = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
    return seal

def seal_meets_difficulty( seal:bytes, difficulty:int ):
    seal_number = int.from_bytes(seal, "big")
    product = seal_number * difficulty
    limit = int(math.pow(2,256) - 1)
    if product > limit:
        return False
    else:
        return True
    
def solve_for_difficulty( block_hash, difficulty ):
    meets = False
    nonce = -1
    while not meets:
        nonce += 1 
        seal = create_seal_hash( block_hash, nonce )
        meets = seal_meets_difficulty( seal, difficulty )
        if nonce > 1:
            break
    return nonce, seal


def solve_for_difficulty_fast( subtensor ):
    block_number = subtensor.get_current_block()
    difficulty = subtensor.difficulty
    block_hash = subtensor.substrate.get_block_hash( block_number )

    meets = False
    nonce = -1
    block_bytes = block_hash.encode('utf-8')[2:]
    limit = int(math.pow(2,256) - 1)
    best = math.inf
    update_interval = 100000
    start_time = time.time()

    console = bittensor.__console__
    with console.status("Solving") as status:
        while not meets:
            nonce += 1 

            # Create seal.
            nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
            pre_seal = nonce_bytes + block_bytes
            seal = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()

            seal_number = int.from_bytes(seal, "big")
            product = seal_number * difficulty
            if product - limit < best:
                best = product - limit
                best_seal = seal

            if product < limit:
                return nonce, block_number, block_hash, difficulty, seal

            if nonce % update_interval == 0:
                itrs_per_sec = update_interval / (time.time() - start_time)
                start_time = time.time()
                difficulty = subtensor.difficulty
                block_number = subtensor.get_current_block()
                block_hash = subtensor.substrate.get_block_hash( block_number)
                status.update("Solving\n  Nonce: [bold white]{}[/bold white]\n  Iters: [bold white]{}/s[/bold white]\n  Difficulty: [bold white]{}[/bold white]\n  Block: [bold white]{}[/bold white]\n  Best: [bold white]{}[/bold white]".format( nonce, int(itrs_per_sec), difficulty, block_hash.encode('utf-8'), binascii.hexlify(best_seal) ))
      

def create_pow( subtensor ):
    nonce, block_number, block_hash, difficulty, seal = solve_for_difficulty_fast( subtensor )
    return {
        'nonce': nonce, 
        'difficulty': difficulty,
        'block_number': block_number, 
        'block_hash': block_hash, 
        'work': binascii.hexlify(seal)
    }
