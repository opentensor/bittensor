# The MIT License (MIT)
# Copyright © 2023 ChainDude

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Optional, List, Dict
import bittensor as bt
import pydantic

"""
Represents a software version with major, minor, and patch components.
"""
class Version (pydantic.BaseModel):
    major_version: Optional[int] = None
    minor_version: Optional[int] = None
    patch_version: Optional[int] = None

"""
Extends the Bittensor Synapse with an additional version attribute, 
used for compatibility and version control in mapreduce operations.
"""
class MapSynapse ( bt.Synapse ):
    version: Optional[Version] = None

"""
Defines the structure of a mapreduce job, including information about network configuration, 
the participating miners, and the job's runtime status.
"""
class Job (pydantic.BaseModel):
    master_hotkey: Optional[str] = None
    client_hotkey: Optional[str] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    world_size: Optional[int] = None
    rank: Optional[int] = None
    peer_count: Optional[int] = None
    miners: Optional[List] = []
    verifier_count: Optional[int] = None
    bandwidth: Optional[int] = None
    started_at: Optional[int] = None
    session_time: Optional[int] = 900
    status: Optional[str] = None
    reason: Optional[str] = None

"""
A specialized Synapse representing the status of a miner, 
including its availability and memory resources.
"""
class MinerStatus( MapSynapse ):
    free_memory: Optional[int] = None
    available: Optional[bool] = None
    # perf_input: Optional[str] = None
    # pert_output: Optional[str] = None

"""
Defines the status of a validator, particularly whether it is available for processing requests.
"""
class ValidatorStatus( pydantic.BaseModel ):
    available: Optional[bool] = None
    
"""
Represents a synapse message for joining a mapreduce job, 
including the job details and ranks of participating nodes.
"""
class Join( MapSynapse ):
    ranks: Optional[Dict] = None
    job: Optional[Job] = None
    joining: Optional[bool] = None
    reason: Optional[str] = None

"""
Synapse message used when a node needs to connect to the master node for a job.
"""
class ConnectMaster( MapSynapse ):
    job: Optional[Job] = None

"""
Synapse message for requesting a benchmarking operation, 
specifying the miner UID and job details.
"""
class RequestBenchmark( MapSynapse ):
    miner_uid: Optional[int] = None
    job: Optional[Job] = None

"""
Synapse message encapsulating the results of a benchmarking operation.
"""
class BenchmarkResults( MapSynapse ):
    results: Optional[List] = None
    bots: Optional[List] = None
    
"""
Defines the structure for the results of a benchmarking operation, 
including metrics like bandwidth, speed, and duration.
"""
class BenchmarkResult( pydantic.BaseModel ):
    bandwidth: Optional[int] = None
    speed: Optional[int] = None
    duration: Optional[int] = None
    data_length: Optional[int] = None
    free_memory: Optional[int] = None
    upload: Optional[int] = None
    download: Optional[int] = None

"""
Speed test
"""
class SpeedTest( MapSynapse ):
    result: Optional[Dict] = None