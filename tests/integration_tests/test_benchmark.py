from benchmarks.template_miner import Benchmark as Benchmark_template_miner
from benchmarks.advanced_server import Benchmark as  Benchmark_advanced_server
from benchmarks.multitron_server import Benchmark as Benchmark_multitron_server

def test_template_miner():
    benchmark = Benchmark_template_miner()
    benchmark.run()

def test_advanced_server():
    benchmark = Benchmark_advanced_server()
    benchmark.run()
