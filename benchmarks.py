#!/usr/bin/env python3
"""
Comprehensive benchmark script for routing logic performance improvements.
Compares old vs new RoundRobinRouter implementations across multiple scenarios.
"""

import time
import statistics
import gc
import tracemalloc
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random
import string
import threading
import concurrent.futures
import json
import os
import sys

# Mock classes to simulate the actual environment
@dataclass
class EndpointInfo:
    url: str
    model_names: List[str] = None
    model_label: str = None
    
    def __hash__(self):
        return hash(self.url)

class MockRequest:
    def __init__(self, headers: Dict[str, str] = None):
        self.headers = headers or {}

class MockEngineStats:
    def __init__(self, load: float = 0.0):
        self.load = load

class MockRequestStats:
    def __init__(self, qps: float = 0.0):
        self.qps = qps

# Old implementation (original)
class OldRoundRobinRouter:
    def __init__(self):
        self.req_id = 0

    def route_request(self, endpoints: List[EndpointInfo], engine_stats, request_stats, request):
        """Original implementation with O(n log n) sorting on every request"""
        len_engines = len(endpoints)
        chosen = sorted(endpoints, key=lambda e: e.url)[self.req_id % len_engines]
        self.req_id += 1
        return chosen.url

# New implementation (optimized)
class NewRoundRobinRouter:
    def __init__(self):
        self.req_id = 0
        self.sorted_endpoints = []
        self.last_endpoints_id = None
        self.last_endpoints_hash = None

    def route_request(self, endpoints: List[EndpointInfo], engine_stats, request_stats, request):
        """Optimized implementation with O(1) amortized complexity"""
        # Fast path: O(1) - check if same list object
        endpoints_id = id(endpoints)
        if endpoints_id != self.last_endpoints_id:
            current_hash = hash(tuple(e.url for e in endpoints))
            if current_hash != self.last_endpoints_hash:
                self.sorted_endpoints = sorted(endpoints, key=lambda e: e.url)
                self.last_endpoints_hash = current_hash
            self.last_endpoints_id = endpoints_id
        
        chosen = self.sorted_endpoints[self.req_id % len(self.sorted_endpoints)]
        self.req_id += 1
        return chosen.url

def generate_endpoints(count: int) -> List[EndpointInfo]:
    """Generate mock endpoints for testing"""
    endpoints = []
    for i in range(count):
        url = f"http://endpoint-{i:03d}.example.com:8000"
        endpoints.append(EndpointInfo(url=url, model_names=[f"model-{i}"]))
    return endpoints

def generate_random_endpoints(count: int) -> List[EndpointInfo]:
    """Generate endpoints with random URLs to test hash changes"""
    endpoints = []
    for i in range(count):
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
        url = f"http://endpoint-{random_suffix}.example.com:8000"
        endpoints.append(EndpointInfo(url=url, model_names=[f"model-{i}"]))
    return endpoints

def benchmark_router(router_class, endpoints_list: List[List[EndpointInfo]], requests_per_endpoint_set: int):
    """Benchmark a router implementation"""
    router = router_class()
    request = MockRequest()
    engine_stats = {}
    request_stats = {}
    
    times = []
    
    for endpoints in endpoints_list:
        for _ in range(requests_per_endpoint_set):
            start_time = time.perf_counter()
            router.route_request(endpoints, engine_stats, request_stats, request)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    return times

def run_benchmark():
    """Run comprehensive benchmarks"""
    print("🚀 Routing Logic Performance Benchmark")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {"name": "Small (5 endpoints)", "endpoint_count": 5, "requests": 1000},
        {"name": "Medium (20 endpoints)", "endpoint_count": 20, "requests": 1000},
        {"name": "Large (100 endpoints)", "endpoint_count": 100, "requests": 1000},
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 40)
        
        endpoint_count = scenario["endpoint_count"]
        requests = scenario["requests"]
        
        # Test 1: Same endpoint list (typical case)
        print(f"Test 1: Same endpoint list ({requests} requests)")
        endpoints = generate_endpoints(endpoint_count)
        endpoints_list = [endpoints] * (requests // 100)  # Reuse same list
        
        # Benchmark old implementation
        old_times = benchmark_router(OldRoundRobinRouter, endpoints_list, 100)
        old_avg = statistics.mean(old_times) * 1_000_000  # Convert to microseconds
        old_std = statistics.stdev(old_times) * 1_000_000
        
        # Benchmark new implementation
        new_times = benchmark_router(NewRoundRobinRouter, endpoints_list, 100)
        new_avg = statistics.mean(new_times) * 1_000_000
        new_std = statistics.stdev(new_times) * 1_000_000
        
        improvement = old_avg / new_avg if new_avg > 0 else float('inf')
        
        print(f"  Old implementation: {old_avg:.2f} ± {old_std:.2f} μs")
        print(f"  New implementation: {new_avg:.2f} ± {new_std:.2f} μs")
        print(f"  Improvement: {improvement:.1f}x faster")
        
        # Test 2: Changing endpoint lists (stress test)
        print(f"Test 2: Changing endpoint lists (hash changes)")
        endpoints_list = [generate_random_endpoints(endpoint_count) for _ in range(requests // 100)]
        
        old_times_changing = benchmark_router(OldRoundRobinRouter, endpoints_list, 100)
        old_avg_changing = statistics.mean(old_times_changing) * 1_000_000
        old_std_changing = statistics.stdev(old_times_changing) * 1_000_000
        
        new_times_changing = benchmark_router(NewRoundRobinRouter, endpoints_list, 100)
        new_avg_changing = statistics.mean(new_times_changing) * 1_000_000
        new_std_changing = statistics.stdev(new_times_changing) * 1_000_000
        
        improvement_changing = old_avg_changing / new_avg_changing if new_avg_changing > 0 else float('inf')
        
        print(f"  Old implementation: {old_avg_changing:.2f} ± {old_std_changing:.2f} μs")
        print(f"  New implementation: {new_avg_changing:.2f} ± {new_std_changing:.2f} μs")
        print(f"  Improvement: {improvement_changing:.1f}x faster")

def run_throughput_test():
    """Test throughput under sustained load"""
    print(f"\n🔥 Throughput Test (10,000 requests)")
    print("-" * 40)
    
    endpoints = generate_endpoints(20)  # 20 endpoints
    request = MockRequest()
    engine_stats = {}
    request_stats = {}
    
    # Old implementation throughput
    old_router = OldRoundRobinRouter()
    start_time = time.perf_counter()
    for _ in range(10000):
        old_router.route_request(endpoints, engine_stats, request_stats, request)
    old_duration = time.perf_counter() - start_time
    old_rps = 10000 / old_duration
    
    # New implementation throughput
    new_router = NewRoundRobinRouter()
    start_time = time.perf_counter()
    for _ in range(10000):
        new_router.route_request(endpoints, engine_stats, request_stats, request)
    new_duration = time.perf_counter() - start_time
    new_rps = 10000 / new_duration
    
    print(f"Old implementation: {old_rps:,.0f} requests/second")
    print(f"New implementation: {new_rps:,.0f} requests/second")
    print(f"Throughput improvement: {new_rps/old_rps:.1f}x")

def benchmark_memory_usage(router_class, endpoints: List[EndpointInfo], num_requests: int) -> Tuple[float, float]:
    """Benchmark actual memory usage of router implementations"""
    router = router_class()
    request = MockRequest()
    engine_stats = {}
    request_stats = {}
    
    # Start memory tracking
    tracemalloc.start()
    gc.collect()  # Clean up before measurement
    
    # Capture baseline memory
    snapshot_before = tracemalloc.take_snapshot()
    
    # Run routing requests
    for _ in range(num_requests):
        router.route_request(endpoints, engine_stats, request_stats, request)
    
    # Capture memory after routing
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # Calculate memory difference
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    
    return total_memory, total_memory / num_requests if num_requests > 0 else 0

def run_memory_test():
    """Test actual memory efficiency"""
    print(f"\n💾 Memory Efficiency Analysis")
    print("-" * 40)
    
    endpoints = generate_endpoints(50)
    num_requests = 1000
    
    # Test old implementation
    print("Testing old implementation memory usage...")
    old_total, old_per_request = benchmark_memory_usage(OldRoundRobinRouter, endpoints, num_requests)
    
    # Test new implementation  
    print("Testing new implementation memory usage...")
    new_total, new_per_request = benchmark_memory_usage(NewRoundRobinRouter, endpoints, num_requests)
    
    print(f"\nMemory Usage Results ({num_requests} requests):")
    print(f"Old implementation: {old_total:,.0f} bytes total, {old_per_request:.1f} bytes/request")
    print(f"New implementation: {new_total:,.0f} bytes total, {new_per_request:.1f} bytes/request")
    
    if new_per_request > 0:
        memory_improvement = old_per_request / new_per_request
        print(f"Memory efficiency: {memory_improvement:.1f}x less memory per request")
    else:
        print("Memory efficiency: Significantly improved (near-zero allocation)")

def run_concurrent_test():
    """Test performance under concurrent load"""
    print(f"\n🔀 Concurrent Load Test")
    print("-" * 40)
    
    endpoints = generate_endpoints(20)
    num_threads = 10
    requests_per_thread = 1000
    
    def worker_old():
        router = OldRoundRobinRouter()
        request = MockRequest()
        engine_stats = {}
        request_stats = {}
        
        start_time = time.perf_counter()
        for _ in range(requests_per_thread):
            router.route_request(endpoints, engine_stats, request_stats, request)
        return time.perf_counter() - start_time
    
    def worker_new():
        router = NewRoundRobinRouter()
        request = MockRequest()
        engine_stats = {}
        request_stats = {}
        
        start_time = time.perf_counter()
        for _ in range(requests_per_thread):
            router.route_request(endpoints, engine_stats, request_stats, request)
        return time.perf_counter() - start_time
    
    # Test old implementation
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        start_time = time.perf_counter()
        old_futures = [executor.submit(worker_old) for _ in range(num_threads)]
        old_times = [f.result() for f in concurrent.futures.as_completed(old_futures)]
        old_total_time = time.perf_counter() - start_time
    
    # Test new implementation
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        start_time = time.perf_counter()
        new_futures = [executor.submit(worker_new) for _ in range(num_threads)]
        new_times = [f.result() for f in concurrent.futures.as_completed(new_futures)]
        new_total_time = time.perf_counter() - start_time
    
    total_requests = num_threads * requests_per_thread
    old_rps = total_requests / old_total_time
    new_rps = total_requests / new_total_time
    
    print(f"Concurrent test ({num_threads} threads, {requests_per_thread} requests each):")
    print(f"Old implementation: {old_rps:,.0f} requests/second")
    print(f"New implementation: {new_rps:,.0f} requests/second")
    print(f"Concurrent improvement: {new_rps/old_rps:.1f}x")

def run_edge_case_tests():
    """Test edge cases and corner scenarios"""
    print(f"\n🎯 Edge Case Tests")
    print("-" * 40)
    
    # Test 1: Single endpoint
    print("Test 1: Single endpoint")
    single_endpoint = generate_endpoints(1)
    old_router = OldRoundRobinRouter()
    new_router = NewRoundRobinRouter()
    
    start_time = time.perf_counter()
    for _ in range(1000):
        old_router.route_request(single_endpoint, {}, {}, MockRequest())
    old_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    for _ in range(1000):
        new_router.route_request(single_endpoint, {}, {}, MockRequest())
    new_time = time.perf_counter() - start_time
    
    print(f"  Old: {old_time*1000:.2f}ms, New: {new_time*1000:.2f}ms, Improvement: {old_time/new_time:.1f}x")
    
    # Test 2: Frequent endpoint changes
    print("Test 2: Frequent endpoint changes (worst case)")
    new_router_changing = NewRoundRobinRouter()
    old_router_changing = OldRoundRobinRouter()
    
    start_time = time.perf_counter()
    for i in range(1000):
        endpoints = generate_random_endpoints(10)  # New endpoints every time
        old_router_changing.route_request(endpoints, {}, {}, MockRequest())
    old_changing_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    for i in range(1000):
        endpoints = generate_random_endpoints(10)  # New endpoints every time
        new_router_changing.route_request(endpoints, {}, {}, MockRequest())
    new_changing_time = time.perf_counter() - start_time
    
    print(f"  Old: {old_changing_time*1000:.2f}ms, New: {new_changing_time*1000:.2f}ms, Improvement: {old_changing_time/new_changing_time:.1f}x")
    
    # Test 3: Large number of endpoints
    print("Test 3: Large endpoint count (1000 endpoints)")
    large_endpoints = generate_endpoints(1000)
    old_router_large = OldRoundRobinRouter()
    new_router_large = NewRoundRobinRouter()
    
    start_time = time.perf_counter()
    for _ in range(100):
        old_router_large.route_request(large_endpoints, {}, {}, MockRequest())
    old_large_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    for _ in range(100):
        new_router_large.route_request(large_endpoints, {}, {}, MockRequest())
    new_large_time = time.perf_counter() - start_time
    
    print(f"  Old: {old_large_time*1000:.2f}ms, New: {new_large_time*1000:.2f}ms, Improvement: {old_large_time/new_large_time:.1f}x")

def save_benchmark_results(results: Dict):
    """Save benchmark results to JSON file"""
    timestamp = int(time.time())
    filename = f"benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: {filename}")

def generate_performance_report():
    """Generate a comprehensive performance report"""
    print(f"\n📈 Performance Analysis Summary")
    print("=" * 50)
    
    # Quick benchmark for report
    endpoints_20 = generate_endpoints(20)
    
    # Measure typical case performance
    old_router = OldRoundRobinRouter()
    new_router = NewRoundRobinRouter()
    
    # Time 1000 requests with same endpoint list
    start = time.perf_counter()
    for _ in range(1000):
        old_router.route_request(endpoints_20, {}, {}, MockRequest())
    old_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(1000):
        new_router.route_request(endpoints_20, {}, {}, MockRequest())
    new_time = time.perf_counter() - start
    
    improvement = old_time / new_time
    old_rps = 1000 / old_time
    new_rps = 1000 / new_time
    
    print(f"\nPerformance Metrics (20 endpoints, 1000 requests):")
    print(f"┌─────────────────────┬─────────────┬─────────────┬─────────────┐")
    print(f"│ Implementation      │ Time (ms)   │ RPS         │ Improvement │")
    print(f"├─────────────────────┼─────────────┼─────────────┼─────────────┤")
    print(f"│ Old (O(n log n))    │ {old_time*1000:8.2f}    │ {old_rps:8.0f}    │ 1.0x        │")
    print(f"│ New (O(1) amort.)   │ {new_time*1000:8.2f}    │ {new_rps:8.0f}    │ {improvement:8.1f}x      │")
    print(f"└─────────────────────┴─────────────┴─────────────┴─────────────┘")
    
    # Calculate theoretical improvements for different scales
    print(f"\nTheoretical Scaling Analysis:")
    print(f"┌─────────────┬─────────────┬─────────────┬─────────────┐")
    print(f"│ Endpoints   │ Old Complexity │ New Complexity │ Expected Improv │")
    print(f"├─────────────┼─────────────┼─────────────┼─────────────┤")
    for n in [5, 20, 100, 1000]:
        old_complexity = n * math.log2(n)  # n log n per request
        new_complexity = 1  # O(1) amortized
        theoretical_improvement = old_complexity / new_complexity
        print(f"│ {n:8d}    │ {old_complexity:8.1f}       │ {new_complexity:8.1f}       │ {theoretical_improvement:8.1f}x      │")
    print(f"└─────────────┴─────────────┴─────────────┴─────────────┘")

if __name__ == "__main__":
    import math
    
    print("🚀 Comprehensive Router Performance Benchmark")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Run all benchmark tests
    run_benchmark()
    run_throughput_test()
    run_memory_test()
    run_concurrent_test()
    run_edge_case_tests()
    generate_performance_report()
