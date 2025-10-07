#!/usr/bin/env python3
"""
实验四：性能测试演示（简化版）

本实验演示负载测试、压力测试和性能分析功能。
"""

import time
import random
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import concurrent.futures

# 测试类型
class TestType(Enum):
    LOAD_TEST = "load_test"        # 负载测试
    STRESS_TEST = "stress_test"    # 压力测试
    SPIKE_TEST = "spike_test"      # 峰值测试
    VOLUME_TEST = "volume_test"    # 容量测试

# 测试状态
class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TestResult:
    """测试结果"""
    request_id: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0

@dataclass
class TestConfig:
    """测试配置"""
    test_type: TestType
    concurrent_users: int
    requests_per_user: int
    ramp_up_time: float  # 启动时间（秒）
    test_duration: float  # 测试持续时间（秒）
    think_time: float = 0.0  # 用户思考时间（秒）

class SimplePerformanceTester:
    """简化的性能测试器"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.active_users = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = 0
        self.end_time = 0
        self._lock = threading.RLock()
        self.status = TestStatus.PENDING
    
    def run_test(self, target_func: Callable, config: TestConfig, test_data: List[Any] = None):
        """运行性能测试"""
        print(f"🚀 开始 {config.test_type.value} 测试")
        print(f"   并发用户: {config.concurrent_users}")
        print(f"   每用户请求数: {config.requests_per_user}")
        print(f"   启动时间: {config.ramp_up_time}s")
        print(f"   测试持续时间: {config.test_duration}s")
        print("\n" + "="*50)
        
        self.status = TestStatus.RUNNING
        self.start_time = time.time()
        
        try:
            # 准备测试数据
            if test_data is None:
                test_data = [f"test_query_{i}" for i in range(config.requests_per_user * config.concurrent_users)]
            
            # 使用线程池执行并发测试
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
                futures = []
                
                for user_id in range(config.concurrent_users):
                    # 计算用户启动延迟
                    start_delay = (config.ramp_up_time / config.concurrent_users) * user_id
                    
                    future = executor.submit(
                        self._simulate_user,
                        user_id,
                        target_func,
                        config,
                        test_data[user_id * config.requests_per_user:(user_id + 1) * config.requests_per_user],
                        start_delay
                    )
                    futures.append(future)
                
                # 等待所有用户完成
                concurrent.futures.wait(futures, timeout=config.test_duration + config.ramp_up_time + 30)
            
            self.end_time = time.time()
            self.status = TestStatus.COMPLETED
            
        except Exception as e:
            self.end_time = time.time()
            self.status = TestStatus.FAILED
            print(f"❌ 测试失败: {str(e)}")
            raise e
    
    def _simulate_user(self, user_id: int, target_func: Callable, config: TestConfig, 
                      user_data: List[Any], start_delay: float):
        """模拟单个用户行为"""
        # 等待启动延迟
        time.sleep(start_delay)
        
        with self._lock:
            self.active_users += 1
        
        print(f"👤 用户 {user_id} 开始测试 (延迟 {start_delay:.2f}s)")
        
        try:
            for i, data in enumerate(user_data):
                request_id = f"user_{user_id}_req_{i}"
                
                # 执行请求
                result = self._execute_request(request_id, target_func, data)
                
                with self._lock:
                    self.results.append(result)
                    self.total_requests += 1
                    if result.success:
                        self.successful_requests += 1
                    else:
                        self.failed_requests += 1
                
                # 用户思考时间
                if config.think_time > 0:
                    time.sleep(config.think_time)
                
                # 检查测试是否应该结束
                if time.time() - self.start_time > config.test_duration:
                    break
        
        finally:
            with self._lock:
                self.active_users -= 1
            print(f"👤 用户 {user_id} 完成测试")
    
    def _execute_request(self, request_id: str, target_func: Callable, data: Any) -> TestResult:
        """执行单个请求"""
        start_time = time.time()
        
        try:
            response = target_func(data)
            end_time = time.time()
            
            return TestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=True,
                response_size=len(str(response)) if response else 0
            )
        
        except Exception as e:
            end_time = time.time()
            
            return TestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=False,
                error_message=str(e)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        if not self.results:
            return {}
        
        # 成功请求的响应时间
        successful_durations = [r.duration for r in self.results if r.success]
        all_durations = [r.duration for r in self.results]
        
        # 计算吞吐量
        test_duration = self.end_time - self.start_time if self.end_time > self.start_time else 1
        throughput = len(self.results) / test_duration
        
        stats = {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'test_duration': test_duration,
            'throughput': throughput,
            'concurrent_users_peak': max([self.active_users, len(set(r.request_id.split('_')[1] for r in self.results))])
        }
        
        if successful_durations:
            stats.update({
                'avg_response_time': statistics.mean(successful_durations),
                'min_response_time': min(successful_durations),
                'max_response_time': max(successful_durations),
                'median_response_time': statistics.median(successful_durations),
                'p95_response_time': self._percentile(successful_durations, 95),
                'p99_response_time': self._percentile(successful_durations, 99)
            })
        
        if all_durations:
            stats['avg_response_time_all'] = statistics.mean(all_durations)
        
        return stats
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def get_error_summary(self) -> Dict[str, int]:
        """获取错误汇总"""
        error_counts = defaultdict(int)
        
        for result in self.results:
            if not result.success and result.error_message:
                error_counts[result.error_message] += 1
        
        return dict(error_counts)
    
    def reset(self):
        """重置测试器"""
        with self._lock:
            self.results.clear()
            self.active_users = 0
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.start_time = 0
            self.end_time = 0
            self.status = TestStatus.PENDING

# 模拟目标服务
class MockRAGService:
    """模拟RAG服务（用于性能测试）"""
    
    def __init__(self):
        self.request_count = 0
        self.overload_threshold = 50  # 过载阈值
        self._lock = threading.RLock()
    
    def process_query(self, query: str) -> str:
        """处理查询"""
        with self._lock:
            self.request_count += 1
            current_load = self.request_count
        
        # 模拟处理时间（随负载增加）
        base_time = 0.1
        load_factor = min(current_load / self.overload_threshold, 3.0)
        processing_time = base_time * (1 + load_factor) + random.uniform(0, 0.1)
        
        time.sleep(processing_time)
        
        # 模拟在高负载下的错误
        if current_load > self.overload_threshold:
            error_rate = min((current_load - self.overload_threshold) / self.overload_threshold * 0.3, 0.5)
            if random.random() < error_rate:
                raise Exception(f"Service overloaded (load: {current_load})")
        
        # 模拟随机错误
        if random.random() < 0.05:  # 5% 随机错误率
            raise Exception("Random service error")
        
        with self._lock:
            self.request_count = max(0, self.request_count - 1)
        
        return f"Processed: {query} (load: {current_load})"
    
    def reset_load(self):
        """重置负载计数"""
        with self._lock:
            self.request_count = 0

def run_performance_demo():
    """运行性能测试演示"""
    print("=" * 60)
    print("🏃 性能测试演示实验")
    print("=" * 60)
    
    # 创建服务和测试器
    rag_service = MockRAGService()
    tester = SimplePerformanceTester()
    
    # 测试场景
    test_scenarios = [
        {
            'name': '负载测试',
            'config': TestConfig(
                test_type=TestType.LOAD_TEST,
                concurrent_users=5,
                requests_per_user=4,
                ramp_up_time=2.0,
                test_duration=10.0,
                think_time=0.5
            )
        },
        {
            'name': '压力测试',
            'config': TestConfig(
                test_type=TestType.STRESS_TEST,
                concurrent_users=10,
                requests_per_user=3,
                ramp_up_time=1.0,
                test_duration=8.0,
                think_time=0.2
            )
        }
    ]
    
    # 执行测试场景
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 场景 {i}: {scenario['name']}")
        print("-" * 40)
        
        # 重置服务和测试器
        rag_service.reset_load()
        tester.reset()
        
        # 准备测试数据
        test_queries = [
            "机器学习基础", "深度学习原理", "自然语言处理", "计算机视觉",
            "推荐系统", "强化学习", "神经网络", "数据挖掘",
            "模式识别", "知识图谱", "语音识别", "图像分类"
        ]
        
        try:
            # 运行测试
            tester.run_test(
                target_func=rag_service.process_query,
                config=scenario['config'],
                test_data=test_queries * 10  # 确保有足够的测试数据
            )
            
            # 显示结果
            print(f"\n📊 {scenario['name']} 结果:")
            stats = tester.get_statistics()
            
            print(f"  总请求数: {stats.get('total_requests', 0)}")
            print(f"  成功请求: {stats.get('successful_requests', 0)}")
            print(f"  失败请求: {stats.get('failed_requests', 0)}")
            print(f"  成功率: {stats.get('success_rate', 0):.1f}%")
            print(f"  测试时长: {stats.get('test_duration', 0):.2f}s")
            print(f"  吞吐量: {stats.get('throughput', 0):.2f} req/s")
            
            if 'avg_response_time' in stats:
                print(f"\n  响应时间统计:")
                print(f"    平均: {stats['avg_response_time']:.3f}s")
                print(f"    最小: {stats['min_response_time']:.3f}s")
                print(f"    最大: {stats['max_response_time']:.3f}s")
                print(f"    中位数: {stats['median_response_time']:.3f}s")
                print(f"    95分位: {stats['p95_response_time']:.3f}s")
                print(f"    99分位: {stats['p99_response_time']:.3f}s")
            
            # 显示错误汇总
            errors = tester.get_error_summary()
            if errors:
                print(f"\n  错误汇总:")
                for error_msg, count in errors.items():
                    print(f"    {error_msg}: {count}次")
        
        except Exception as e:
            print(f"❌ 测试执行失败: {str(e)}")
        
        print("\n" + "="*50)
    
    print("\n✅ 性能测试演示完成！")
    
    # 性能建议
    print("\n💡 性能优化建议:")
    print("  1. 监控响应时间，保持在可接受范围内")
    print("  2. 设置合理的并发限制，避免服务过载")
    print("  3. 实施熔断器模式，快速失败保护系统")
    print("  4. 使用缓存减少重复计算")
    print("  5. 优化数据库查询和网络调用")
    print("  6. 定期进行性能测试，及时发现瓶颈")

if __name__ == "__main__":
    run_performance_demo()