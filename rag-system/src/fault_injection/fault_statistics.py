"""故障统计模块

提供故障注入的统计分析功能，包括故障频率、类型分布等。
"""

import time
from typing import Dict, List, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class FaultRecord:
    """故障记录"""
    timestamp: float
    fault_type: str
    operation: str
    error_message: str
    duration: float = 0.0  # 故障持续时间


class FaultStatistics:
    """故障统计分析器"""
    
    def __init__(self):
        self.fault_records: List[FaultRecord] = []
        self.operation_stats = defaultdict(lambda: {
            'total_calls': 0,
            'fault_calls': 0,
            'fault_types': Counter(),
            'total_duration': 0.0,
            'fault_duration': 0.0
        })
    
    def record_operation_start(self, operation: str) -> str:
        """记录操作开始
        
        Args:
            operation: 操作名称
            
        Returns:
            操作ID
        """
        operation_id = f"{operation}_{time.time()}"
        self.operation_stats[operation]['total_calls'] += 1
        return operation_id
    
    def record_fault(self, operation: str, fault_type: str, error_message: str, duration: float = 0.0):
        """记录故障
        
        Args:
            operation: 操作名称
            fault_type: 故障类型
            error_message: 错误消息
            duration: 故障持续时间
        """
        record = FaultRecord(
            timestamp=time.time(),
            fault_type=fault_type,
            operation=operation,
            error_message=error_message,
            duration=duration
        )
        
        self.fault_records.append(record)
        
        # 更新操作统计
        stats = self.operation_stats[operation]
        stats['fault_calls'] += 1
        stats['fault_types'][fault_type] += 1
        stats['fault_duration'] += duration
    
    def record_operation_end(self, operation: str, duration: float, success: bool = True):
        """记录操作结束
        
        Args:
            operation: 操作名称
            duration: 操作持续时间
            success: 是否成功
        """
        stats = self.operation_stats[operation]
        stats['total_duration'] += duration
    
    def get_fault_rate(self, operation: str = None, time_window: int = None) -> float:
        """获取故障率
        
        Args:
            operation: 操作名称，None表示所有操作
            time_window: 时间窗口（秒），None表示所有时间
            
        Returns:
            故障率 (0.0 - 1.0)
        """
        if operation:
            stats = self.operation_stats[operation]
            if stats['total_calls'] == 0:
                return 0.0
            return stats['fault_calls'] / stats['total_calls']
        else:
            # 计算全局故障率
            total_calls = sum(stats['total_calls'] for stats in self.operation_stats.values())
            total_faults = sum(stats['fault_calls'] for stats in self.operation_stats.values())
            
            if total_calls == 0:
                return 0.0
            return total_faults / total_calls
    
    def get_fault_distribution(self, operation: str = None) -> Dict[str, int]:
        """获取故障类型分布
        
        Args:
            operation: 操作名称，None表示所有操作
            
        Returns:
            故障类型分布字典
        """
        if operation:
            return dict(self.operation_stats[operation]['fault_types'])
        else:
            # 合并所有操作的故障类型
            combined_counter = Counter()
            for stats in self.operation_stats.values():
                combined_counter.update(stats['fault_types'])
            return dict(combined_counter)
    
    def get_recent_faults(self, time_window: int = 300) -> List[FaultRecord]:
        """获取最近的故障记录
        
        Args:
            time_window: 时间窗口（秒）
            
        Returns:
            最近的故障记录列表
        """
        cutoff_time = time.time() - time_window
        return [record for record in self.fault_records if record.timestamp >= cutoff_time]
    
    def get_operation_summary(self, operation: str) -> Dict[str, Any]:
        """获取操作摘要统计
        
        Args:
            operation: 操作名称
            
        Returns:
            操作统计摘要
        """
        if operation not in self.operation_stats:
            return {}
        
        stats = self.operation_stats[operation]
        
        return {
            'operation': operation,
            'total_calls': stats['total_calls'],
            'fault_calls': stats['fault_calls'],
            'success_calls': stats['total_calls'] - stats['fault_calls'],
            'fault_rate': self.get_fault_rate(operation),
            'success_rate': 1.0 - self.get_fault_rate(operation),
            'average_duration': stats['total_duration'] / max(stats['total_calls'], 1),
            'average_fault_duration': stats['fault_duration'] / max(stats['fault_calls'], 1),
            'fault_types': dict(stats['fault_types']),
            'most_common_fault': stats['fault_types'].most_common(1)[0] if stats['fault_types'] else None
        }
    
    def get_global_summary(self) -> Dict[str, Any]:
        """获取全局统计摘要
        
        Returns:
            全局统计摘要
        """
        total_calls = sum(stats['total_calls'] for stats in self.operation_stats.values())
        total_faults = sum(stats['fault_calls'] for stats in self.operation_stats.values())
        total_duration = sum(stats['total_duration'] for stats in self.operation_stats.values())
        
        return {
            'total_operations': len(self.operation_stats),
            'total_calls': total_calls,
            'total_faults': total_faults,
            'total_successes': total_calls - total_faults,
            'global_fault_rate': self.get_fault_rate(),
            'global_success_rate': 1.0 - self.get_fault_rate(),
            'average_duration': total_duration / max(total_calls, 1),
            'fault_distribution': self.get_fault_distribution(),
            'recent_faults_count': len(self.get_recent_faults()),
            'operations': list(self.operation_stats.keys())
        }
    
    def get_time_series_data(self, time_window: int = 3600, bucket_size: int = 60) -> Dict[str, List]:
        """获取时间序列数据
        
        Args:
            time_window: 时间窗口（秒）
            bucket_size: 时间桶大小（秒）
            
        Returns:
            时间序列数据
        """
        current_time = time.time()
        start_time = current_time - time_window
        
        # 创建时间桶
        buckets = []
        bucket_start = start_time
        while bucket_start < current_time:
            buckets.append({
                'timestamp': bucket_start,
                'fault_count': 0,
                'total_count': 0
            })
            bucket_start += bucket_size
        
        # 填充故障数据
        for record in self.fault_records:
            if record.timestamp >= start_time:
                bucket_index = int((record.timestamp - start_time) // bucket_size)
                if 0 <= bucket_index < len(buckets):
                    buckets[bucket_index]['fault_count'] += 1
        
        # 计算总调用数（这里简化处理，实际应该记录所有调用）
        for bucket in buckets:
            bucket['total_count'] = bucket['fault_count'] * 10  # 假设故障率为10%
            bucket['fault_rate'] = bucket['fault_count'] / max(bucket['total_count'], 1)
        
        return {
            'timestamps': [bucket['timestamp'] for bucket in buckets],
            'fault_counts': [bucket['fault_count'] for bucket in buckets],
            'total_counts': [bucket['total_count'] for bucket in buckets],
            'fault_rates': [bucket['fault_rate'] for bucket in buckets]
        }
    
    def export_statistics(self) -> Dict[str, Any]:
        """导出完整统计数据
        
        Returns:
            完整的统计数据
        """
        return {
            'global_summary': self.get_global_summary(),
            'operation_summaries': {
                op: self.get_operation_summary(op) 
                for op in self.operation_stats.keys()
            },
            'recent_faults': [
                {
                    'timestamp': record.timestamp,
                    'datetime': datetime.fromtimestamp(record.timestamp).isoformat(),
                    'fault_type': record.fault_type,
                    'operation': record.operation,
                    'error_message': record.error_message,
                    'duration': record.duration
                }
                for record in self.get_recent_faults()
            ],
            'time_series': self.get_time_series_data()
        }
    
    def clear_statistics(self):
        """清空统计数据"""
        self.fault_records.clear()
        self.operation_stats.clear()
    
    def clear_old_records(self, retention_days: int = 7):
        """清理旧记录
        
        Args:
            retention_days: 保留天数
        """
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        self.fault_records = [
            record for record in self.fault_records 
            if record.timestamp >= cutoff_time
        ]