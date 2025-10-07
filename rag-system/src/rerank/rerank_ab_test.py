#!/usr/bin/env python3
"""
重排序A/B测试模块

提供重排序功能的A/B测试框架，包括：
- 用户分组管理
- 实验配置
- 结果对比分析
- 统计显著性测试
"""

import logging
import hashlib
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from ..rag.rag_service import RAGService
from .rerank_service import RerankService
from .enhanced_rag_system import EnhancedRAGSystem

logger = logging.getLogger(__name__)

class ABTestGroup(Enum):
    """A/B测试分组"""
    CONTROL = "control"  # 对照组（标准RAG）
    TREATMENT = "treatment"  # 实验组（重排序RAG）

class RerankABTest:
    """重排序A/B测试类"""
    
    def __init__(
        self,
        rag_service: RAGService,
        test_name: str = "rerank_ab_test",
        treatment_ratio: float = 0.5,
        rerank_model: str = "all-MiniLM-L6-v2"
    ):
        """
        初始化A/B测试
        
        Args:
            rag_service: RAG服务实例
            test_name: 测试名称
            treatment_ratio: 实验组比例（0-1）
            rerank_model: 重排序模型名称
        """
        self.rag_service = rag_service
        self.test_name = test_name
        self.treatment_ratio = treatment_ratio
        
        # 初始化增强RAG系统
        self.enhanced_rag = EnhancedRAGSystem(
            rag_service=rag_service,
            use_cache=True,
            rerank_model=rerank_model
        )
        
        # 测试数据存储
        self.test_results = {
            ABTestGroup.CONTROL.value: [],
            ABTestGroup.TREATMENT.value: []
        }
        
        # 用户分组缓存
        self.user_groups = {}
        
        # 测试统计
        self.test_stats = {
            'total_queries': 0,
            'control_queries': 0,
            'treatment_queries': 0,
            'start_time': datetime.now().isoformat()
        }
    
    def get_user_group(self, user_id: str) -> ABTestGroup:
        """
        获取用户所属的测试组
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户所属的测试组
        """
        # 检查缓存
        if user_id in self.user_groups:
            return self.user_groups[user_id]
        
        # 基于用户ID的哈希值进行分组，确保同一用户始终在同一组
        hash_value = hashlib.md5(f"{self.test_name}_{user_id}".encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)
        
        # 根据哈希值和实验组比例分组
        if (hash_int % 100) / 100.0 < self.treatment_ratio:
            group = ABTestGroup.TREATMENT
        else:
            group = ABTestGroup.CONTROL
        
        # 缓存结果
        self.user_groups[user_id] = group
        
        logger.info(f"用户 {user_id} 分配到 {group.value} 组")
        return group
    
    def query_for_user(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 20,
        record_result: bool = True
    ) -> Dict[str, Any]:
        """
        为特定用户执行查询（根据A/B测试分组）
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回的文档数量
            retrieve_k: 重排序时初始检索的文档数量
            record_result: 是否记录测试结果
        
        Returns:
            查询结果字典
        """
        start_time = time.time()
        
        # 获取用户分组
        user_group = self.get_user_group(user_id)
        
        # 根据分组执行不同的查询策略
        if user_group == ABTestGroup.CONTROL:
            # 对照组：标准RAG查询
            result = self.enhanced_rag.query_standard(query, top_k)
            result['ab_test_group'] = ABTestGroup.CONTROL.value
            
            if record_result:
                self.test_stats['control_queries'] += 1
        else:
            # 实验组：重排序RAG查询
            result = self.enhanced_rag.query_with_rerank(query, top_k, retrieve_k)
            result['ab_test_group'] = ABTestGroup.TREATMENT.value
            
            if record_result:
                self.test_stats['treatment_queries'] += 1
        
        # 添加测试相关信息
        result.update({
            'user_id': user_id,
            'test_name': self.test_name,
            'query_timestamp': datetime.now().isoformat()
        })
        
        # 记录测试结果
        if record_result:
            self.test_results[user_group.value].append(result)
            self.test_stats['total_queries'] += 1
        
        query_time = time.time() - start_time
        logger.info(f"用户 {user_id} ({user_group.value}) 查询完成，耗时: {query_time:.3f}秒")
        
        return result
    
    def batch_test(
        self,
        test_queries: List[Dict[str, Any]],
        simulate_users: bool = True
    ) -> Dict[str, Any]:
        """
        批量A/B测试
        
        Args:
            test_queries: 测试查询列表，每个元素包含 {'user_id': str, 'query': str}
            simulate_users: 是否模拟用户ID（如果查询中没有user_id）
        
        Returns:
            批量测试结果
        """
        logger.info(f"开始批量A/B测试，共 {len(test_queries)} 个查询")
        
        batch_start_time = time.time()
        batch_results = []
        
        for i, query_data in enumerate(test_queries):
            # 获取或生成用户ID
            if 'user_id' in query_data:
                user_id = query_data['user_id']
            elif simulate_users:
                user_id = f"test_user_{i % 100}"  # 模拟100个用户
            else:
                user_id = f"anonymous_{i}"
            
            query = query_data['query']
            top_k = query_data.get('top_k', 5)
            retrieve_k = query_data.get('retrieve_k', 20)
            
            # 执行查询
            result = self.query_for_user(user_id, query, top_k, retrieve_k)
            batch_results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i + 1}/{len(test_queries)} 个查询")
        
        batch_time = time.time() - batch_start_time
        
        # 生成批量测试报告
        report = self.generate_test_report()
        report.update({
            'batch_info': {
                'total_queries': len(test_queries),
                'batch_time': batch_time,
                'avg_query_time': batch_time / len(test_queries) if test_queries else 0
            },
            'batch_results': batch_results
        })
        
        logger.info(f"批量A/B测试完成，总耗时: {batch_time:.2f}秒")
        return report
    
    def collect_user_feedback(
        self,
        user_id: str,
        query: str,
        feedback_score: float,
        feedback_text: Optional[str] = None
    ):
        """
        收集用户反馈
        
        Args:
            user_id: 用户ID
            query: 查询文本
            feedback_score: 反馈分数（1-5）
            feedback_text: 反馈文本
        """
        user_group = self.get_user_group(user_id)
        
        feedback_data = {
            'user_id': user_id,
            'query': query,
            'feedback_score': feedback_score,
            'feedback_text': feedback_text,
            'user_group': user_group.value,
            'timestamp': datetime.now().isoformat()
        }
        
        # 查找对应的测试结果并添加反馈
        group_results = self.test_results[user_group.value]
        for result in reversed(group_results):  # 从最新的开始查找
            if result.get('user_id') == user_id and result.get('query') == query:
                if 'feedback' not in result:
                    result['feedback'] = []
                result['feedback'].append(feedback_data)
                break
        
        logger.info(f"收到用户 {user_id} ({user_group.value}) 的反馈: {feedback_score}/5")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """
        生成A/B测试报告
        
        Returns:
            测试报告字典
        """
        control_results = self.test_results[ABTestGroup.CONTROL.value]
        treatment_results = self.test_results[ABTestGroup.TREATMENT.value]
        
        # 基础统计
        report = {
            'test_name': self.test_name,
            'test_stats': self.test_stats.copy(),
            'group_stats': {
                'control': {
                    'count': len(control_results),
                    'avg_query_time': self._calculate_avg_time(control_results),
                    'success_rate': self._calculate_success_rate(control_results)
                },
                'treatment': {
                    'count': len(treatment_results),
                    'avg_query_time': self._calculate_avg_time(treatment_results),
                    'success_rate': self._calculate_success_rate(treatment_results)
                }
            },
            'performance_comparison': {},
            'user_feedback_analysis': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # 性能对比分析
        if control_results and treatment_results:
            control_avg_time = report['group_stats']['control']['avg_query_time']
            treatment_avg_time = report['group_stats']['treatment']['avg_query_time']
            
            report['performance_comparison'] = {
                'time_difference': treatment_avg_time - control_avg_time,
                'time_improvement_percent': (
                    (control_avg_time - treatment_avg_time) / control_avg_time * 100
                    if control_avg_time > 0 else 0
                ),
                'statistical_significance': self._test_statistical_significance(
                    control_results, treatment_results
                )
            }
        
        # 用户反馈分析
        report['user_feedback_analysis'] = self._analyze_user_feedback()
        
        # 生成建议
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _calculate_avg_time(self, results: List[Dict[str, Any]]) -> float:
        """计算平均查询时间"""
        if not results:
            return 0.0
        
        times = [r.get('query_time', 0) for r in results if 'query_time' in r]
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """计算成功率"""
        if not results:
            return 0.0
        
        success_count = sum(1 for r in results if 'error' not in r)
        return success_count / len(results) * 100
    
    def _test_statistical_significance(
        self,
        control_results: List[Dict[str, Any]],
        treatment_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """测试统计显著性（简化版）"""
        # 这里实现一个简化的统计显著性测试
        # 在实际应用中，应该使用更严格的统计方法
        
        control_times = [r.get('query_time', 0) for r in control_results if 'query_time' in r]
        treatment_times = [r.get('query_time', 0) for r in treatment_results if 'query_time' in r]
        
        if len(control_times) < 10 or len(treatment_times) < 10:
            return {
                'is_significant': False,
                'reason': '样本量不足（每组至少需要10个样本）',
                'control_sample_size': len(control_times),
                'treatment_sample_size': len(treatment_times)
            }
        
        # 简单的均值差异检验
        control_mean = sum(control_times) / len(control_times)
        treatment_mean = sum(treatment_times) / len(treatment_times)
        
        # 计算标准差
        control_std = (sum((x - control_mean) ** 2 for x in control_times) / len(control_times)) ** 0.5
        treatment_std = (sum((x - treatment_mean) ** 2 for x in treatment_times) / len(treatment_times)) ** 0.5
        
        # 简化的t检验
        pooled_std = ((control_std ** 2 + treatment_std ** 2) / 2) ** 0.5
        t_stat = abs(treatment_mean - control_mean) / (pooled_std * (2 / min(len(control_times), len(treatment_times))) ** 0.5)
        
        # 简单的显著性判断（t > 2.0 认为显著）
        is_significant = t_stat > 2.0
        
        return {
            'is_significant': is_significant,
            't_statistic': t_stat,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'mean_difference': treatment_mean - control_mean,
            'control_sample_size': len(control_times),
            'treatment_sample_size': len(treatment_times)
        }
    
    def _analyze_user_feedback(self) -> Dict[str, Any]:
        """分析用户反馈"""
        control_feedback = []
        treatment_feedback = []
        
        # 收集反馈数据
        for result in self.test_results[ABTestGroup.CONTROL.value]:
            if 'feedback' in result:
                control_feedback.extend(result['feedback'])
        
        for result in self.test_results[ABTestGroup.TREATMENT.value]:
            if 'feedback' in result:
                treatment_feedback.extend(result['feedback'])
        
        analysis = {
            'control_feedback_count': len(control_feedback),
            'treatment_feedback_count': len(treatment_feedback),
            'control_avg_score': 0.0,
            'treatment_avg_score': 0.0,
            'feedback_difference': 0.0
        }
        
        # 计算平均分数
        if control_feedback:
            control_scores = [f['feedback_score'] for f in control_feedback]
            analysis['control_avg_score'] = sum(control_scores) / len(control_scores)
        
        if treatment_feedback:
            treatment_scores = [f['feedback_score'] for f in treatment_feedback]
            analysis['treatment_avg_score'] = sum(treatment_scores) / len(treatment_scores)
        
        analysis['feedback_difference'] = (
            analysis['treatment_avg_score'] - analysis['control_avg_score']
        )
        
        return analysis
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于性能对比的建议
        perf_comp = report.get('performance_comparison', {})
        if perf_comp.get('statistical_significance', {}).get('is_significant', False):
            time_improvement = perf_comp.get('time_improvement_percent', 0)
            if time_improvement > 5:
                recommendations.append(f"重排序显著提升了查询性能（提升{time_improvement:.1f}%），建议全面启用")
            elif time_improvement < -5:
                recommendations.append(f"重排序显著降低了查询性能（降低{abs(time_improvement):.1f}%），建议优化或禁用")
        else:
            recommendations.append("性能差异不显著，需要更多数据或优化重排序算法")
        
        # 基于用户反馈的建议
        feedback_analysis = report.get('user_feedback_analysis', {})
        feedback_diff = feedback_analysis.get('feedback_difference', 0)
        if feedback_diff > 0.5:
            recommendations.append(f"用户反馈显示重排序效果更好（提升{feedback_diff:.2f}分），建议推广")
        elif feedback_diff < -0.5:
            recommendations.append(f"用户反馈显示重排序效果较差（降低{abs(feedback_diff):.2f}分），需要改进")
        
        # 基于样本量的建议
        control_count = report['group_stats']['control']['count']
        treatment_count = report['group_stats']['treatment']['count']
        if control_count < 100 or treatment_count < 100:
            recommendations.append("样本量较小，建议收集更多数据以获得更可靠的结论")
        
        return recommendations
    
    def reset_test(self):
        """重置测试数据"""
        self.test_results = {
            ABTestGroup.CONTROL.value: [],
            ABTestGroup.TREATMENT.value: []
        }
        self.user_groups.clear()
        self.test_stats = {
            'total_queries': 0,
            'control_queries': 0,
            'treatment_queries': 0,
            'start_time': datetime.now().isoformat()
        }
        self.enhanced_rag.reset_stats()
        
        logger.info(f"A/B测试 '{self.test_name}' 已重置")
    
    def export_test_data(self) -> Dict[str, Any]:
        """导出测试数据"""
        return {
            'test_name': self.test_name,
            'test_config': {
                'treatment_ratio': self.treatment_ratio,
                'start_time': self.test_stats['start_time']
            },
            'test_results': self.test_results,
            'test_stats': self.test_stats,
            'user_groups': {k: v.value for k, v in self.user_groups.items()},
            'export_timestamp': datetime.now().isoformat()
        }