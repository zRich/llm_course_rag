#!/usr/bin/env python3
"""
深度代码调查脚本
详细分析每个有问题lesson分支的实际代码内容和缺失情况
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import difflib

class DeepCodeInvestigator:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.investigation_results = {}
        
    def get_branch_files(self, branch: str) -> Dict[str, Any]:
        """获取指定分支的所有文件信息"""
        try:
            # 切换到指定分支
            subprocess.run(['git', 'checkout', branch], 
                         cwd=self.repo_path, capture_output=True, check=True)
            
            # 获取所有Python文件
            python_files = []
            for root, dirs, files in os.walk(self.repo_path):
                # 跳过.git目录和虚拟环境目录
                if '.git' in root or '.venv' in root or 'venv' in root or '__pycache__' in root:
                    continue
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        rel_path = file_path.relative_to(self.repo_path)
                        python_files.append(str(rel_path))
            
            # 获取文件内容和统计信息
            file_details = {}
            total_lines = 0
            
            for file_path in python_files:
                full_path = self.repo_path / file_path
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                        
                        file_details[file_path] = {
                            'total_lines': len(lines),
                            'code_lines': len(code_lines),
                            'content_preview': content[:500] + '...' if len(content) > 500 else content,
                            'imports': self.extract_imports(content),
                            'functions': self.extract_functions(content),
                            'classes': self.extract_classes(content)
                        }
                        total_lines += len(lines)
                except Exception as e:
                    file_details[file_path] = {'error': str(e)}
            
            return {
                'python_files': python_files,
                'file_count': len(python_files),
                'total_lines': total_lines,
                'file_details': file_details
            }
            
        except subprocess.CalledProcessError as e:
            return {'error': f'Failed to checkout branch {branch}: {e}'}
    
    def extract_imports(self, content: str) -> List[str]:
        """提取import语句"""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports
    
    def extract_functions(self, content: str) -> List[str]:
        """提取函数定义"""
        functions = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('def ') and '(' in line:
                functions.append(line.split('(')[0].replace('def ', ''))
        return functions
    
    def extract_classes(self, content: str) -> List[str]:
        """提取类定义"""
        classes = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('class ') and ':' in line:
                classes.append(line.split(':')[0].replace('class ', ''))
        return classes
    
    def analyze_lesson_implementation(self, lesson_id: str, expected_features: List[str]) -> Dict[str, Any]:
        """分析lesson的具体实现情况"""
        branch_name = lesson_id
        branch_info = self.get_branch_files(branch_name)
        
        if 'error' in branch_info:
            return {
                'lesson': lesson_id,
                'status': 'branch_not_found',
                'error': branch_info['error']
            }
        
        # 分析代码内容
        analysis = {
            'lesson': lesson_id,
            'branch_info': branch_info,
            'feature_analysis': {},
            'code_quality': {},
            'missing_implementations': []
        }
        
        # 检查预期功能的实现情况
        for feature in expected_features:
            analysis['feature_analysis'][feature] = self.check_feature_implementation(
                branch_info['file_details'], feature
            )
        
        # 代码质量分析
        analysis['code_quality'] = self.analyze_code_quality(branch_info['file_details'])
        
        return analysis
    
    def check_feature_implementation(self, file_details: Dict, feature: str) -> Dict[str, Any]:
        """检查特定功能的实现情况"""
        feature_keywords = {
            'hybrid_search': ['bm25', 'vector', 'hybrid', 'fusion', 'rrf'],
            'metadata_filter': ['metadata', 'filter', 'where', 'condition'],
            'rerank': ['rerank', 'reranker', 'cross_encoder', 'cohere'],
            'cache': ['cache', 'redis', 'memory', 'lru'],
            'batch_processing': ['batch', 'bulk', 'queue', 'async'],
            'incremental_update': ['incremental', 'update', 'delta', 'version'],
            'structured_data': ['database', 'sql', 'api', 'json', 'csv'],
            'text_cleaning': ['clean', 'preprocess', 'normalize', 'denoise'],
            'chunking_strategy': ['chunk', 'split', 'segment', 'overlap'],
            'citation': ['citation', 'reference', 'source', 'provenance']
        }
        
        keywords = feature_keywords.get(feature, [feature.lower()])
        found_evidence = []
        
        for file_path, details in file_details.items():
            if 'error' in details:
                continue
                
            content = details.get('content_preview', '').lower()
            functions = [f.lower() for f in details.get('functions', [])]
            classes = [c.lower() for c in details.get('classes', [])]
            imports = [i.lower() for i in details.get('imports', [])]
            
            for keyword in keywords:
                if (keyword in content or 
                    any(keyword in f for f in functions) or
                    any(keyword in c for c in classes) or
                    any(keyword in i for i in imports)):
                    found_evidence.append({
                        'file': file_path,
                        'keyword': keyword,
                        'context': 'Found in code content'
                    })
        
        return {
            'implemented': len(found_evidence) > 0,
            'evidence': found_evidence,
            'confidence': min(len(found_evidence) * 0.3, 1.0)
        }
    
    def analyze_code_quality(self, file_details: Dict) -> Dict[str, Any]:
        """分析代码质量"""
        total_files = len(file_details)
        total_lines = sum(d.get('total_lines', 0) for d in file_details.values() if 'error' not in d)
        total_code_lines = sum(d.get('code_lines', 0) for d in file_details.values() if 'error' not in d)
        
        avg_file_size = total_lines / total_files if total_files > 0 else 0
        code_ratio = total_code_lines / total_lines if total_lines > 0 else 0
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_code_lines': total_code_lines,
            'avg_file_size': avg_file_size,
            'code_ratio': code_ratio,
            'quality_score': min(code_ratio * 100, 100)
        }
    
    def investigate_problematic_lessons(self):
        """调查有问题的lesson"""
        # 基于之前的分析结果，定义有问题的lesson和预期功能
        problematic_lessons = {
            'lesson02': ['docker', 'container', 'postgresql', 'redis'],
            'lesson03': ['database', 'model', 'migration', 'schema'],
            'lesson04': ['pdf', 'parse', 'chunk', 'split'],
            'lesson06': ['rag', 'retrieval', 'generation', 'query'],
            'lesson08': ['hybrid_search', 'bm25', 'vector', 'fusion'],
            'lesson09': ['metadata_filter', 'condition', 'where'],
            'lesson11': ['chunk', 'overlap', 'experiment', 'size'],
            'lesson12': ['multi_document', 'word', 'excel', 'ppt'],
            'lesson13': ['citation', 'reference', 'source', 'provenance'],
            'lesson14': ['cache', 'redis', 'performance'],
            'lesson15': ['batch_processing', 'bulk', 'resume'],
            'lesson16': ['incremental_update', 'delta', 'version'],
            'lesson17': ['structured_data', 'database', 'api']
        }
        
        results = {}
        
        for lesson_id, expected_features in problematic_lessons.items():
            print(f"\n🔍 调查 {lesson_id}...")
            analysis = self.analyze_lesson_implementation(lesson_id, expected_features)
            results[lesson_id] = analysis
            
            # 打印简要结果
            if 'error' in analysis:
                print(f"❌ {lesson_id}: {analysis['error']}")
            else:
                implemented_features = sum(1 for f in analysis['feature_analysis'].values() if f['implemented'])
                total_features = len(expected_features)
                print(f"📊 {lesson_id}: {implemented_features}/{total_features} 功能已实现")
                print(f"📁 文件数: {analysis['branch_info']['file_count']}")
                print(f"📝 代码行数: {analysis['branch_info']['total_lines']}")
        
        return results
    
    def save_investigation_results(self, results: Dict, filename: str = 'deep_investigation_results.json'):
        """保存调查结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 调查结果已保存到 {filename}")

def main():
    print("🔍 开始深度代码调查...")
    
    # 初始化调查器
    investigator = DeepCodeInvestigator('.')
    
    # 调查有问题的lesson
    results = investigator.investigate_problematic_lessons()
    
    # 保存结果
    investigator.save_investigation_results(results)
    
    # 生成摘要报告
    print("\n📋 调查摘要:")
    print("=" * 50)
    
    for lesson_id, analysis in results.items():
        if 'error' in analysis:
            print(f"❌ {lesson_id}: 分支不存在或无法访问")
        else:
            implemented = sum(1 for f in analysis['feature_analysis'].values() if f['implemented'])
            total = len(analysis['feature_analysis'])
            quality = analysis['code_quality']['quality_score']
            print(f"📊 {lesson_id}: {implemented}/{total} 功能实现, 质量分数: {quality:.1f}")
    
    print("\n✅ 深度调查完成！")

if __name__ == '__main__':
    main()