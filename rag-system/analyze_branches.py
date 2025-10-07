#!/usr/bin/env python3

import subprocess
import json
from collections import defaultdict

def run_git_command(cmd):
    """执行git命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def analyze_branch_changes():
    """分析所有lesson分支的增量变更"""
    branches = []
    for i in range(1, 20):
        branches.append(f"lesson{i:02d}")
    
    analysis_results = []
    
    # 分析lesson01作为基准
    print("分析lesson01基准...")
    run_git_command("git checkout lesson01")
    
    # 统计lesson01的文件和代码行数
    py_files = run_git_command("find src -name '*.py' | wc -l")
    total_lines = run_git_command("find src -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $1}'")
    
    lesson01_info = {
        "branch": "lesson01",
        "python_files": int(py_files) if py_files else 0,
        "total_lines": int(total_lines) if total_lines else 0,
        "changes_from_previous": "基准分支",
        "new_files": [],
        "modified_files": [],
        "deleted_files": [],
        "insertions": 0,
        "deletions": 0
    }
    analysis_results.append(lesson01_info)
    
    # 分析lesson02到lesson19的增量变更
    for i in range(2, 20):
        current_branch = f"lesson{i:02d}"
        previous_branch = f"lesson{i-1:02d}"
        
        print(f"分析{current_branch}相对于{previous_branch}的变更...")
        
        # 切换到当前分支
        run_git_command(f"git checkout {current_branch}")
        
        # 获取差异统计
        diff_stat = run_git_command(f"git diff {previous_branch}..{current_branch} --stat")
        
        # 获取详细的文件变更信息
        diff_name_status = run_git_command(f"git diff {previous_branch}..{current_branch} --name-status")
        
        # 获取插入和删除的行数
        diff_numstat = run_git_command(f"git diff {previous_branch}..{current_branch} --numstat")
        
        # 统计当前分支的文件和代码行数
        py_files = run_git_command("find src -name '*.py' | wc -l")
        total_lines = run_git_command("find src -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $1}'")
        
        # 解析文件变更
        new_files = []
        modified_files = []
        deleted_files = []
        
        if diff_name_status:
            for line in diff_name_status.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        status = parts[0]
                        filename = parts[1]
                        if status == 'A':
                            new_files.append(filename)
                        elif status == 'M':
                            modified_files.append(filename)
                        elif status == 'D':
                            deleted_files.append(filename)
        
        # 计算总的插入和删除行数
        total_insertions = 0
        total_deletions = 0
        
        if diff_numstat:
            for line in diff_numstat.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            insertions = int(parts[0]) if parts[0] != '-' else 0
                            deletions = int(parts[1]) if parts[1] != '-' else 0
                            total_insertions += insertions
                            total_deletions += deletions
                        except ValueError:
                            continue
        
        branch_info = {
            "branch": current_branch,
            "python_files": int(py_files) if py_files else 0,
            "total_lines": int(total_lines) if total_lines else 0,
            "changes_from_previous": diff_stat if diff_stat else "无变更",
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "insertions": total_insertions,
            "deletions": total_deletions
        }
        
        analysis_results.append(branch_info)
    
    return analysis_results

def generate_report(analysis_results):
    """生成分析报告"""
    print("\n" + "="*80)
    print("RAG系统分支增量开发分析报告")
    print("="*80)
    
    print(f"\n总共分析了 {len(analysis_results)} 个分支 (lesson01 到 lesson19)")
    
    print("\n详细分析结果:")
    print("-"*80)
    
    for i, result in enumerate(analysis_results):
        print(f"\n{i+1}. {result['branch']}:")
        print(f"   Python文件数: {result['python_files']}")
        print(f"   总代码行数: {result['total_lines']}")
        
        if result['branch'] != 'lesson01':
            prev_result = analysis_results[i-1]
            file_diff = result['python_files'] - prev_result['python_files']
            line_diff = result['total_lines'] - prev_result['total_lines']
            
            print(f"   相对于前一分支变化:")
            print(f"     文件数变化: {file_diff:+d}")
            print(f"     代码行数变化: {line_diff:+d}")
            print(f"     插入行数: +{result['insertions']}")
            print(f"     删除行数: -{result['deletions']}")
            
            if result['new_files']:
                print(f"     新增文件: {', '.join(result['new_files'])}")
            if result['modified_files']:
                print(f"     修改文件: {', '.join(result['modified_files'])}")
            if result['deleted_files']:
                print(f"     删除文件: {', '.join(result['deleted_files'])}")
        else:
            print(f"   基准分支 (初始状态)")
    
    # 增量开发模式分析
    print("\n" + "="*80)
    print("增量开发模式分析")
    print("="*80)
    
    incremental_pattern = True
    issues = []
    
    for i in range(1, len(analysis_results)):
        current = analysis_results[i]
        previous = analysis_results[i-1]
        
        # 检查是否有代码增长
        if current['total_lines'] < previous['total_lines']:
            issues.append(f"{current['branch']}: 代码行数减少了 {previous['total_lines'] - current['total_lines']} 行")
        
        # 检查是否有大量删除而没有相应增加
        if current['deletions'] > current['insertions'] * 2:
            issues.append(f"{current['branch']}: 删除行数({current['deletions']})远大于插入行数({current['insertions']})")
    
    if issues:
        print("\n发现的潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        incremental_pattern = False
    else:
        print("\n✅ 所有分支都遵循增量开发模式")
    
    # 总结统计
    print("\n" + "="*80)
    print("总结统计")
    print("="*80)
    
    total_files_added = analysis_results[-1]['python_files'] - analysis_results[0]['python_files']
    total_lines_added = analysis_results[-1]['total_lines'] - analysis_results[0]['total_lines']
    
    print(f"从lesson01到lesson19:")
    print(f"  总共新增Python文件: {total_files_added} 个")
    print(f"  总共新增代码行数: {total_lines_added} 行")
    print(f"  平均每个lesson新增: {total_lines_added/18:.1f} 行代码")
    
    return incremental_pattern

if __name__ == "__main__":
    print("开始分析RAG系统的分支增量开发情况...")
    
    # 保存当前分支
    current_branch = run_git_command("git branch --show-current")
    
    try:
        # 执行分析
        results = analyze_branch_changes()
        
        # 生成报告
        is_incremental = generate_report(results)
        
        # 保存结果到JSON文件
        with open('branch_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_results': results,
                'is_incremental_development': is_incremental,
                'summary': {
                    'total_branches': len(results),
                    'total_files_growth': results[-1]['python_files'] - results[0]['python_files'],
                    'total_lines_growth': results[-1]['total_lines'] - results[0]['total_lines']
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析完成！详细结果已保存到 branch_analysis_report.json")
        
    finally:
        # 恢复到原始分支
        if current_branch:
            run_git_command(f"git checkout {current_branch}")
        else:
            run_git_command("git checkout main")