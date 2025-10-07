#!/usr/bin/env python3
"""
实际代码变更与课程要求对比分析脚本
"""

import json
import subprocess
from typing import Dict, List, Any, Tuple
from pathlib import Path

def load_actual_changes(filename: str = "branch_analysis_report.json") -> Dict[str, Any]:
    """
    加载实际分支变更数据
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 找不到文件 {filename}")
        return {}

def load_expected_requirements(filename: str = "lesson_requirements.json") -> Dict[str, Any]:
    """
    加载课程要求数据
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 找不到文件 {filename}")
        return {}

def analyze_lesson_implementation(lesson_id: str, actual_data: Dict, expected_data: Dict) -> Dict[str, Any]:
    """
    分析单个lesson的实现情况
    """
    analysis = {
        "lesson_id": lesson_id,
        "title": expected_data.get("title", "未知"),
        "module": expected_data.get("module", "未知"),
        "status": "unknown",
        "issues": [],
        "matches": [],
        "actual_changes": {},
        "expected_changes": expected_data.get("expected_changes", []),
        "dependency_analysis": {},
        "code_analysis": {}
    }
    
    # 查找实际变更数据
    branch_data = None
    for result in actual_data.get("analysis_results", []):
        if result.get("branch") == lesson_id:
            branch_data = result
            break
    
    if branch_data:
        analysis["actual_changes"] = {
            "files_changed": len(branch_data.get("new_files", [])) + len(branch_data.get("modified_files", [])),
            "insertions": branch_data.get("insertions", 0),
            "deletions": branch_data.get("deletions", 0),
            "new_files": branch_data.get("new_files", []),
            "modified_files": branch_data.get("modified_files", []),
            "deleted_files": branch_data.get("deleted_files", [])
        }
        
        # 分析代码变更情况
        insertions = branch_data.get("insertions", 0)
        deletions = branch_data.get("deletions", 0)
        
        if insertions == 0 and deletions == 0:
            analysis["status"] = "no_changes"
            analysis["issues"].append("该课程没有任何代码变更")
        elif deletions > insertions * 2:  # 删除行数远大于新增行数
            analysis["status"] = "major_refactor"
            analysis["issues"].append(f"存在大量删除操作 (删除{deletions}行, 新增{insertions}行)")
        elif insertions < 50:  # 新增代码很少
            analysis["status"] = "minimal_changes"
            analysis["issues"].append(f"代码变更量很少 (仅新增{insertions}行)")
        elif insertions > 1000:  # 新增代码很多
            analysis["status"] = "major_changes"
            analysis["matches"].append(f"有大量代码新增 ({insertions}行)")
        else:
            analysis["status"] = "normal_changes"
            analysis["matches"].append(f"代码变更量正常 ({insertions}行新增)")
    else:
        analysis["status"] = "missing_branch"
        analysis["issues"].append("找不到对应的分支数据")
    
    # 分析依赖变更
    expected_deps = expected_data.get("dependencies", [])
    if expected_deps:
        analysis["dependency_analysis"] = {
            "expected_dependencies": expected_deps,
            "dependency_match": "需要检查pyproject.toml变更"
        }
    
    return analysis

def generate_comparison_report(actual_data: Dict, expected_data: Dict) -> Dict[str, Any]:
    """
    生成完整的对比分析报告
    """
    report = {
        "summary": {
            "total_lessons": len(expected_data),
            "analyzed_lessons": 0,
            "issues_found": 0,
            "major_issues": [],
            "minor_issues": [],
            "normal_implementations": []
        },
        "detailed_analysis": {},
        "recommendations": []
    }
    
    # 逐个分析每个lesson
    for lesson_id, expected_info in expected_data.items():
        if lesson_id.startswith("lesson"):
            analysis = analyze_lesson_implementation(lesson_id, actual_data, expected_info)
            report["detailed_analysis"][lesson_id] = analysis
            report["summary"]["analyzed_lessons"] += 1
            
            # 统计问题
            if analysis["issues"]:
                report["summary"]["issues_found"] += 1
                
                if analysis["status"] in ["no_changes", "major_refactor", "missing_branch"]:
                    report["summary"]["major_issues"].append({
                        "lesson": lesson_id,
                        "title": analysis["title"],
                        "status": analysis["status"],
                        "issues": analysis["issues"]
                    })
                else:
                    report["summary"]["minor_issues"].append({
                        "lesson": lesson_id,
                        "title": analysis["title"],
                        "status": analysis["status"],
                        "issues": analysis["issues"]
                    })
            else:
                report["summary"]["normal_implementations"].append({
                    "lesson": lesson_id,
                    "title": analysis["title"],
                    "status": analysis["status"]
                })
    
    # 生成建议
    if report["summary"]["major_issues"]:
        report["recommendations"].append("需要重点关注存在重大问题的课程")
    
    if len(report["summary"]["major_issues"]) > 3:
        report["recommendations"].append("建议重新审查课程实现与讲义的匹配度")
    
    return report

def print_comparison_summary(report: Dict[str, Any]):
    """
    打印对比分析摘要
    """
    summary = report["summary"]
    
    print("\n=== 课程实现与要求对比分析报告 ===")
    print(f"总课程数: {summary['total_lessons']}")
    print(f"已分析课程数: {summary['analyzed_lessons']}")
    print(f"发现问题的课程数: {summary['issues_found']}")
    
    print("\n=== 重大问题 ===")
    for issue in summary["major_issues"]:
        print(f"  {issue['lesson']}: {issue['title']}")
        print(f"    状态: {issue['status']}")
        for problem in issue["issues"]:
            print(f"    问题: {problem}")
    
    print("\n=== 轻微问题 ===")
    for issue in summary["minor_issues"]:
        print(f"  {issue['lesson']}: {issue['title']}")
        print(f"    状态: {issue['status']}")
        for problem in issue["issues"]:
            print(f"    问题: {problem}")
    
    print("\n=== 正常实现 ===")
    for normal in summary["normal_implementations"]:
        print(f"  {normal['lesson']}: {normal['title']} ({normal['status']})")
    
    print("\n=== 建议 ===")
    for rec in report["recommendations"]:
        print(f"  - {rec}")

def save_comparison_report(report: Dict[str, Any], filename: str = "lesson_comparison_report.json"):
    """
    保存对比分析报告
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n对比分析报告已保存到 {filename}")

def investigate_lesson11_refactor(actual_data: Dict) -> Dict[str, Any]:
    """
    深入调查lesson11的重构情况
    """
    investigation = {
        "lesson_id": "lesson11",
        "refactor_analysis": {},
        "impact_assessment": {},
        "recommendations": []
    }
    
    # 查找lesson11数据
    lesson11_data = None
    for result in actual_data.get("analysis_results", []):
        if result.get("branch") == "lesson11":
            lesson11_data = result
            break
    
    if lesson11_data:
        investigation["refactor_analysis"] = {
            "insertions": lesson11_data.get("insertions", 0),
            "deletions": lesson11_data.get("deletions", 0),
            "deletion_ratio": lesson11_data.get("deletions", 0) / max(lesson11_data.get("insertions", 1), 1),
            "new_files": len(lesson11_data.get("new_files", [])),
            "deleted_files": len(lesson11_data.get("deleted_files", [])),
            "modified_files": len(lesson11_data.get("modified_files", []))
        }
        
        # 分析影响
        if lesson11_data.get("deletions", 0) > 1000:
            investigation["impact_assessment"]["severity"] = "high"
            investigation["impact_assessment"]["description"] = "大规模重构，可能影响系统稳定性"
        
        # 生成建议
        investigation["recommendations"] = [
            "需要详细检查lesson11的具体变更内容",
            "确认重构是否符合课程教学目标",
            "评估对后续课程的影响"
        ]
    
    return investigation

if __name__ == "__main__":
    # 加载数据
    print("加载实际变更数据...")
    actual_data = load_actual_changes()
    
    print("加载课程要求数据...")
    expected_data = load_expected_requirements()
    
    if not actual_data or not expected_data:
        print("错误: 无法加载必要的数据文件")
        exit(1)
    
    # 生成对比报告
    print("\n生成对比分析报告...")
    comparison_report = generate_comparison_report(actual_data, expected_data)
    
    # 调查lesson11重构
    print("\n调查lesson11重构情况...")
    lesson11_investigation = investigate_lesson11_refactor(actual_data)
    comparison_report["lesson11_investigation"] = lesson11_investigation
    
    # 保存报告
    save_comparison_report(comparison_report)
    
    # 打印摘要
    print_comparison_summary(comparison_report)
    
    # 打印lesson11调查结果
    print("\n=== Lesson11 重构调查 ===")
    refactor = lesson11_investigation["refactor_analysis"]
    if refactor:
        print(f"新增行数: {refactor['insertions']}")
        print(f"删除行数: {refactor['deletions']}")
        print(f"删除/新增比例: {refactor['deletion_ratio']:.2f}")
        print(f"新增文件: {refactor['new_files']}个")
        print(f"删除文件: {refactor['deleted_files']}个")
        print(f"修改文件: {refactor['modified_files']}个")
    else:
        print("未找到lesson11的数据")
    
    for rec in lesson11_investigation["recommendations"]:
        print(f"建议: {rec}")