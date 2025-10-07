#!/bin/bash

# 全面验证所有lesson分支的rerank模块完整性
# 检查lesson11-19分支的修复状态和功能完整性

echo "🔍 开始全面验证所有lesson分支..."
echo "===========================================\n"

# 定义所有lesson分支
branches=("lesson11" "lesson12" "lesson13" "lesson14" "lesson15" "lesson16" "lesson17" "lesson18" "lesson19")

# 核心文件列表
core_files=(
    "__init__.py"
    "rerank_service.py"
    "cached_rerank_service.py"
    "enhanced_rag_system.py"
    "rerank_ab_test.py"
    "ab_test_rerank.py"
    "enhanced_rag_query.py"
    "test_basic_rerank.py"
    "test_cached_rerank.py"
)

# 统计变量
total_branches=${#branches[@]}
complete_branches=0
incomplete_branches=0
missing_branches=0

# 详细检查结果
declare -A branch_status
declare -A missing_files

for branch in "${branches[@]}"; do
    echo "📋 检查分支: $branch"
    echo "-------------------------------------------"
    
    # 尝试切换分支
    if git checkout "$branch" &>/dev/null; then
        echo "✅ 成功切换到分支 $branch"
        
        # 检查rerank目录
        if [ -d "src/rerank" ]; then
            echo "📁 rerank目录存在"
            
            # 检查每个核心文件
            missing_count=0
            missing_list=()
            
            for file in "${core_files[@]}"; do
                if [ -f "src/rerank/$file" ]; then
                    file_size=$(wc -c < "src/rerank/$file" 2>/dev/null || echo "0")
                    if [ "$file_size" -gt 100 ]; then
                        echo "  ✅ $file (${file_size} bytes)"
                    else
                        echo "  ⚠️  $file (${file_size} bytes - 可能为空文件)"
                        missing_list+=("$file")
                        ((missing_count++))
                    fi
                else
                    echo "  ❌ $file (缺失)"
                    missing_list+=("$file")
                    ((missing_count++))
                fi
            done
            
            # 判断分支状态
            if [ $missing_count -eq 0 ]; then
                echo "🎉 分支 $branch: 完整 (${#core_files[@]}/${#core_files[@]} 文件)"
                branch_status[$branch]="完整"
                ((complete_branches++))
            elif [ $missing_count -lt 3 ]; then
                echo "⚠️  分支 $branch: 基本完整 (缺失 $missing_count 个文件)"
                branch_status[$branch]="基本完整"
                missing_files[$branch]="${missing_list[*]}"
                ((incomplete_branches++))
            else
                echo "❌ 分支 $branch: 不完整 (缺失 $missing_count 个文件)"
                branch_status[$branch]="不完整"
                missing_files[$branch]="${missing_list[*]}"
                ((incomplete_branches++))
            fi
            
        else
            echo "❌ rerank目录不存在"
            branch_status[$branch]="目录缺失"
            ((missing_branches++))
        fi
        
    else
        echo "❌ 无法切换到分支 $branch"
        branch_status[$branch]="分支不存在"
        ((missing_branches++))
    fi
    
    echo "\n"
done

# 生成总结报告
echo "\n\n📊 总结报告"
echo "==========================================="
echo "总分支数: $total_branches"
echo "完整分支: $complete_branches"
echo "不完整分支: $incomplete_branches"
echo "缺失分支: $missing_branches"
echo "\n📈 完整率: $(( complete_branches * 100 / total_branches ))%"

echo "\n\n📋 详细状态:"
echo "-------------------------------------------"
for branch in "${branches[@]}"; do
    status="${branch_status[$branch]}"
    case $status in
        "完整")
            echo "✅ $branch: $status"
            ;;
        "基本完整")
            echo "⚠️  $branch: $status (缺失: ${missing_files[$branch]})"
            ;;
        "不完整")
            echo "❌ $branch: $status (缺失: ${missing_files[$branch]})"
            ;;
        *)
            echo "❌ $branch: $status"
            ;;
    esac
done

# 提供修复建议
if [ $incomplete_branches -gt 0 ] || [ $missing_branches -gt 0 ]; then
    echo "\n\n🔧 修复建议:"
    echo "-------------------------------------------"
    
    if [ $incomplete_branches -gt 0 ]; then
        echo "对于不完整的分支，可以运行以下命令修复:"
        echo "  ./fix_rerank_core_services.sh"
    fi
    
    if [ $missing_branches -gt 0 ]; then
        echo "对于缺失rerank目录的分支，需要先恢复基础结构:"
        echo "  ./fix_rerank_modules.sh"
    fi
else
    echo "\n\n🎉 所有分支状态良好！"
    echo "RAG系统课程的rerank模块已全面修复完成。"
fi

echo "\n验证完成！"