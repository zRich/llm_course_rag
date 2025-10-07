#!/bin/bash

# 批量修复lesson13-19分支的rerank模块缺失问题

echo "开始批量修复rerank模块..."

# 要修复的分支列表
branches=("lesson13" "lesson14" "lesson15" "lesson16" "lesson17" "lesson18" "lesson19")

for branch in "${branches[@]}"; do
    echo "\n=== 修复分支: $branch ==="
    
    # 切换到目标分支
    git checkout "$branch"
    if [ $? -ne 0 ]; then
        echo "错误: 无法切换到分支 $branch"
        continue
    fi
    
    # 检查是否已存在rerank目录
    if [ -d "src/rerank" ]; then
        echo "分支 $branch 已存在rerank模块，跳过"
        continue
    fi
    
    # 复制rerank模块
    cp -r /tmp/rerank_backup src/rerank
    if [ $? -ne 0 ]; then
        echo "错误: 无法复制rerank模块到分支 $branch"
        continue
    fi
    
    # 添加并提交更改
    git add src/rerank
    git commit -m "修复${branch}: 恢复缺失的rerank模块"
    
    if [ $? -eq 0 ]; then
        echo "✓ 分支 $branch 修复成功"
    else
        echo "✗ 分支 $branch 提交失败"
    fi
done

echo "\n=== 批量修复完成 ==="
echo "已修复的分支:"
for branch in "${branches[@]}"; do
    git checkout "$branch" 2>/dev/null
    if [ -d "src/rerank" ]; then
        echo "✓ $branch - rerank模块已恢复"
    else
        echo "✗ $branch - rerank模块仍缺失"
    fi
done