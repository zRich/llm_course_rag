#!/bin/bash

# 批量修复lesson分支的rerank核心服务类
# 为lesson11-19分支添加缺失的核心服务实现

echo "开始批量修复rerank核心服务类..."

# 定义需要修复的分支列表
branches=("lesson11" "lesson12" "lesson13" "lesson14" "lesson16" "lesson17" "lesson18" "lesson19")

# 核心服务文件列表
core_files=(
    "rerank_service.py"
    "cached_rerank_service.py"
    "enhanced_rag_system.py"
    "rerank_ab_test.py"
)

# 记录修复结果
fixed_branches=()
skipped_branches=()
failed_branches=()

for branch in "${branches[@]}"; do
    echo "\n=== 处理分支: $branch ==="
    
    # 切换到目标分支
    if git checkout "$branch" 2>/dev/null; then
        echo "✅ 成功切换到分支 $branch"
    else
        echo "❌ 切换到分支 $branch 失败"
        failed_branches+=("$branch")
        continue
    fi
    
    # 检查rerank目录是否存在
    if [ ! -d "src/rerank" ]; then
        echo "❌ 分支 $branch 缺少 src/rerank 目录，跳过"
        skipped_branches+=("$branch")
        continue
    fi
    
    # 检查是否需要修复
    needs_fix=false
    for file in "${core_files[@]}"; do
        if [ ! -f "src/rerank/$file" ]; then
            echo "⚠️  缺少文件: src/rerank/$file"
            needs_fix=true
        fi
    done
    
    if [ "$needs_fix" = false ]; then
        echo "✅ 分支 $branch 的核心服务文件已完整，跳过"
        skipped_branches+=("$branch")
        continue
    fi
    
    # 从lesson15分支复制核心服务文件
    echo "📋 从lesson15分支复制核心服务文件..."
    
    # 临时保存当前分支的文件
    temp_dir="/tmp/rerank_core_backup_$branch"
    mkdir -p "$temp_dir"
    
    # 切换到lesson15获取最新的核心服务文件
    git checkout lesson15 2>/dev/null
    if [ $? -eq 0 ]; then
        # 复制核心服务文件到临时目录
        for file in "${core_files[@]}"; do
            if [ -f "src/rerank/$file" ]; then
                cp "src/rerank/$file" "$temp_dir/"
                echo "  📄 复制 $file"
            fi
        done
        
        # 切换回目标分支
        git checkout "$branch" 2>/dev/null
        
        # 复制文件到目标分支
        for file in "${core_files[@]}"; do
            if [ -f "$temp_dir/$file" ]; then
                cp "$temp_dir/$file" "src/rerank/"
                echo "  ✅ 安装 $file 到分支 $branch"
            fi
        done
        
        # 更新__init__.py文件以确保正确导入
        if [ -f "$temp_dir/../__init__.py" ]; then
            cp "src/rerank/__init__.py" "$temp_dir/__init__.py.backup" 2>/dev/null
        fi
        
        # 从lesson15复制更新的__init__.py
        git checkout lesson15 2>/dev/null
        cp "src/rerank/__init__.py" "$temp_dir/__init__.py.new" 2>/dev/null
        git checkout "$branch" 2>/dev/null
        
        if [ -f "$temp_dir/__init__.py.new" ]; then
            cp "$temp_dir/__init__.py.new" "src/rerank/__init__.py"
            echo "  ✅ 更新 __init__.py"
        fi
        
        # 清理临时文件
        rm -rf "$temp_dir"
        
        # 提交更改
        git add src/rerank/
        if git commit -m "修复$branch: 添加缺失的rerank核心服务类" 2>/dev/null; then
            echo "✅ 成功提交修复到分支 $branch"
            fixed_branches+=("$branch")
        else
            echo "⚠️  分支 $branch 没有需要提交的更改"
            skipped_branches+=("$branch")
        fi
    else
        echo "❌ 无法切换到lesson15分支获取核心服务文件"
        failed_branches+=("$branch")
    fi
done

# 生成修复报告
echo "\n\n=== 修复报告 ==="
echo "成功修复的分支 (${#fixed_branches[@]}个):"
for branch in "${fixed_branches[@]}"; do
    echo "  ✅ $branch"
done

echo "\n跳过的分支 (${#skipped_branches[@]}个):"
for branch in "${skipped_branches[@]}"; do
    echo "  ⏭️  $branch (已完整或无需修复)"
done

echo "\n失败的分支 (${#failed_branches[@]}个):"
for branch in "${failed_branches[@]}"; do
    echo "  ❌ $branch"
done

# 验证修复效果
echo "\n=== 验证修复效果 ==="
for branch in "${fixed_branches[@]}"; do
    echo "\n检查分支 $branch:"
    git checkout "$branch" 2>/dev/null
    
    echo "  核心服务文件检查:"
    for file in "${core_files[@]}"; do
        if [ -f "src/rerank/$file" ]; then
            file_size=$(wc -c < "src/rerank/$file")
            echo "    ✅ $file (${file_size} bytes)"
        else
            echo "    ❌ $file (缺失)"
        fi
    done
done

echo "\n🎉 批量修复完成！"
echo "总计处理 ${#branches[@]} 个分支"
echo "成功修复 ${#fixed_branches[@]} 个分支"
echo "跳过 ${#skipped_branches[@]} 个分支"
echo "失败 ${#failed_branches[@]} 个分支"