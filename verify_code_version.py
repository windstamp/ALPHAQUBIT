#!/usr/bin/env python3
"""
Quick script to verify code version and check for skip logic.
Run this on the server to verify code is up-to-date.

Usage:
    python verify_code_version.py
"""

import subprocess
import os
from pathlib import Path

def main():
    print("="*60)
    print("  代码版本验证")
    print("="*60)
    
    # 1. Show git commit
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        print(f"\n当前 Git 提交: {result.stdout.strip()}")
    except Exception as e:
        print(f"\n无法获取 Git 提交: {e}")
    
    # 2. Show last commit message
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--oneline'],
            capture_output=True, text=True, check=True
        )
        print(f"最后提交: {result.stdout.strip()}")
    except Exception as e:
        print(f"无法获取最后提交: {e}")
    
    # 3. Check for uncommitted changes
    try:
        result = subprocess.run(
            ['git', 'status', '--short'],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print(f"\n⚠️ 有未提交的更改:")
            print(result.stdout)
        else:
            print("\n✓ 没有未提交的更改")
    except Exception as e:
        print(f"无法获取 Git 状态: {e}")
    
    # 4. Check run_full_pipeline.py for backup function
    print("\n" + "-"*60)
    print("检查 run_full_pipeline.py:")
    
    pipeline_file = Path("run_full_pipeline.py")
    if pipeline_file.exists():
        content = pipeline_file.read_text()
        
        if "_backup_old_results" in content:
            print("  ✓ _backup_old_results 函数存在")
        else:
            print("  ✗ _backup_old_results 函数不存在 - 需要 git pull!")
        
        if "backup_until_" in content:
            print("  ✓ backup_until_ 备份逻辑存在")
        else:
            print("  ✗ backup_until_ 备份逻辑不存在 - 需要 git pull!")
    else:
        print("  ✗ run_full_pipeline.py 不存在!")
    
    # 5. Check run_server_pipeline.py
    print("\n" + "-"*60)
    print("检查 run_server_pipeline.py:")
    
    server_file = Path("run_server_pipeline.py")
    if server_file.exists():
        content = server_file.read_text()
        
        if "force_regenerate" in content:
            print("  ✓ force_regenerate 参数存在")
        else:
            print("  ✗ force_regenerate 参数不存在 - 需要 git pull!")
        
        if "--skip-existing" in content:
            print("  ✓ --skip-existing 参数存在")
        else:
            print("  ✗ --skip-existing 参数不存在")
        
        # Check for skip logic issues
        if "and not self.force_regenerate" in content:
            print("  ✓ 跳过逻辑已更新为检查 force_regenerate")
        elif "if output_file.exists():" in content and "self.force_regenerate" not in content:
            print("  ✗ 跳过逻辑仍在使用旧版本 - 需要 git pull!")
    else:
        print("  ✗ run_server_pipeline.py 不存在!")
    
    # 6. Check run_evaluation_and_plot.py
    print("\n" + "-"*60)
    print("检查 run_evaluation_and_plot.py:")
    
    eval_file = Path("run_evaluation_and_plot.py")
    if eval_file.exists():
        content = eval_file.read_text()
        
        if "input(" in content:
            print("  ✗ 仍有 input() 调用 - 需要 git pull!")
        else:
            print("  ✓ 没有 input() 调用")
        
        if "find_directory" in content:
            print("  ✓ find_directory 函数存在")
        else:
            print("  ✗ find_directory 函数不存在")
    else:
        print("  ✗ run_evaluation_and_plot.py 不存在!")
    
    # 7. Search for problematic patterns
    print("\n" + "-"*60)
    print("搜索问题模式:")
    
    problematic_patterns = [
        ("test_summary.json 已存在", "不应该存在的跳过消息"),
        ("跳过步骤1", "不应该存在的跳过消息"),
        ("Re-run evaluation", "不应该存在的用户提示"),
    ]
    
    py_files = list(Path(".").glob("*.py")) + list(Path("ai_models").glob("*.py"))
    
    found_issues = False
    for pattern, desc in problematic_patterns:
        for py_file in py_files:
            try:
                content = py_file.read_text()
                if pattern in content:
                    print(f"  ✗ 在 {py_file} 中发现 '{pattern}' ({desc})")
                    found_issues = True
            except:
                pass
    
    if not found_issues:
        print("  ✓ 没有发现问题模式")
    
    print("\n" + "="*60)
    print("建议:")
    print("  如果发现任何问题,请运行:")
    print("    git fetch origin")
    print("    git reset --hard origin/main")
    print("  这将强制同步到最新代码")
    print("="*60)

if __name__ == "__main__":
    main()
