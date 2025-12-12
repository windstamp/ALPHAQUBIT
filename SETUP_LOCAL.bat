@echo off
REM =============================================================================
REM AlphaQubit 本地快速设置 (Windows Batch)
REM 双击运行此文件进行设置
REM =============================================================================

echo ==============================================
echo AlphaQubit 本地环境快速设置
echo ==============================================
echo.

REM 检查是否在正确目录
if not exist "run_server_pipeline.py" (
    echo 错误: 请在 ALPHAQUBIT 目录下运行此脚本
    pause
    exit /b 1
)

echo 开始设置...
echo.

REM 运行 PowerShell 设置脚本
powershell -ExecutionPolicy Bypass -File ".\local_setup\setup_local.ps1"

echo.
echo ==============================================
echo 设置完成!
echo ==============================================
pause
