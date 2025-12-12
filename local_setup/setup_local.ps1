# =============================================================================
# AlphaQubit 本地环境设置脚本 (Windows PowerShell)
# =============================================================================
#
# 使用方法:
#   1. 以管理员身份运行 PowerShell
#   2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   3. .\setup_local.ps1
#
# =============================================================================

Write-Host "=============================================="
Write-Host "AlphaQubit 本地环境设置"
Write-Host "=============================================="
Write-Host ""

# =============================================================================
# Step 1: 检查并安装必要工具
# =============================================================================

Write-Host "[Step 1] 检查必要工具..."

# 检查 SSH
if (Get-Command ssh -ErrorAction SilentlyContinue) {
    Write-Host "  ✓ SSH 已安装"
} else {
    Write-Host "  ! SSH 未找到，请确保 Windows OpenSSH 已启用"
    Write-Host "    设置 -> 应用 -> 可选功能 -> 添加功能 -> OpenSSH客户端"
}

# 检查 scp
if (Get-Command scp -ErrorAction SilentlyContinue) {
    Write-Host "  ✓ SCP 已安装"
} else {
    Write-Host "  ! SCP 未找到"
}

Write-Host ""

# =============================================================================
# Step 2: 配置 SSH
# =============================================================================

Write-Host "[Step 2] 配置 SSH..."

$sshDir = "$env:USERPROFILE\.ssh"
$sshConfig = "$sshDir\config"

# 创建 .ssh 目录
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
    Write-Host "  ✓ 创建 .ssh 目录"
}

# 检查是否有 SSH 密钥
$keyFile = "$sshDir\id_rsa"
if (-not (Test-Path $keyFile)) {
    Write-Host "  ! 未找到 SSH 密钥"
    Write-Host "  生成新的 SSH 密钥? (y/n)"
    $response = Read-Host
    if ($response -eq 'y') {
        ssh-keygen -t rsa -b 4096 -f $keyFile -N '""'
        Write-Host "  ✓ SSH 密钥已生成"
        Write-Host ""
        Write-Host "  请将以下公钥添加到远程服务器的 ~/.ssh/authorized_keys:"
        Write-Host "  =================================================="
        Get-Content "$keyFile.pub"
        Write-Host "  =================================================="
    }
} else {
    Write-Host "  ✓ SSH 密钥已存在"
}

Write-Host ""

# =============================================================================
# Step 3: 创建 SSH 配置
# =============================================================================

Write-Host "[Step 3] 配置远程服务器连接..."
Write-Host ""
Write-Host "请输入服务器信息:"

$serverIP = Read-Host "  服务器 IP 地址"
$serverUser = Read-Host "  用户名"
$serverPort = Read-Host "  SSH 端口 (默认 22)"
if ([string]::IsNullOrEmpty($serverPort)) { $serverPort = "22" }

# 生成 SSH 配置
$sshConfigContent = @"

# 中国移动华为昇腾NPU服务器 - AlphaQubit
Host cmcc-npu
    HostName $serverIP
    User $serverUser
    Port $serverPort
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ForwardAgent yes

"@

# 追加到 config 文件
Add-Content -Path $sshConfig -Value $sshConfigContent
Write-Host "  ✓ SSH 配置已添加到 $sshConfig"

Write-Host ""

# =============================================================================
# Step 4: 测试连接
# =============================================================================

Write-Host "[Step 4] 测试 SSH 连接..."
Write-Host "  运行: ssh cmcc-npu"
Write-Host ""
Write-Host "  按 Enter 测试连接，或按 Ctrl+C 跳过"
Read-Host

try {
    ssh cmcc-npu "echo '连接成功!' && uname -a"
} catch {
    Write-Host "  ! 连接失败，请检查配置"
}

Write-Host ""

# =============================================================================
# Step 5: 创建便捷脚本
# =============================================================================

Write-Host "[Step 5] 创建便捷脚本..."

$scriptsDir = "$(Get-Location)\local_scripts"
if (-not (Test-Path $scriptsDir)) {
    New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null
}

# 连接服务器脚本
$connectScript = @"
# 连接到 NPU 服务器
Write-Host "连接到 NPU 服务器..."
ssh cmcc-npu
"@
Set-Content -Path "$scriptsDir\connect_server.ps1" -Value $connectScript

# 上传代码脚本
$uploadScript = @"
# 上传 AlphaQubit 代码到服务器
Write-Host "上传代码到服务器..."
`$localPath = "$(Get-Location)"
scp -r "`$localPath" cmcc-npu:~/alphaqubit/
Write-Host "上传完成!"
"@
Set-Content -Path "$scriptsDir\upload_code.ps1" -Value $uploadScript

# 下载结果脚本
$downloadScript = @"
# 从服务器下载结果
Write-Host "下载训练结果..."
`$localDir = "$(Get-Location)\downloaded_results"
if (-not (Test-Path `$localDir)) {
    New-Item -ItemType Directory -Path `$localDir -Force | Out-Null
}
scp -r "cmcc-npu:~/alphaqubit/ALPHAQUBIT/results_*" `$localDir/
Write-Host "下载完成! 结果保存在: `$localDir"
"@
Set-Content -Path "$scriptsDir\download_results.ps1" -Value $downloadScript

# 查看训练日志脚本
$logScript = @"
# 查看远程训练日志
Write-Host "查看训练日志 (按 Ctrl+C 退出)..."
ssh cmcc-npu "tail -f ~/alphaqubit/ALPHAQUBIT/training_*.log"
"@
Set-Content -Path "$scriptsDir\view_log.ps1" -Value $logScript

# 运行分析脚本
$analyzeScript = @"
# 分析下载的结果
Write-Host "分析训练结果..."
`$resultsDir = Get-ChildItem -Path "$(Get-Location)\downloaded_results" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (`$resultsDir) {
    python analyze_server_results.py --results-dir `$resultsDir.FullName --output-dir ./analysis_output --pdf
    Write-Host "分析完成! 结果保存在: ./analysis_output"
} else {
    Write-Host "未找到结果目录，请先运行 download_results.ps1"
}
"@
Set-Content -Path "$scriptsDir\analyze_results.ps1" -Value $analyzeScript

Write-Host "  ✓ 创建了以下脚本:"
Write-Host "    - $scriptsDir\connect_server.ps1    # 连接服务器"
Write-Host "    - $scriptsDir\upload_code.ps1       # 上传代码"
Write-Host "    - $scriptsDir\download_results.ps1  # 下载结果"
Write-Host "    - $scriptsDir\view_log.ps1          # 查看日志"
Write-Host "    - $scriptsDir\analyze_results.ps1   # 分析结果"

Write-Host ""

# =============================================================================
# Step 6: VSCode Remote-SSH 扩展
# =============================================================================

Write-Host "[Step 6] VSCode Remote-SSH 配置..."
Write-Host ""
Write-Host "  请在 VSCode 中安装 'Remote - SSH' 扩展:"
Write-Host "  1. 打开 VSCode"
Write-Host "  2. 按 Ctrl+Shift+X 打开扩展市场"
Write-Host "  3. 搜索 'Remote - SSH'"
Write-Host "  4. 点击安装"
Write-Host ""
Write-Host "  安装后，按 F1 输入 'Remote-SSH: Connect to Host'"
Write-Host "  选择 'cmcc-npu' 即可连接到服务器"

Write-Host ""

# =============================================================================
# 完成
# =============================================================================

Write-Host "=============================================="
Write-Host "本地设置完成!"
Write-Host "=============================================="
Write-Host ""
Write-Host "下一步操作:"
Write-Host ""
Write-Host "1. 在 VSCode 中安装 Remote-SSH 扩展"
Write-Host ""
Write-Host "2. 上传代码到服务器:"
Write-Host "   .\local_scripts\upload_code.ps1"
Write-Host ""
Write-Host "3. 连接到服务器并运行部署脚本:"
Write-Host "   .\local_scripts\connect_server.ps1"
Write-Host "   然后在服务器上运行:"
Write-Host "   chmod +x ~/alphaqubit/ALPHAQUBIT/remote_scripts/*.sh"
Write-Host "   ~/alphaqubit/ALPHAQUBIT/remote_scripts/setup_remote_env.sh"
Write-Host ""
Write-Host "4. 运行训练:"
Write-Host "   ~/alphaqubit/run_quick_test.sh      # 快速测试"
Write-Host "   ~/alphaqubit/run_full_training.sh   # 完整训练"
Write-Host ""
Write-Host "5. 下载并分析结果:"
Write-Host "   .\local_scripts\download_results.ps1"
Write-Host "   .\local_scripts\analyze_results.ps1"
Write-Host ""
Write-Host "=============================================="
