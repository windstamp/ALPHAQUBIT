# =============================================================================
# Pull AlphaQubit Training Results from S3 to Local (PowerShell)
# Only pulls analysis files (results, logs, plots) - NOT model weights
# =============================================================================
#
# Usage:
#   .\pull_results_from_s3.ps1                    # Pull latest results
#   .\pull_results_from_s3.ps1 -List              # List available results
#   .\pull_results_from_s3.ps1 -Timestamp "20241230_123456"  # Pull specific
#   .\pull_results_from_s3.ps1 -All               # Pull all results
#   .\pull_results_from_s3.ps1 -IncludeModels     # Include model files (large!)
#
# =============================================================================

param(
    [switch]$List,
    [switch]$All,
    [string]$Timestamp,
    [switch]$Help,
    [switch]$IncludeModels
)

# Configuration
$S3_REMOTE = "nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
$PROJECT_NAME = "ALPHAQUBIT"
$S3_RESULTS_PATH = "$S3_REMOTE/$PROJECT_NAME/training_results"
$LOCAL_RESULTS_DIR = ".\s3_results"

# Files to exclude (large model files)
$EXCLUDE_PATTERNS = @(
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.ckpt",
    "*.h5",
    "*.npy",
    "*.npz"
)

Write-Host "==============================================" -ForegroundColor Blue
Write-Host "  Pull AlphaQubit Results from S3" -ForegroundColor Blue
Write-Host "==============================================" -ForegroundColor Blue
Write-Host "S3 Source: $S3_RESULTS_PATH"
Write-Host "Local Destination: $LOCAL_RESULTS_DIR"
Write-Host ""

# Check rclone
if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    Write-Host "rclone not found!" -ForegroundColor Red
    Write-Host "Install rclone first: https://rclone.org/downloads/"
    exit 1
}

# Help
if ($Help) {
    Write-Host "Usage:"
    Write-Host "  .\pull_results_from_s3.ps1                    Pull latest results (analysis files only)"
    Write-Host "  .\pull_results_from_s3.ps1 -List              List available results"
    Write-Host "  .\pull_results_from_s3.ps1 -Timestamp X       Pull specific timestamp"
    Write-Host "  .\pull_results_from_s3.ps1 -All               Pull all timestamps"
    Write-Host "  .\pull_results_from_s3.ps1 -IncludeModels     Also download model weights (large!)"
    Write-Host ""
    Write-Host "By default excludes: .pth, .bin, .safetensors, .pt, .ckpt, .h5, .npy, .npz"
    exit 0
}

# List results
if ($List) {
    Write-Host "Available results on S3:" -ForegroundColor Blue
    $results = rclone lsd $S3_RESULTS_PATH 2>$null
    if ($results) {
        $results | ForEach-Object {
            $folder = ($_ -split '\s+')[-1]
            Write-Host "   $folder"
        }
    }
    else {
        Write-Host "   No results found"
    }
    exit 0
}

# Build filter arguments
function Get-RcloneFilterArgs {
    $filterArgs = @()
    if (-not $script:IncludeModels) {
        foreach ($pattern in $script:EXCLUDE_PATTERNS) {
            $filterArgs += "--exclude"
            $filterArgs += $pattern
        }
    }
    return $filterArgs
}

# Get latest timestamp
function Get-LatestTimestamp {
    $results = rclone lsd $script:S3_RESULTS_PATH 2>$null
    if ($results) {
        $lastLine = $results | Select-Object -Last 1
        $latest = ($lastLine -split '\s+')[-1]
        return $latest
    }
    return $null
}

# Pull function
function Pull-Results {
    param([string]$ts)
    
    $dest = Join-Path $script:LOCAL_RESULTS_DIR $ts
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    
    Write-Host "Downloading: $ts" -ForegroundColor Yellow
    Write-Host "   Destination: $dest"
    
    if (-not $script:IncludeModels) {
        Write-Host "   Mode: Analysis files only (excluding model weights)" -ForegroundColor Cyan
    }
    else {
        Write-Host "   Mode: All files including models" -ForegroundColor Magenta
    }
    Write-Host ""
    
    $filterArgs = Get-RcloneFilterArgs
    $source = "$script:S3_RESULTS_PATH/$ts"
    
    if ($filterArgs.Count -gt 0) {
        & rclone copy $source $dest --progress --transfers 4 @filterArgs
    }
    else {
        & rclone copy $source $dest --progress --transfers 4
    }
    
    Write-Host ""
    Write-Host "Download complete!" -ForegroundColor Green
    Write-Host "   Results saved to: $dest"
}

# Main logic
if ($All) {
    Write-Host "Downloading ALL results (analysis files only)..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $LOCAL_RESULTS_DIR | Out-Null
    
    $filterArgs = Get-RcloneFilterArgs
    
    if ($filterArgs.Count -gt 0) {
        & rclone copy $S3_RESULTS_PATH $LOCAL_RESULTS_DIR --progress --transfers 4 @filterArgs
    }
    else {
        & rclone copy $S3_RESULTS_PATH $LOCAL_RESULTS_DIR --progress --transfers 4
    }
    
    Write-Host "All results downloaded to: $LOCAL_RESULTS_DIR" -ForegroundColor Green
}
elseif ($Timestamp) {
    Pull-Results -ts $Timestamp
}
else {
    # Pull latest
    $latest = Get-LatestTimestamp
    if (-not $latest) {
        Write-Host "No results found on S3" -ForegroundColor Red
        exit 1
    }
    Write-Host "Latest: $latest"
    Pull-Results -ts $latest
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Done!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
