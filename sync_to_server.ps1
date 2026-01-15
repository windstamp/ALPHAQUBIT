# AlphaQubit Auto-Sync Script
# Syncs local changes to S3, then you can pull on server

param(
    [switch]$Watch,      # Continuous watch mode
    [switch]$Pull,       # Pull from S3 instead of push
    [int]$Interval = 10  # Seconds between syncs in watch mode
)

$LocalDir = "c:\Users\Lenovo\software\ALPHAQUBIT"
$S3Bucket = "nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/"
$Excludes = @(".git/**", "*.pyc", "__pycache__/**", "*.tar.gz", "trained_models/**", "*.log")

$ExcludeArgs = ($Excludes | ForEach-Object { "--exclude", $_ }) -join " "

function Sync-ToS3 {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Syncing to S3..." -ForegroundColor Cyan
    $files = @(
        "train_parallel_models.py",
        "train_8npu_simple.py",
        "train_full_pipeline.py",
        "run_training_pipeline.sh"
    )
    foreach ($f in $files) {
        $path = Join-Path $LocalDir $f
        if (Test-Path $path) {
            rclone copy $path $S3Bucket --s3-no-check-bucket
            Write-Host "  Uploaded: $f" -ForegroundColor Green
        }
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Sync complete!" -ForegroundColor Green
}

function Sync-FromS3 {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pulling from S3..." -ForegroundColor Cyan
    rclone copy $S3Bucket $LocalDir --s3-no-check-bucket --exclude "*.npz" --exclude "*.pth" --exclude "trained_models/**"
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pull complete!" -ForegroundColor Green
}

if ($Pull) {
    Sync-FromS3
    exit
}

if ($Watch) {
    Write-Host "Starting watch mode (Ctrl+C to stop)..." -ForegroundColor Yellow
    Write-Host "Syncing every $Interval seconds" -ForegroundColor Yellow
    while ($true) {
        Sync-ToS3
        Start-Sleep -Seconds $Interval
    }
} else {
    Sync-ToS3
}
