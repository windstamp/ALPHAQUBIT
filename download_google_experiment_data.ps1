# Download Google Quantum AI Experiment Data from Zenodo
# Data for "Quantum error correction below the surface code threshold"
# DOI: 10.5281/zenodo.13273331
# Total size: 112.5 GB

$outputDir = "google_experiment_data"
$zenodoBase = "https://zenodo.org/records/13273331/files"

# Create output directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}

# Files to download
$files = @(
    @{
        Name = "google_105Q_surface_code_d3_d5_d7.zip"
        Size = "5.7 GB"
        MD5 = "21fa6ad35b395d838ebcdbc92e364a12"
        Description = "105 Qubit Surface Code (d3, d5, d7)"
    },
    @{
        Name = "google_72Q_surface_code_d3_d5_set1.zip"
        Size = "30.2 GB"
        MD5 = "7875d0fa37fab89e37cf5cd305114cef"
        Description = "72 Qubit Surface Code (d3, d5) Set 1 - Primary dataset"
    },
    @{
        Name = "google_72Q_surface_code_d3_d5_set2.zip"
        Size = "11.6 GB"
        MD5 = "cb331869b9fe7e95d6ec9c945d5278a6"
        Description = "72 Qubit Surface Code (d3, d5) Set 2"
    },
    @{
        Name = "google_72Q_repetition_code_d29.zip"
        Size = "65.0 GB"
        MD5 = "12c2cc1e5a924c273342b2cb5654394d"
        Description = "72 Qubit Repetition Code (d29) - Large dataset"
    }
)

Write-Host "=" * 70
Write-Host "Google Quantum AI Experiment Data Downloader"
Write-Host "Source: https://zenodo.org/records/13273331"
Write-Host "=" * 70
Write-Host ""

# Function to download with progress
function Download-FileWithProgress {
    param(
        [string]$Url,
        [string]$OutputPath
    )
    
    $webClient = New-Object System.Net.WebClient
    
    # Register event handler for progress
    $eventHandler = {
        param($sender, $e)
        $percent = $e.ProgressPercentage
        $received = [math]::Round($e.BytesReceived / 1MB, 2)
        Write-Progress -Activity "Downloading" -Status "$received MB downloaded ($percent%)" -PercentComplete $percent
    }
    
    Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -Action $eventHandler | Out-Null
    
    try {
        $webClient.DownloadFile($Url, $OutputPath)
    }
    finally {
        Write-Progress -Activity "Downloading" -Completed
        $webClient.Dispose()
    }
}

# Download each file
foreach ($file in $files) {
    $outputPath = Join-Path $outputDir $file.Name
    $url = "$zenodoBase/$($file.Name)?download=1"
    
    Write-Host ""
    Write-Host "File: $($file.Name)"
    Write-Host "Size: $($file.Size)"
    Write-Host "Description: $($file.Description)"
    
    # Check if file already exists
    if (Test-Path $outputPath) {
        Write-Host "Status: Already exists, skipping..." -ForegroundColor Yellow
        continue
    }
    
    Write-Host "Downloading from: $url"
    Write-Host ""
    
    try {
        # Use BITS transfer for large files (more reliable)
        Start-BitsTransfer -Source $url -Destination $outputPath -DisplayName $file.Name -Description "Downloading $($file.Description)"
        Write-Host "Status: Downloaded successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "BITS transfer failed, trying WebClient..." -ForegroundColor Yellow
        try {
            Download-FileWithProgress -Url $url -OutputPath $outputPath
            Write-Host "Status: Downloaded successfully!" -ForegroundColor Green
        }
        catch {
            Write-Host "Status: Download failed - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=" * 70
Write-Host "Download complete! Files are in: $outputDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Extract the zip files:"
Write-Host "   cd $outputDir"
Write-Host "   Expand-Archive -Path *.zip -DestinationPath . -Force"
Write-Host ""
Write-Host "2. The data will be used by the following scripts:"
Write-Host "   - run_create_all_samples.py"
Write-Host "   - make_all_pretraining_noise.py"
Write-Host "   - run_finetune_all.py"
Write-Host "=" * 70
