# Rclone S3 Sync Guide

This document describes how to use rclone to sync data between local, S3, and remote locations.

## Prerequisites

1. Install rclone: https://rclone.org/install/
2. Configure your S3 remote (e.g., `s3remote`) and any other remotes using `rclone config`

## Common Commands

### 1. Local to S3

Upload files from your local machine to S3:

```bash
# Sync local folder to S3
rclone sync /path/to/local/folder s3remote:bucket-name/path --progress

# Copy (doesn't delete files at destination)
rclone copy /path/to/local/folder s3remote:bucket-name/path --progress

# Example with specific folder
rclone sync ./data s3remote:my-bucket/alphaqubit/data --progress
```

### 2. S3 to Remote

Sync files from S3 to another remote (e.g., Google Drive, another cloud storage):

```bash
# Sync S3 to remote
rclone sync s3remote:bucket-name/path remote:destination/path --progress

# Copy from S3 to remote
rclone copy s3remote:bucket-name/path remote:destination/path --progress

# Example: S3 to Google Drive
rclone sync s3remote:my-bucket/alphaqubit gdrive:backup/alphaqubit --progress
```

### 3. Remote to S3

Sync files from a remote location to S3:

```bash
# Sync remote to S3
rclone sync remote:source/path s3remote:bucket-name/path --progress

# Copy from remote to S3
rclone copy remote:source/path s3remote:bucket-name/path --progress

# Example: Google Drive to S3
rclone sync gdrive:data/alphaqubit s3remote:my-bucket/alphaqubit --progress
```

### 4. S3 to Local

Download files from S3 to your local machine:

```bash
# Sync S3 to local
rclone sync s3remote:bucket-name/path /path/to/local/folder --progress

# Copy from S3 to local
rclone copy s3remote:bucket-name/path /path/to/local/folder --progress

# Example with specific folder
rclone sync s3remote:my-bucket/alphaqubit/models ./models --progress
```

## Useful Options

| Option | Description |
|--------|-------------|
| `--progress` | Show progress during transfer |
| `--dry-run` | Test without making changes |
| `--verbose` | Increase logging verbosity |
| `--exclude "*.tmp"` | Exclude files matching pattern |
| `--include "*.pth"` | Include only files matching pattern |
| `--transfers 4` | Number of parallel transfers |
| `--checkers 8` | Number of checkers to run in parallel |
| `--bandwidth 10M` | Limit bandwidth to 10 MB/s |

## Examples with Options

```bash
# Dry run to see what would be transferred
rclone sync ./data s3remote:my-bucket/data --dry-run --progress

# Sync with bandwidth limit
rclone sync ./models s3remote:my-bucket/models --bandwidth 50M --progress

# Sync only .pth files
rclone sync ./models s3remote:my-bucket/models --include "*.pth" --progress

# Exclude temporary files
rclone sync ./data s3remote:my-bucket/data --exclude "*.tmp" --exclude "*.log" --progress
```

## Checking Remote Configuration

```bash
# List configured remotes
rclone listremotes

# List contents of S3 bucket
rclone ls s3remote:bucket-name

# List directories in S3 bucket
rclone lsd s3remote:bucket-name

# Check config
rclone config show
```

## Troubleshooting

1. **Authentication errors**: Run `rclone config` to reconfigure your remote
2. **Permission denied**: Check your S3 bucket policies and IAM permissions
3. **Slow transfers**: Increase `--transfers` and `--checkers` values
4. **Connection issues**: Check your network and firewall settings

## Setting Up S3 Remote

```bash
rclone config

# Then follow prompts:
# n) New remote
# name> s3remote
# Storage> s3
# provider> AWS (or other S3-compatible)
# Enter AWS credentials or use environment variables
```
