# Remote Server Quick Reference

## Server Paths
- **Remote work directory**: `/root/work/ALPHAQUBIT`
- **S3 bucket**: `nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT`

## Quick Commands

### Upload to S3 (from local Windows)
```powershell
cd c:\Users\Lenovo\software\ALPHAQUBIT
rclone sync . nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT --exclude ".git/**" --exclude "__pycache__/**" --exclude "*.pyc" --exclude ".pytest_cache/**" --progress
```

### On Remote Server

#### Pull code from S3
```bash
cd /root/work
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT ./ALPHAQUBIT --progress
```

#### Stop running processes
```bash
pkill -f 'python.*run_'
```

#### Run full pipeline (all 8 NPUs)
```bash
cd /root/work/ALPHAQUBIT
NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python run_complete_server_pipeline.py > pipeline.log 2>&1 &
```

#### Monitor progress
```bash
tail -f /root/work/ALPHAQUBIT/pipeline.log
```

#### Check running processes
```bash
ps aux | grep python
```

### Download results to local (from Windows)
```powershell
rclone sync nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/ALPHAQUBIT/output ./s3_results --progress
```
