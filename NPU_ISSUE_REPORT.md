# Kubernetes Pod 中 Ascend NPU 初始化失败问题报告

**日期：** 2024年12月24日

**主题：** ACL_PRECISION_MODE error 500001

---

## 环境信息

| 组件 | 版本 |
|------|------|
| NPU 芯片 | Ascend 910B2 × 8 |
| NPU 驱动 | 24.1.1 |
| CANN Toolkit | 8.2.RC1 |
| torch | 2.6.0+cpu |
| torch_npu | 2.6.0 |
| Python | 3.11 |
| 运行环境 | Kubernetes Pod (容器) |
| 操作系统 | BCLinux (kernel 4.19.90) |

---

## 问题描述

1. `npu-smi info` 可以正常检测到 8 个 NPU 设备，状态显示 OK
2. `torch.npu.is_available()` 返回 `True`
3. `torch.npu.device_count()` 返回 `8`
4. 但执行 `torch.randn(10).to('npu:0')` 时失败

---

## 错误信息

```
RuntimeError: SetPrecisionMode:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:155 
NPU function error: at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode), error code is 500001

[ERROR] ERR00100 PTA call acl api failed
[Error]: The internal ACL of the system is incorrect.

E49999: Inner Error!
E49999: AOE Failed to call InitCannKB
TraceBack:
- Failed to initialize TeConfigInfo
- [GraphOpt][InitializeInner][InitTbeFunc] Failed to init tbe
- [SubGraphOpt][PreCompileOp][InitAdapter] InitializeAdapter adapter [tbe_op_adapter] failed! Ret [4294967295]
- GEInitialize failed. ge result = 4294967295
- [Init][Compiler]Init compiler failed
- [Set][Options]OpCompileProcessor init failed!
```

---

## 已尝试的解决方法（均无效）

1. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
2. 设置 `ASCEND_PRECISION_MODE` 为 `must_keep_origin_dtype` / `allow_fp32_to_fp16` / `force_fp16`
3. 设置 `HCCL_WHITELIST_DISABLE=1`
4. 设置 `SOC_VERSION=Ascend910B2`
5. 升级 torch_npu 到 2.6.0.post3
6. 清除 `/tmp/ascend_*` 缓存
7. 修改 `/dev/davinci*` 权限为 666
8. 设置 `FORCE_USE_AICPU=1`
9. 设置 `GE_USE_STATIC_MEMORY=0`

---

## 环境变量（已正确设置）

```bash
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
ASCEND_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
ASCEND_SLOG_PRINT_TO_STDOUT=0
ASCEND_GLOBAL_LOG_LEVEL=3

# LD_LIBRARY_PATH 包含:
# /usr/local/Ascend/ascend-toolkit/latest/lib64
# /usr/local/Ascend/driver/lib64
# 等相关路径

# PYTHONPATH 包含:
# /usr/local/Ascend/ascend-toolkit/latest/python/site-packages
# /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
```

---

## 关键疑问

1. **版本兼容性：** CANN 8.2.RC1 与 torch_npu 2.6.0 是否兼容？
2. **容器配置：** 在 Kubernetes 容器中使用 NPU 是否需要额外配置？
3. **错误含义：** `ACL_PRECISION_MODE error 500001` 具体表示什么问题？
4. **TBE 初始化：** 如何正确配置 TBE 编译器在容器中的初始化？

---

## npu-smi 输出

```
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.1                   Version: 24.1.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 86.9        42                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3532 / 65536         |
+===========================+===============+====================================================+
| 1     910B2               | OK            | 93.7        43                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3400 / 65536         |
+===========================+===============+====================================================+
| 2     910B2               | OK            | 92.5        41                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3401 / 65536         |
+===========================+===============+====================================================+
| 3     910B2               | OK            | 90.7        43                0    / 0             |
| 0                         | 0000:02:00.0  | 0           0    / 0          3398 / 65536         |
+===========================+===============+====================================================+
| 4     910B2               | OK            | 93.6        41                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          3396 / 65536         |
+===========================+===============+====================================================+
| 5     910B2               | OK            | 92.0        41                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          3397 / 65536         |
+===========================+===============+====================================================+
| 6     910B2               | OK            | 91.3        41                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          3397 / 65536         |
+===========================+===============+====================================================+
| 7     910B2               | OK            | 95.7        43                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          3402 / 65536         |
+===========================+===============+====================================================+
```

---

## 设备权限

```bash
$ ls -la /dev/davinci*
crw-rw---- 1 1000 1000 235, 0 Dec 23 11:16 /dev/davinci0
crw-rw---- 1 1000 1000 235, 1 Dec 23 11:16 /dev/davinci1
crw-rw---- 1 1000 1000 235, 2 Dec 23 11:16 /dev/davinci2
crw-rw---- 1 1000 1000 235, 3 Dec 23 11:16 /dev/davinci3
crw-rw---- 1 1000 1000 235, 4 Dec 23 11:16 /dev/davinci4
crw-rw---- 1 1000 1000 235, 5 Dec 23 11:16 /dev/davinci5
crw-rw---- 1 1000 1000 235, 6 Dec 23 11:16 /dev/davinci6
crw-rw---- 1 1000 1000 235, 7 Dec 23 11:16 /dev/davinci7
crw-rw---- 1 1000 1000 236, 0 Dec 23 11:16 /dev/davinci_manager
```

---

## 容器信息

```bash
$ cat /proc/1/cgroup | head -5
13:pids:/kubepods/pod438d7532-cc1b-44fb-878a-64c4156a92ce/eb62a59daf1d6ac12f19c28243d5cb45d5122af6f301b3fe06e2682aa5d7c9f1
12:devices:/kubepods/pod438d7532-cc1b-44fb-878a-64c4156a92ce/...
11:cpu,cpuacct:/kubepods/pod438d7532-cc1b-44fb-878a-64c4156a92ce/...
```

---

## 测试代码

```python
import torch
import torch_npu

print('torch:', torch.__version__)        # 2.6.0+cpu
print('torch_npu:', torch_npu.__version__) # 2.6.0
print('NPU available:', torch.npu.is_available())  # True
print('NPU count:', torch.npu.device_count())      # 8

# 以下代码失败
x = torch.randn(10).to('npu:0')  # RuntimeError
```

---

## 期望帮助

1. 确认 CANN 8.2.RC1 + torch_npu 2.6.0 的正确配置方法
2. 提供在 Kubernetes 容器中使用 NPU 的最佳实践
3. 解释 error code 500001 的具体原因和解决方案

---

**联系人信息：**
- 姓名：（请填写）
- 公司：（请填写）
- 邮箱：（请填写）
- 电话：（请填写）
