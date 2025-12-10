# 6 个稳定子代码位置快速参考

## 📂 文件位置

```
C:\Users\Lenovo\software\ALPHAQUBIT\my_noise_model\circuit_builder.py
```

## 📍 关键代码位置

### 1. 主函数：`_measure_all_stabilizers()` 
**行数**: 437-469

这个函数实现了所有 6 个稳定子的测量：

```python
def _measure_all_stabilizers(
    self,
    circuit: stim.Circuit,
    data_qubits: List[int],
    ancilla_qubits: List[int]
):
    """测量所有 6 个稳定子。"""
    
    circuit.append_operation("TICK", [])
    
    # Z 稳定子测量（3 个）
    self._measure_z_stabilizer(circuit, data_qubits, ancilla_qubits[0], [0,1,2,3])  # S1
    self._measure_z_stabilizer(circuit, data_qubits, ancilla_qubits[1], [0,1,4,5])  # S2
    self._measure_z_stabilizer(circuit, data_qubits, ancilla_qubits[2], [0,2,4,6])  # S3
    
    circuit.append_operation("TICK", [])
    
    # X 稳定子测量（3 个）
    self._measure_x_stabilizer(circuit, data_qubits, ancilla_qubits[3], [0,1,2,3])  # S4
    self._measure_x_stabilizer(circuit, data_qubits, ancilla_qubits[4], [0,1,4,5])  # S5
    self._measure_x_stabilizer(circuit, data_qubits, ancilla_qubits[5], [0,2,4,6])  # S6
    
    circuit.append_operation("TICK", [])
```

### 2. Z 稳定子测量：`_measure_z_stabilizer()`
**行数**: 390-410

```python
def _measure_z_stabilizer(
    self,
    circuit: stim.Circuit,
    data_qubits: List[int],
    ancilla: int,
    z_support: List[int]
):
    """测量一个 Z 型稳定子。"""
    # 辅助比特初始化为 |0⟩
    circuit.append_operation("R", [ancilla])
    
    # CNOT 从数据比特到辅助比特（测量 Z）
    for idx in z_support:
        circuit.append_operation("CX", [data_qubits[idx], ancilla])
    
    # 测量辅助比特
    circuit.append_operation("M", [ancilla])
```

### 3. X 稳定子测量：`_measure_x_stabilizer()`
**行数**: 412-435

```python
def _measure_x_stabilizer(
    self,
    circuit: stim.Circuit,
    data_qubits: List[int],
    ancilla: int,
    x_support: List[int]
):
    """测量一个 X 型稳定子。"""
    # 辅助比特初始化为 |+⟩
    circuit.append_operation("R", [ancilla])
    circuit.append_operation("H", [ancilla])
    
    # CNOT 从辅助比特到数据比特（测量 X）
    for idx in x_support:
        circuit.append_operation("CX", [ancilla, data_qubits[idx]])
    
    # 测量前应用 H
    circuit.append_operation("H", [ancilla])
    circuit.append_operation("M", [ancilla])
```

## 📊 6 个稳定子详细对应

| 稳定子 | 类型 | 代码行 | 支撑比特 | 辅助比特 |
|--------|------|--------|----------|----------|
| S1 | Z型 | 460 | [0,1,2,3] | ancilla_qubits[0] (索引7) |
| S2 | Z型 | 461 | [0,1,4,5] | ancilla_qubits[1] (索引8) |
| S3 | Z型 | 462 | [0,2,4,6] | ancilla_qubits[2] (索引9) |
| S4 | X型 | 467 | [0,1,2,3] | ancilla_qubits[3] (索引10) |
| S5 | X型 | 468 | [0,1,4,5] | ancilla_qubits[4] (索引11) |
| S6 | X型 | 469 | [0,2,4,6] | ancilla_qubits[5] (索引12) |

## 🔍 如何查看代码

### 方法 1: 在 VS Code 中打开
```powershell
code C:\Users\Lenovo\software\ALPHAQUBIT\my_noise_model\circuit_builder.py
```
然后按 `Ctrl+G` 输入行号跳转：
- 输入 `437` 查看主函数
- 输入 `390` 查看 Z 稳定子函数
- 输入 `412` 查看 X 稳定子函数

### 方法 2: 使用 PowerShell 查看特定行
```powershell
# 查看主函数（437-469行）
Get-Content my_noise_model\circuit_builder.py | Select-Object -Skip 436 -First 33

# 查看 Z 稳定子函数（390-410行）
Get-Content my_noise_model\circuit_builder.py | Select-Object -Skip 389 -First 21

# 查看 X 稳定子函数（412-435行）
Get-Content my_noise_model\circuit_builder.py | Select-Object -Skip 411 -First 24
```

### 方法 3: 搜索函数名
```powershell
# 在文件中搜索
Select-String -Path my_noise_model\circuit_builder.py -Pattern "_measure_all_stabilizers"
```

## 📝 代码调用示例

```python
from my_noise_model.circuit_builder import SteaneCodeCircuitBuilder

# 创建构建器（会自动使用 6 个稳定子）
builder = SteaneCodeCircuitBuilder(
    logical_state='0',
    include_stabilizers=True,  # 启用稳定子测量
    num_syndrome_rounds=1
)

# 构建电路
circuit = builder.build_ideal_circuit()

# 打印电路信息
print(f"量子比特数: {circuit.num_qubits}")  # 输出: 13
print(f"测量次数: {circuit.num_measurements}")
```

## 🎯 快速导航命令

```powershell
# 跳转到文件目录
cd C:\Users\Lenovo\software\ALPHAQUBIT\my_noise_model

# 用编辑器打开
code circuit_builder.py

# 或用记事本打开
notepad circuit_builder.py
```

---

**更新日期**: 2024-11-03  
**验证状态**: ✅ 所有 6 个稳定子已确认实现
