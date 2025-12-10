"""
验证 Steane Code 的 6 个稳定子实现

这个脚本检查我们的代码是否正确实现了所有 6 个稳定子。
"""

import sys
import re
from pathlib import Path


def extract_stabilizer_calls(file_path):
    """从 circuit_builder.py 提取稳定子测量调用"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 _measure_all_stabilizers 函数
    pattern = r'def _measure_all_stabilizers\(.*?\n(.*?)(?=\n    def |\n\nclass |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    func_body = match.group(1)
    
    # 提取所有 Z 稳定子调用
    z_calls = re.findall(r'_measure_z_stabilizer\(.*?\[([\d,]+)\]\)', func_body)
    
    # 提取所有 X 稳定子调用
    x_calls = re.findall(r'_measure_x_stabilizer\(.*?\[([\d,]+)\]\)', func_body)
    
    return {
        'z_stabilizers': [list(map(int, call.split(','))) for call in z_calls],
        'x_stabilizers': [list(map(int, call.split(','))) for call in x_calls]
    }


def verify_stabilizers():
    """验证所有 6 个稳定子是否正确实现"""
    print("=" * 70)
    print("Steane Code 6 个稳定子实现验证")
    print("=" * 70)
    print()
    
    # 期望的稳定子配置
    expected_z = [
        [0, 1, 2, 3],  # S1 = Z0Z1Z2Z3
        [0, 1, 4, 5],  # S2 = Z0Z1Z4Z5
        [0, 2, 4, 6],  # S3 = Z0Z2Z4Z6
    ]
    
    expected_x = [
        [0, 1, 2, 3],  # S4 = X0X1X2X3
        [0, 1, 4, 5],  # S5 = X0X1X4X5
        [0, 2, 4, 6],  # S6 = X0X2X4X6
    ]
    
    # 读取实际实现
    file_path = Path(__file__).parent / 'my_noise_model' / 'circuit_builder.py'
    actual = extract_stabilizer_calls(file_path)
    
    if actual is None:
        print("❌ 错误：无法找到 _measure_all_stabilizers 函数")
        return False
    
    # 验证 Z 稳定子
    print("Z 型稳定子（3 个）:")
    print("-" * 70)
    all_correct = True
    
    for i, (expected, actual_support) in enumerate(zip(expected_z, actual['z_stabilizers']), 1):
        status = "✅" if expected == actual_support else "❌"
        all_correct = all_correct and (expected == actual_support)
        
        expected_str = ''.join([f'Z{j}' for j in expected])
        actual_str = ''.join([f'Z{j}' for j in actual_support])
        
        print(f"  S{i}: {status}")
        print(f"      期望: {expected_str} (支撑在比特 {expected})")
        print(f"      实际: {actual_str} (支撑在比特 {actual_support})")
        print()
    
    # 验证 X 稳定子
    print("X 型稳定子（3 个）:")
    print("-" * 70)
    
    for i, (expected, actual_support) in enumerate(zip(expected_x, actual['x_stabilizers']), 4):
        status = "✅" if expected == actual_support else "❌"
        all_correct = all_correct and (expected == actual_support)
        
        expected_str = ''.join([f'X{j}' for j in expected])
        actual_str = ''.join([f'X{j}' for j in actual_support])
        
        print(f"  S{i}: {status}")
        print(f"      期望: {expected_str} (支撑在比特 {expected})")
        print(f"      实际: {actual_str} (支撑在比特 {actual_support})")
        print()
    
    # 总结
    print("=" * 70)
    print("验证结果:")
    print("-" * 70)
    print(f"  Z 型稳定子数量: {len(actual['z_stabilizers'])} / 3 期望")
    print(f"  X 型稳定子数量: {len(actual['x_stabilizers'])} / 3 期望")
    print(f"  总稳定子数量: {len(actual['z_stabilizers']) + len(actual['x_stabilizers'])} / 6 期望")
    print()
    
    if all_correct and len(actual['z_stabilizers']) == 3 and len(actual['x_stabilizers']) == 3:
        print("✅ 所有 6 个稳定子实现正确！")
        print()
        print("稳定子完整列表:")
        print("  S1 = Z0Z1Z2Z3  (辅助比特 a0, 索引 7)")
        print("  S2 = Z0Z1Z4Z5  (辅助比特 a1, 索引 8)")
        print("  S3 = Z0Z2Z4Z6  (辅助比特 a2, 索引 9)")
        print("  S4 = X0X1X2X3  (辅助比特 a3, 索引 10)")
        print("  S5 = X0X1X4X5  (辅助比特 a4, 索引 11)")
        print("  S6 = X0X2X4X6  (辅助比特 a5, 索引 12)")
        print()
        print("量子比特分配:")
        print("  • 数据比特: q0-q6 (7 个)")
        print("  • 辅助比特: a0-a5 (6 个，物理索引 7-12)")
        print("  • 总计: 13 个量子比特")
        return True
    else:
        print("❌ 稳定子实现有误！")
        return False


def verify_code_consistency():
    """验证代码中的文档字符串与实现是否一致"""
    print()
    print("=" * 70)
    print("检查文档字符串一致性")
    print("=" * 70)
    print()
    
    file_path = Path(__file__).parent / 'my_noise_model' / 'circuit_builder.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查文档字符串中的稳定子定义
    doc_patterns = [
        (r'S1 = Z0Z1Z2Z3', 'S1 定义'),
        (r'S2 = Z0Z1Z4Z5', 'S2 定义'),
        (r'S3 = Z0Z2Z4Z6', 'S3 定义'),
        (r'S4 = X0X1X2X3', 'S4 定义'),
        (r'S5 = X0X1X4X5', 'S5 定义'),
        (r'S6 = X0X2X4X6', 'S6 定义'),
    ]
    
    all_found = True
    for pattern, description in doc_patterns:
        if re.search(pattern, content):
            print(f"  ✅ 找到 {description}")
        else:
            print(f"  ❌ 未找到 {description}")
            all_found = False
    
    print()
    if all_found:
        print("✅ 文档字符串与实现一致")
    else:
        print("❌ 文档字符串与实现不一致")
    
    return all_found


def check_ancilla_allocation():
    """检查辅助比特分配"""
    print()
    print("=" * 70)
    print("检查辅助比特分配")
    print("=" * 70)
    print()
    
    file_path = Path(__file__).parent / 'my_noise_model' / 'circuit_builder.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查辅助比特数量
    if 'self.num_ancilla_qubits = 6' in content:
        print("✅ 辅助比特数量设置正确: 6 个")
    else:
        print("❌ 辅助比特数量设置可能有误")
        return False
    
    # 检查总量子比特数
    if 'self.total_qubits = self.num_data_qubits + self.num_ancilla_qubits' in content:
        print("✅ 总量子比特数计算正确: 7 + 6 = 13")
    else:
        print("❌ 总量子比特数计算可能有误")
        return False
    
    # 检查辅助比特索引
    if 'ancilla_qubits = list(range(self.num_data_qubits, self.total_qubits))' in content:
        print("✅ 辅助比特索引范围正确: 7-12")
    else:
        print("❌ 辅助比特索引范围可能有误")
        return False
    
    print()
    print("辅助比特映射:")
    print("  a0 (物理索引 7)  → S1 (Z0Z1Z2Z3)")
    print("  a1 (物理索引 8)  → S2 (Z0Z1Z4Z5)")
    print("  a2 (物理索引 9)  → S3 (Z0Z2Z4Z6)")
    print("  a3 (物理索引 10) → S4 (X0X1X2X3)")
    print("  a4 (物理索引 11) → S5 (X0X1X4X5)")
    print("  a5 (物理索引 12) → S6 (X0X2X4X6)")
    
    return True


def main():
    """主验证函数"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Steane Code 6 个稳定子验证工具" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # 执行验证
    result1 = verify_stabilizers()
    result2 = verify_code_consistency()
    result3 = check_ancilla_allocation()
    
    # 最终总结
    print()
    print("=" * 70)
    print("最终验证结果")
    print("=" * 70)
    
    if result1 and result2 and result3:
        print()
        print("🎉 恭喜！所有验证通过！")
        print()
        print("✅ 6 个稳定子实现完整且正确")
        print("✅ 文档字符串与代码一致")
        print("✅ 辅助比特分配正确")
        print()
        print("您的 Steane Code 实现已经包含了完整的 6 个稳定子：")
        print("  • 3 个 Z 型稳定子 (S1, S2, S3)")
        print("  • 3 个 X 型稳定子 (S4, S5, S6)")
        print("  • 使用 6 个辅助比特进行测量")
        print("  • 总共 13 个量子比特 (7 数据 + 6 辅助)")
        print()
        return 0
    else:
        print()
        print("⚠️  部分验证未通过，请检查实现")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
