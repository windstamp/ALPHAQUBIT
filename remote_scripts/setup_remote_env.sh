#!/bin/bash#!/bin/bash

# =============================================================================# =============================================================================

# AlphaQubit 远程NPU服务器部署脚本# AlphaQubit 远程NPU服务器部署脚本

# 中国移动华为昇腾开发环境# 中国移动华为昇腾开发环境

# =============================================================================# =============================================================================

# # 

# 使用方法:# 使用方法:

#   1. 上传此脚本到服务器#   1. 上传此脚本到服务器

#   2. chmod +x setup_remote_env.sh#   2. chmod +x setup_remote_env.sh

#   3. bash setup_remote_env.sh#   3. ./setup_remote_env.sh

##

# =============================================================================# =============================================================================



# 不使用 set -e，改为手动处理错误set -e  # 遇到错误立即退出



echo "=============================================="echo "=============================================="

echo "AlphaQubit 远程NPU服务器环境部署"echo "AlphaQubit 远程NPU服务器环境部署"

echo "=============================================="echo "=============================================="

echo ""echo ""



# =============================================================================# =============================================================================

# 配置变量 (根据实际情况修改)# 配置变量 (根据实际情况修改)

# =============================================================================# =============================================================================



WORK_DIR="${HOME}/alphaqubit"WORK_DIR="${HOME}/alphaqubit"

CONDA_ENV_NAME="alphaqubit"CONDA_ENV_NAME="alphaqubit"

PYTHON_VERSION="3.10"PYTHON_VERSION="3.10"

REPO_URL="https://github.com/xuda1979/ALPHAQUBIT.git"REPO_URL="https://github.com/xuda1979/ALPHAQUBIT.git"

MINICONDA_DIR="${HOME}/miniconda3"

# =============================================================================

# =============================================================================# Step 1: 检查NPU环境

# 辅助函数# =============================================================================

# =============================================================================

echo "[Step 1] 检查NPU环境..."

setup_conda_path() {

    # 设置conda路径和环境# 检查华为昇腾NPU

    export PATH="${MINICONDA_DIR}/bin:$PATH"if command -v npu-smi &> /dev/null; then

    if [ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then    echo "  ✓ 检测到华为昇腾NPU"

        source "${MINICONDA_DIR}/etc/profile.d/conda.sh"    npu-smi info

    fielse

}    echo "  ! 未检测到npu-smi，可能NPU驱动未安装"

fi

check_conda() {

    # 检查conda是否可用# 检查CANN toolkit

    if command -v conda &> /dev/null; thenif [ -d "/usr/local/Ascend" ]; then

        return 0    echo "  ✓ 检测到CANN toolkit: /usr/local/Ascend"

    else    # 设置环境变量

        return 1    export ASCEND_HOME=/usr/local/Ascend

    fi    export PATH=$ASCEND_HOME/bin:$PATH

}    export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH

else

# =============================================================================    echo "  ! 未检测到CANN toolkit"

# Step 1: 检查NPU环境fi

# =============================================================================

echo ""

echo "[Step 1] 检查NPU环境..."

# =============================================================================

# 检查华为昇腾NPU# Step 2: 创建工作目录

if command -v npu-smi &> /dev/null; then# =============================================================================

    echo "  ✓ 检测到华为昇腾NPU"

    npu-smi infoecho "[Step 2] 创建工作目录..."

else

    echo "  ! 未检测到npu-smi，可能NPU驱动未安装"mkdir -p ${WORK_DIR}

ficd ${WORK_DIR}



# 检查CANN toolkitecho "  ✓ 工作目录: ${WORK_DIR}"

if [ -d "/usr/local/Ascend" ]; thenecho ""

    echo "  ✓ 检测到CANN toolkit: /usr/local/Ascend"

    # 设置环境变量# =============================================================================

    export ASCEND_HOME=/usr/local/Ascend# Step 3: 设置Conda环境

    export PATH=$ASCEND_HOME/bin:$PATH# =============================================================================

    export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH

elseecho "[Step 3] 设置Conda环境..."

    echo "  ! 未检测到CANN toolkit"

fi# 检查conda是否存在

if command -v conda &> /dev/null; then

echo ""    echo "  ✓ Conda已安装"

elif [ -d "${HOME}/miniconda3" ]; then

# =============================================================================    echo "  ! Conda命令未找到，但检测到 ${HOME}/miniconda3 目录"

# Step 2: 创建工作目录    echo "  尝试使用现有安装..."

# =============================================================================    export PATH="${HOME}/miniconda3/bin:$PATH"

    source "${HOME}/miniconda3/etc/profile.d/conda.sh"

echo "[Step 2] 创建工作目录..."    

    # 再次检查

mkdir -p ${WORK_DIR}    if command -v conda &> /dev/null; then

cd ${WORK_DIR}        echo "  ✓ 已激活现有Conda安装"

        conda init bash 2>/dev/null || true

echo "  ✓ 工作目录: ${WORK_DIR}"    else

echo ""        echo "  ! 无法激活 ${HOME}/miniconda3 中的Conda"

        exit 1

# =============================================================================    fi

# Step 3: 设置Conda环境else

# =============================================================================    echo "  ! Conda未安装，尝试安装Miniconda..."

    

echo "[Step 3] 设置Conda环境..."    # 下载并安装Miniconda (使用 -u 选项更新现有安装)

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

CONDA_READY=false    bash miniconda.sh -b -u -p ${HOME}/miniconda3

    rm -f miniconda.sh

# 方案1: 检查conda命令是否已经可用    

if check_conda; then    # 初始化conda

    echo "  ✓ Conda已安装并可用"    source "${HOME}/miniconda3/etc/profile.d/conda.sh"

    CONDA_READY=true    ${HOME}/miniconda3/bin/conda init bash

fi    export PATH="${HOME}/miniconda3/bin:$PATH"

    

# 方案2: 检查miniconda3目录是否存在，尝试激活    echo "  ✓ Miniconda安装完成"

if [ "$CONDA_READY" = false ] && [ -d "${MINICONDA_DIR}" ]; thenfi

    echo "  检测到 ${MINICONDA_DIR} 目录，尝试激活..."

    setup_conda_path# 创建conda环境

    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then

    if check_conda; then    echo "  ✓ Conda环境 '${CONDA_ENV_NAME}' 已存在"

        echo "  ✓ 已激活现有Conda安装"else

        # 初始化bash (忽略错误)    echo "  创建Conda环境 '${CONDA_ENV_NAME}'..."

        conda init bash 2>/dev/null || true    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

        CONDA_READY=true    echo "  ✓ Conda环境创建完成"

    elsefi

        echo "  ! 现有安装已损坏，将重新安装..."

        rm -rf "${MINICONDA_DIR}"# 激活环境

    fisource $(conda info --base)/etc/profile.d/conda.sh

ficonda activate ${CONDA_ENV_NAME}



# 方案3: 全新安装Minicondaecho "  ✓ 已激活环境: ${CONDA_ENV_NAME}"

if [ "$CONDA_READY" = false ]; thenecho ""

    echo "  安装Miniconda..."

    # =============================================================================

    cd ${WORK_DIR}# Step 4: 克隆/更新代码仓库

    # =============================================================================

    # 下载Miniconda

    if [ -f "miniconda.sh" ]; thenecho "[Step 4] 获取代码仓库..."

        rm -f miniconda.sh

    fiif [ -d "${WORK_DIR}/ALPHAQUBIT" ]; then

        echo "  代码仓库已存在，更新中..."

    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh    cd ${WORK_DIR}/ALPHAQUBIT

        git pull origin main

    if [ $? -ne 0 ]; thenelse

        echo "  ! Miniconda下载失败"    echo "  克隆代码仓库..."

        exit 1    cd ${WORK_DIR}

    fi    git clone ${REPO_URL}

        cd ${WORK_DIR}/ALPHAQUBIT

    # 安装Miniconda (-b静默安装, -u更新现有安装, -p指定路径)fi

    bash miniconda.sh -b -u -p ${MINICONDA_DIR}

    echo "  ✓ 代码已就绪: ${WORK_DIR}/ALPHAQUBIT"

    if [ $? -ne 0 ]; thenecho ""

        echo "  ! Miniconda安装失败"

        exit 1# =============================================================================

    fi# Step 5: 安装Python依赖

    # =============================================================================

    rm -f miniconda.sh

    echo "[Step 5] 安装Python依赖..."

    # 设置路径并激活

    setup_conda_pathcd ${WORK_DIR}/ALPHAQUBIT

    

    # 初始化bash# 安装基础依赖

    conda init bash 2>/dev/null || truepip install --upgrade pip

    

    if check_conda; then# 安装PyTorch (NPU版本)

        echo "  ✓ Miniconda安装完成"echo "  安装PyTorch NPU版本..."

        CONDA_READY=true

    else# 华为昇腾PyTorch安装 (根据CANN版本选择)

        echo "  ! Conda安装后仍不可用"# 方法1: 使用华为官方源

        exit 1pip install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

    fipip install torch_npu -i https://pypi.tuna.tsinghua.edu.cn/simple

fi

# 安装其他依赖

# 确保conda已激活echo "  安装项目依赖..."

setup_conda_pathpip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple



# 更新conda# 安装stim (量子纠错模拟器)

echo "  更新conda..."pip install stim -i https://pypi.tuna.tsinghua.edu.cn/simple

conda update -n base -c defaults conda -y 2>/dev/null || true

echo "  ✓ 依赖安装完成"

# 创建或更新conda环境echo ""

echo "  检查conda环境 '${CONDA_ENV_NAME}'..."

# =============================================================================

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then# Step 6: 验证安装

    echo "  ✓ Conda环境 '${CONDA_ENV_NAME}' 已存在"# =============================================================================

else

    echo "  创建Conda环境 '${CONDA_ENV_NAME}'..."echo "[Step 6] 验证安装..."

    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

    python << 'EOF'

    if [ $? -ne 0 ]; thenimport sys

        echo "  ! Conda环境创建失败"print(f"Python版本: {sys.version}")

        exit 1

    fi# 检查PyTorch

    echo "  ✓ Conda环境创建完成"try:

fi    import torch

    print(f"PyTorch版本: {torch.__version__}")

# 激活环境    

echo "  激活环境..."    # 检查NPU

conda activate ${CONDA_ENV_NAME}    try:

        import torch_npu

if [ $? -ne 0 ]; then        if torch.npu.is_available():

    echo "  ! 环境激活失败，尝试备用方法..."            print(f"NPU可用: {torch.npu.device_count()} 个设备")

    source activate ${CONDA_ENV_NAME} 2>/dev/null || {            print(f"NPU设备: {torch.npu.get_device_name(0)}")

        echo "  ! 无法激活conda环境"        else:

        exit 1            print("NPU不可用，将使用CPU")

    }    except ImportError:

fi        print("torch_npu未安装，检查CUDA...")

        if torch.cuda.is_available():

echo "  ✓ 已激活环境: ${CONDA_ENV_NAME}"            print(f"CUDA可用: {torch.cuda.device_count()} 个GPU")

echo "  Python路径: $(which python)"        else:

echo ""            print("将使用CPU训练")

except ImportError as e:

# =============================================================================    print(f"PyTorch导入失败: {e}")

# Step 4: 克隆/更新代码仓库

# =============================================================================# 检查stim

try:

echo "[Step 4] 获取代码仓库..."    import stim

    print(f"Stim版本: {stim.__version__}")

cd ${WORK_DIR}except ImportError as e:

    print(f"Stim导入失败: {e}")

if [ -d "${WORK_DIR}/ALPHAQUBIT" ]; then

    echo "  代码仓库已存在，更新中..."# 检查numpy

    cd ${WORK_DIR}/ALPHAQUBITtry:

    git fetch origin 2>/dev/null || true    import numpy as np

    git pull origin main 2>/dev/null || echo "  ! Git pull失败，使用现有代码"    print(f"NumPy版本: {np.__version__}")

elseexcept ImportError as e:

    echo "  克隆代码仓库..."    print(f"NumPy导入失败: {e}")

    git clone ${REPO_URL}

    print("\n验证完成!")

    if [ $? -ne 0 ]; thenEOF

        echo "  ! Git clone失败"

        echo "  请手动上传代码到 ${WORK_DIR}/ALPHAQUBIT"echo ""

        exit 1

    fi# =============================================================================

fi# Step 7: 运行论文对齐验证

# =============================================================================

cd ${WORK_DIR}/ALPHAQUBIT

echo "  ✓ 代码已就绪: ${WORK_DIR}/ALPHAQUBIT"echo "[Step 7] 运行论文对齐验证..."

echo ""

cd ${WORK_DIR}/ALPHAQUBIT

# =============================================================================python -m verification.verify_paper_alignment || echo "  ! 验证脚本运行失败，请检查"

# Step 5: 安装Python依赖

# =============================================================================echo ""



echo "[Step 5] 安装Python依赖..."# =============================================================================

# Step 8: 创建运行脚本

cd ${WORK_DIR}/ALPHAQUBIT# =============================================================================



# 确保pip是最新的echo "[Step 8] 创建运行脚本..."

echo "  升级pip..."

pip install --upgrade pip -q# 创建快速测试脚本

cat > ${WORK_DIR}/run_quick_test.sh << 'SCRIPT'

# 安装PyTorch (优先尝试NPU版本)#!/bin/bash

echo "  安装PyTorch..."# 快速测试脚本

source $(conda info --base)/etc/profile.d/conda.sh

# 首先检查是否已安装conda activate alphaqubit

TORCH_INSTALLED=$(python -c "import torch; print('yes')" 2>/dev/null || echo "no")cd ~/alphaqubit/ALPHAQUBIT

python run_server_pipeline.py --quick-test --output-dir ./results_quick_test

if [ "$TORCH_INSTALLED" = "no" ]; thenSCRIPT

    # 尝试安装华为昇腾NPU版本chmod +x ${WORK_DIR}/run_quick_test.sh

    echo "  尝试安装PyTorch NPU版本..."

    pip install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>/dev/null || \# 创建完整运行脚本

    pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple -qcat > ${WORK_DIR}/run_full_training.sh << 'SCRIPT'

    #!/bin/bash

    # 安装torch_npu (如果有NPU)# 完整训练脚本 (论文复现)

    if command -v npu-smi &> /dev/null; thensource $(conda info --base)/etc/profile.d/conda.sh

        pip install torch_npu -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>/dev/null || \conda activate alphaqubit

        echo "  ! torch_npu安装失败，将使用CPU模式"cd ~/alphaqubit/ALPHAQUBIT

    fi

else# 使用nohup在后台运行，防止SSH断开

    echo "  ✓ PyTorch已安装"nohup python run_server_pipeline.py \

fi    --output-dir ./results_full_$(date +%Y%m%d_%H%M%S) \

    --device npu \

# 安装项目依赖    > training.log 2>&1 &

echo "  安装项目依赖..."

if [ -f "requirements.txt" ]; thenecho "训练已在后台启动，PID: $!"

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>/dev/null || \echo "查看日志: tail -f training.log"

    pip install -r requirements.txt -qSCRIPT

fichmod +x ${WORK_DIR}/run_full_training.sh



# 安装stim# 创建环境激活脚本

echo "  安装stim..."cat > ${WORK_DIR}/activate_env.sh << 'SCRIPT'

pip install stim -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>/dev/null || \#!/bin/bash

pip install stim -q# 激活AlphaQubit环境

source $(conda info --base)/etc/profile.d/conda.sh

# 安装项目本身conda activate alphaqubit

echo "  安装alphaqubit包..."cd ~/alphaqubit/ALPHAQUBIT

pip install -e . -q 2>/dev/null || trueecho "AlphaQubit环境已激活"

echo "工作目录: $(pwd)"

echo "  ✓ 依赖安装完成"SCRIPT

echo ""chmod +x ${WORK_DIR}/activate_env.sh



# =============================================================================echo "  ✓ 已创建以下脚本:"

# Step 6: 验证安装echo "    - ${WORK_DIR}/run_quick_test.sh      # 快速测试"

# =============================================================================echo "    - ${WORK_DIR}/run_full_training.sh   # 完整训练"

echo "    - ${WORK_DIR}/activate_env.sh        # 激活环境"

echo "[Step 6] 验证安装..."echo ""



python << 'EOF'# =============================================================================

import sys# Step 9: 设置SSH保持连接

print(f"  Python版本: {sys.version.split()[0]}")# =============================================================================



# 检查PyTorchecho "[Step 9] 配置SSH..."

try:

    import torch# 创建SSH配置目录

    print(f"  PyTorch版本: {torch.__version__}")mkdir -p ~/.ssh

    

    # 检查NPU# 添加保持连接配置

    try:if ! grep -q "ServerAliveInterval" ~/.ssh/config 2>/dev/null; then

        import torch_npu    cat >> ~/.ssh/config << 'SSHCONFIG'

        if torch.npu.is_available():Host *

            print(f"  ✓ NPU可用: {torch.npu.device_count()} 个设备")    ServerAliveInterval 60

        else:    ServerAliveCountMax 3

            print("  ! NPU不可用，将使用CPU")SSHCONFIG

    except ImportError:    echo "  ✓ SSH保持连接配置已添加"

        # 检查CUDAelse

        if torch.cuda.is_available():    echo "  ✓ SSH配置已存在"

            print(f"  ✓ CUDA可用: {torch.cuda.device_count()} 个GPU")fi

        else:

            print("  ! 将使用CPU训练")echo ""

except ImportError as e:

    print(f"  ! PyTorch导入失败: {e}")# =============================================================================

    sys.exit(1)# 完成

# =============================================================================

# 检查stim

try:echo "=============================================="

    import stimecho "部署完成!"

    print(f"  ✓ Stim版本: {stim.__version__}")echo "=============================================="

except ImportError as e:echo ""

    print(f"  ! Stim导入失败: {e}")echo "下一步操作:"

echo ""

# 检查numpyecho "1. 激活环境:"

try:echo "   source ~/alphaqubit/activate_env.sh"

    import numpy as npecho ""

    print(f"  ✓ NumPy版本: {np.__version__}")echo "2. 运行快速测试:"

except ImportError as e:echo "   ~/alphaqubit/run_quick_test.sh"

    print(f"  ! NumPy导入失败: {e}")echo ""

echo "3. 运行完整训练 (后台):"

print("")echo "   ~/alphaqubit/run_full_training.sh"

print("  ✓ 验证完成!")echo ""

EOFecho "4. 查看训练日志:"

echo "   tail -f ~/alphaqubit/ALPHAQUBIT/training.log"

echo ""echo ""

echo "5. 下载结果到本地:"

# =============================================================================echo "   scp -r user@server:~/alphaqubit/ALPHAQUBIT/results_* ."

# Step 7: 创建运行脚本echo ""

# =============================================================================echo "=============================================="


echo "[Step 7] 创建运行脚本..."

# 创建环境激活脚本
cat > ${WORK_DIR}/activate_env.sh << 'SCRIPT'
#!/bin/bash
# 激活AlphaQubit环境
export PATH="${HOME}/miniconda3/bin:$PATH"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate alphaqubit
cd ~/alphaqubit/ALPHAQUBIT
echo "✓ AlphaQubit环境已激活"
echo "  工作目录: $(pwd)"
echo "  Python: $(which python)"
SCRIPT
chmod +x ${WORK_DIR}/activate_env.sh

# 创建快速测试脚本
cat > ${WORK_DIR}/run_quick_test.sh << 'SCRIPT'
#!/bin/bash
# 快速测试脚本
export PATH="${HOME}/miniconda3/bin:$PATH"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate alphaqubit
cd ~/alphaqubit/ALPHAQUBIT

echo "运行快速测试..."
python run_server_pipeline.py --quick-test --output-dir ./results_quick_test
SCRIPT
chmod +x ${WORK_DIR}/run_quick_test.sh

# 创建完整训练脚本
cat > ${WORK_DIR}/run_full_training.sh << 'SCRIPT'
#!/bin/bash
# 完整训练脚本 (论文复现)
export PATH="${HOME}/miniconda3/bin:$PATH"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate alphaqubit
cd ~/alphaqubit/ALPHAQUBIT

# 检测设备
DEVICE="cpu"
python -c "import torch_npu; import torch; print(torch.npu.is_available())" 2>/dev/null | grep -q "True" && DEVICE="npu"
python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True" && DEVICE="cuda"

echo "使用设备: $DEVICE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results_full_${TIMESTAMP}"

# 使用nohup在后台运行
nohup python run_server_pipeline.py \
    --output-dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    > training_${TIMESTAMP}.log 2>&1 &

PID=$!
echo "✓ 训练已在后台启动"
echo "  PID: $PID"
echo "  日志: training_${TIMESTAMP}.log"
echo "  输出: ${OUTPUT_DIR}"
echo ""
echo "查看日志: tail -f training_${TIMESTAMP}.log"
echo "停止训练: kill $PID"
SCRIPT
chmod +x ${WORK_DIR}/run_full_training.sh

# 创建打包结果脚本
cat > ${WORK_DIR}/pack_results.sh << 'SCRIPT'
#!/bin/bash
# 打包结果用于下载
cd ~/alphaqubit/ALPHAQUBIT
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACK_NAME="alphaqubit_results_${TIMESTAMP}.tar.gz"

echo "打包结果文件..."
tar -czvf ~/${PACK_NAME} \
    results_* \
    output/ \
    *.log \
    2>/dev/null || echo "部分文件不存在"

echo ""
echo "✓ 结果已打包: ~/${PACK_NAME}"
echo ""
echo "下载命令 (在本地执行):"
echo "  scp user@server:~/${PACK_NAME} ."
SCRIPT
chmod +x ${WORK_DIR}/pack_results.sh

echo "  ✓ 已创建以下脚本:"
echo "    - ~/alphaqubit/activate_env.sh       # 激活环境"
echo "    - ~/alphaqubit/run_quick_test.sh     # 快速测试"
echo "    - ~/alphaqubit/run_full_training.sh  # 完整训练(后台)"
echo "    - ~/alphaqubit/pack_results.sh       # 打包结果"
echo ""

# =============================================================================
# 完成
# =============================================================================

echo "=============================================="
echo "✓ 部署完成!"
echo "=============================================="
echo ""
echo "下一步操作:"
echo ""
echo "1. 激活环境:"
echo "   source ~/alphaqubit/activate_env.sh"
echo ""
echo "2. 运行快速测试:"
echo "   bash ~/alphaqubit/run_quick_test.sh"
echo ""
echo "3. 运行完整训练 (后台):"
echo "   bash ~/alphaqubit/run_full_training.sh"
echo ""
echo "4. 查看训练日志:"
echo "   tail -f ~/alphaqubit/ALPHAQUBIT/training_*.log"
echo ""
echo "5. 打包并下载结果:"
echo "   bash ~/alphaqubit/pack_results.sh"
echo ""
