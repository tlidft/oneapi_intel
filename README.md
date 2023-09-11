# Intel® oneAPI简介

Intel oneAPI 是一个跨行业、开放、基于标准的统一的编程模型，它为跨 CPU、GPU、FPGA、专用加速器的开发者提供统一的体验。

它由一项行业计划和一款英特尔beta产品组成。

oneAPI 开放规范基于行业标准和现有开发者编程模型，广泛适用于不同架构和来自不同供应商的硬件。

oneAPI 行业计划鼓励生态系统内基于oneAPI规范的合作以及兼容 oneAPI的实践。

通过oneAPI，我们可以最大程度地忽略硬件之间的差异，而使用统一的编程模型编写程序，有效地提高我们的开发效率

# 基于Intel® oneAPI 的 CS-LSTM 算法实现

## 简介

我们可以基于oneAPI编写、训练、使用神经网络，下方的代码展示了如何使用Intel® oneAPI的pytorch extension编写并训练一个 CS-LSTM 网络

本代码是CS-LSTM算法的一个简单实现，实现了使用CS-LSTM算法训练实现轨迹预测的代码。

## Intel®Extension for PyTorch

该拓展包拓展了pytorch所支持的硬件设备，不再拘泥于CUDA或CPU运行。其利用了英特尔CPU上的AVX-512矢量神经网络指令（AVX512 VNNI）和英特尔Xe高级矩阵扩展，以及英特尔GPU上的Xe矩阵扩展（XMX）AI引擎对AI训练进行加速。

## 依赖

本项目需要intel pytorch extension、pytorch等库才能运行。

安装：
```shell
pip install torch intel_extension_for_pytorch
```

## 使用XPU加速运算

```python
# 引入oneAPI中的Intel Extension for Pytorch
import intel_extension_for_pytorch as ipex

# 选择xpu作为torch运算硬件
device = torch.device("xpu")
```

## 运行

运行python文件即可。
```shell
python train.py
```

运行时，会显示每轮训练所需时间，可将其与纯CPU环境进行对比