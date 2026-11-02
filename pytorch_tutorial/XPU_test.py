import torch

print("PyTorch Version:", torch.__version__)
print("XPU is available:", torch.xpu.is_available())

if torch.xpu.is_available():
    print("Device Count:", torch.xpu.device_count())
    print("Device Name:", torch.xpu.get_device_name(0))

    # 测试张量计算
    x = torch.randn(3, 3, device="xpu")
    print("Tensor on Arc GPU:\n", x)
else:
    print("未能识别到 Intel GPU，请检查显卡驱动及版本！")