# 11785 HW2P2 Submission
huiyenc, Chia Hui Yen, 10 Mar 25

## Link:
wandb link: https://wandb.ai/hychia2024-carnegie-mellon-university/hw2p2-ablations?nw=nwuserhychia2024 

## Log:
1. **Attempt 1:**  
A basic try with 5 layer of CNN, 1 epoch, without data augmentation.  
Public score: 0.16037  

```
config={'batch_size': 64, 'lr': 0.1, 'epochs': 1, 'num_classes': 8631, 'cls_data_dir': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/cls_data', 'ver_data_dir': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/ver_data', 'val_pairs_file': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/val_pairs.txt', 'test_pairs_file': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/test_pairs.txt', 'checkpoint_dir': '/kaggle/working/', 'augument': False}
```
<br>

2. **Attempt 2:**  
Second attempt, 5 later of CNN, 10 epochs, with data augmentation, add scheduler CosineAnnealingLR, OneCycleLR.  
Public score: 0.09140

```
Result={'ACC': 91.4, 'EER': 8.840864439974025, 'AUC': 96.44604851972038, 'TPRs': [('TPR@FPR=1e-4', 57.23014256619145), ('TPR@FPR=5e-4', 57.23014256619145), ('TPR@FPR=1e-3', 57.23014256619145), ('TPR@FPR=5e-3', 62.11812627291242), ('TPR@FPR=5e-2', 87.16904276985743)]}
Val Ret. Acc 91.4000%
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 64, 28, 28]           9,472
       BatchNorm2d-2           [-1, 64, 28, 28]             128
              ReLU-3           [-1, 64, 28, 28]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
       BatchNorm2d-5          [-1, 128, 14, 14]             256
              ReLU-6          [-1, 128, 14, 14]               0
            Conv2d-7            [-1, 256, 7, 7]         295,168
       BatchNorm2d-8            [-1, 256, 7, 7]             512
              ReLU-9            [-1, 256, 7, 7]               0
           Conv2d-10            [-1, 512, 4, 4]       1,180,160
      BatchNorm2d-11            [-1, 512, 4, 4]           1,024
             ReLU-12            [-1, 512, 4, 4]               0
           Conv2d-13           [-1, 1024, 2, 2]       4,719,616
      BatchNorm2d-14           [-1, 1024, 2, 2]           2,048
             ReLU-15           [-1, 1024, 2, 2]               0
    AdaptiveAvgPool2d-16       [-1, 1024, 1, 1]               0
          Flatten-17                 [-1, 1024]               0
           Linear-18                 [-1, 8631]       8,846,775

```
config={'batch_size': 64, 'lr': 0.1, 'epochs': 10, 'num_classes': 8631, 'cls_data_dir': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/cls_data', 'ver_data_dir': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/ver_data', 'val_pairs_file': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/val_pairs.txt', 'test_pairs_file': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/test_pairs.txt', 'checkpoint_dir': '/kaggle/working/', 'augument': True}
```
<br>

3. **Attempt 3:**  
Third attempt, 5 layers of CNN, add ResNet block with residual connections, 30 epochs, with data augmentation and add scheduler CosineAnnealingLR, OneCycleLR.  
Public score: 0.04009  
```
Result={'ACC': 94.9, 'EER': 5.702647657846236, 'AUC': 98.2626370944186, 'TPRs': [('TPR@FPR=1e-4', 67.82077393075356), ('TPR@FPR=5e-4', 67.82077393075356), ('TPR@FPR=1e-3', 67.82077393075356), ('TPR@FPR=5e-3', 82.89205702647658), ('TPR@FPR=5e-2', 93.68635437881873)]}
Val Ret. Acc 94.9000%
```

```
torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
```

```
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main branch: two conv layers
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        # Shortcut branch: if shape changes, adjust with 1x1 conv
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
```

```
config={'batch_size': 128, 'lr': 0.1, 'epochs': 30, 'num_classes': 8631, 'cls_data_dir': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/cls_data', 'ver_data_dir': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/ver_data', 'val_pairs_file': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/val_pairs.txt', 'test_pairs_file': '/kaggle/input/11785-hw-2-p-2-face-verification-spring-2025/HW2p2_S25/test_pairs.txt', 'checkpoint_dir': '/kaggle/working/', 'augument': True}
```
