# 卷积神经网络轻量化：原理、实现与实验

作者：wyx

## 目录
- [环境配置](#环境配置)
- [训练模型](#训练模型)
- [测试模型](#测试模型)

## 环境配置

在开始训练和测试模型之前，请确保你已经在一个虚拟环境中安装了所有必要的依赖项。你可以使用 `requirements.txt` 文件来安装这些依赖项。

1. 创建并激活虚拟环境
   
    ```
    conda create --name myenv
    conda activate myenv
    ```
    
2. 安装依赖项
    ```bash
    pip install -r requirements.txt
    ```

## 训练模型

你可以使用以下指令来训练不同的模型。请根据你的需求选择合适的模型和训练周期。

- 训练 MobileNetV1 模型
    ```bash
    python train.py --model mobilenetv1 --num_epochs 100
    ```

- 训练 MobileNetV2 模型
    ```bash
    python train.py --model mobilenetv2 --num_epochs 100
    ```

- 训练 MobileNetV3 Small 模型
    ```bash
    python train.py --model mobilenetv3_small --num_epochs 100
    ```

- 训练 MobileNetV3 Large 模型
    ```bash
    python train.py --model mobilenetv3_large --num_epochs 100
    ```

- 训练DenseNet模型

  ```bash
  python train.py --model densenet --num_epochs 100
  ```

  

如果你希望一次性训练五个模型，也可以使用：

```
./train_models.sh
```



## 测试模型

在训练完成后，你可以使用以下指令来测试不同的模型。

- 测试 MobileNetV1 模型
    ```bash
    python test.py --model mobilenetv1
    ```

- 测试 MobileNetV2 模型
    ```bash
    python test.py --model mobilenetv2
    ```

- 测试 MobileNetV3 Small 模型
    ```bash
    python test.py --model mobilenetv3_small
    ```

- 测试 MobileNetV3 Large 模型
    ```bash
    python test.py --model mobilenetv3_large
    ```

- 测试 DenseNet 模型
    ```bash
    python test.py --model densenet
    ```

## 注意事项

- 请确保在训练和测试前，数据已经准备好并正确配置。
- 请确保你创建了result/data文件夹和result/model文件夹，否则在保存模型时会报错：Cannot save file into a non-existent directory: 'result/data'。
- 可以根据需要调整 `num_epochs` 参数以适应你的训练需求。
- 如果你遇到任何问题，请检查 `requirements.txt` 文件中的依赖项是否安装正确，并确保你的虚拟环境已激活。



感谢你使用本项目！
