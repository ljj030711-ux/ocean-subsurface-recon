# B4 C+ 当前实现说明（架构 v2，patch size 8）

本文对应当前 `models/b4_cplus/` 代码：前一深度温度场的输入编码采用固定的
`output_patch_size=8`，空间/垂向 Transformer 均在 **37×25** 网格上运行。
先前 `B4_CPlus_all_size` 的结果和权重来自 **架构 v1、patch size 16、19×13
瓶颈、四尺度输出**，不能当作本文 v2 的实验结果或直接载入 v2 模型。
截至本文更新，v2 尚无独立训练和测试结果。

## 分支与目录

- 分支：`B4`
- 工作目录：`/Users/lijunjie/Documents/python/ocean-subsurface-recon-b4`
- B3目录保持在 `/Users/lijunjie/Documents/python/ocean-subsurface-recon-b3`，分支为`B3`，没有保留上一次U-Net改动。
- B4的Embedding、causal depth self-attention和memory cross-attention均位于`models/b4_cplus/`，运行B4不导入或依赖`models/b3_depthformer/`。

## 数据与目标

默认使用`glorys12v1_all_size`：SST为`(B,1,480,320)`，SSH/SSS合并为`(B,2,192,128)`，输出为`(B,D,289,193)`。不指定`--depth-indices`时`D=28`；与旧实验对照所用的`0,4,9,17,24,27`对应`D=6`。空间范围为0–24°N、105–121°E，输出深度使用GLORYS12V1的原生深度坐标。

## 当前结构与每层尺度

下表按目标场`289×193`、默认通道数列出实际代码的尺度。`B`为batch size，
`D`为输出深度数；解码器内部将`B×D`展平以对每层温度场共享同一组卷积权重。

| 阶段 | 输入及操作 | 输出特征 / 预测尺度 | skip 与输出头 |
| --- | --- | --- | --- |
| SST原生分支 | `480×320`经4×4、stride-2卷积及残差卷积 | `(B,64,240,160)` | 无 |
| SSH/SSS原生分支 | `192×128`经两个残差卷积块 | `(B,64,192,128)` | 无 |
| 坐标重网格与融合 | 两路特征分别重网格到`ceil(289/2)×ceil(193/2)`，拼接后卷积融合 | `(B,64,145,97)` | 保存为skip-1；此处无预测头 |
| 共享编码层1 | 3×3、stride-2 DownBlock | `(B,128,73,49)` | 保存为skip-2；此处无预测头 |
| 共享编码层2 / 瓶颈 | 3×3、stride-2 DownBlock | `(B,128,37,25)` | 无skip、无预测头；展平为925个token |
| Surface Transformer | 4层全局空间注意力 | memory `(B,925,128)` | 供深度解码器读取 |
| Depth Transformer | 3层因果深度注意力及surface-memory cross-attention | `(B,D,925,128)`，还原为`(B,D,128,37,25)` | 从浅到深生成 |
| U-Net解码层1 | 从`37×25`双线性上采样，拼接skip-2，再做残差卷积 | `(B×D,128,73,49)` | 73×49预测头 |
| U-Net解码层2 | 从`73×49`双线性上采样，拼接skip-1，再做残差卷积 | `(B×D,64,145,97)` | 145×97预测头 |
| U-Net最终层 | 从`145×97`双线性上采样，通道投影及残差卷积 | `(B×D,32,289,193)` | **无289×193 skip**；最终预测头 |

三个预测头都只预测**温度**，分别在73×49、145×97、289×193上生成
`(B,D,H,W)`。低分辨率预测先双线性上采样到289×193，然后用一组
**对所有样本和深度共用**的可学习softmax权重加权相加。权重初始各为1/3，
训练后可变化；这是三尺度**输出融合**，与三维温度的`D`个深度不是同一概念。
目前没有仅使用最终预测头的消融实现或该消融的结果。

这里的`output_patch_size=8`用于把上一深度的289×193温度场编码为
`ceil(289/8)×ceil(193/8)=37×25`的token网格，并约束CNN瓶颈与之对齐。
B4最终温度场通过U-Net卷积解码得到，**不使用B3的8×8 patch-fold重建头**。

```text
SST (B,1,480,320)
  -> 4x4 stride-2 CNN + residual CNN
  -> (B,64,240,160)
  -> 经纬度重网格一次 -> (B,64,145,97) --+
                                                   |
SSH/SSS (B,2,192,128)                              |
  -> two residual CNN blocks                       |
  -> (B,64,192,128)                                |
  -> 经纬度重网格一次 -> (B,64,145,97) -----------+
                                                   |
                         concat + 1x1/3x3 fusion --+
                                                   v
skip-1 (B,64,145,97)
  -> DownBlock -> skip-2 (B,128,73,49)
  -> DownBlock -> bottleneck (B,128,37,25)
  -> flatten -> 925 surface tokens
  -> 4-layer Surface Transformer Encoder
  -> memory (B,925,128)

BOS / previous temperature field + continuous depth embedding
  -> 3-layer causal Depth Transformer Decoder
     * depth self-attention：浅层到深层的因果注意力
     * memory cross-attention：每一层均读取925个surface memory token
  -> decoded depth tokens (B,D,925,128)

decoded tokens -> 37x25
  -> upsample + skip-2 -> 73x49 -> head-1
  -> upsample + skip-1 -> 145x97 -> head-2
  -> skip-free upsample -> 289x193 -> head-3
  -> head-1/head-2上采样至289x193，与head-3做softmax加权融合
  -> T-Field (B,D,289,193)
```

这里的Transformer不是二选一：4层`surface_encoder`负责生成memory；3层`depth_decoder`同时包含垂向self-attention和对surface memory的cross-attention。37×25与B3的patch-size-8 Transformer网格相同，但输入memory由B4的原生网格CNN和坐标重网格生成，重建方式也不同。

## 经纬度重网格

重网格不按数组长宽比例直接插值，而是先建立目标经纬度到源经纬度的归一化采样坐标，再用双线性`grid_sample`采样。卫星网格按cell centre解释，GLORYS目标网格按包含区域端点的规则点解释。该算子无可训练参数，但对输入特征可微，因此前面的原生网格CNN可端到端学习。

若原始`.npy`纬度索引为从北到南，训练时加`--b4-source-lat-descending`；默认假设从南到北。由于`.npy`本身不携带坐标，必须按生成这些文件时的纬度顺序设置该选项。

## 参数量

以下为当前v2默认28层配置：`d_model=128`、8头、4层Surface Encoder、3层Depth Decoder、FFN宽度512、原生基础通道32、融合通道64、`output_patch_size=8`。

| 模块 | 参数量 |
| --- | ---: |
| C+原生分支与共享Encoder | 1,039,168 |
| Surface Transformer Encoder | 793,344 |
| 前一深度场Patch Embedding | 8,320 |
| 连续深度Embedding | 16,768 |
| Depth Transformer Decoder（含cross-attention） | 794,752 |
| C+三尺度U-Net Decoder | 640,102 |
| 位置编码与BOS | 32,896 |
| **总参数量/可训练参数量** | **3,325,350** |

对比当前干净B3默认28层模型：

| 模型 | 参数量 | 相对B3变化 |
| --- | ---: | ---: |
| B3 DepthFormer | 1,889,441 | — |
| B4 C+ | 3,325,350 | **+1,435,909（+76.00%）** |

坐标重网格本身增加0个参数。增加量主要来自C+卷积Encoder和多尺度Decoder。

## 训练

不传`--depth-indices`时训练全部28层；B4默认batch size为16，下面显式写出该值以便复现实验：

```bash
cd /Users/lijunjie/Documents/python/ocean-subsurface-recon-b4
python -u train.py \
  --method b4_cplus --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir ./data/raw_all_size \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --epochs 50 --patience 10 --batch-size 16 --lr 1e-4 \
  --checkpoint-dir ./checkpoints --output-dir ./outputs --seed 42
```

若只做与旧实验相同的6层对照，可加：

```text
--depth-indices 0,4,9,17,24,27
```

## 测试

```bash
python -u test.py \
  --method b4_cplus --target-var temperature \
  --dataset-profile glorys12v1_all_size \
  --data-dir ./data/raw_all_size \
  --checkpoint ./checkpoints/2dto2d/B4_CPlus/temperature/B4_CPlus_temperature_28depths_best.pth \
  --start-date 2002-01-01 --end-date 2023-12-31 \
  --eval-start 2023-01-01 --eval-end 2023-12-31 \
  --eval-batch-size 1 --output-dir ./outputs
```

checkpoint文件名中的深度标签由`--depth-indices`自动产生；以上`28depths`名称对应不指定深度索引的情况。
