# B4 U-Net 模型结构说明

当前代码为架构 v5：从原生分辨率提取两路海表特征，在 `289×193` 融合后
构建完整金字塔，并仅使用最终 `289×193` 温度输出 head。下方架构图仍是此前 v2 示意；
新增的 `289×193` skip 与金字塔起点以本文第 3、6–8 节和代码为准。

本文说明当前 B4 U-Net 的实际实现。代码目录和训练方法名仍保留
`models/b4_cplus/`、`b4_cplus` 等历史标识，但本文统一将模型称为
**B4 U-Net**。

模型的目标是：根据同一天的海表温度（SST）、海表高度（SSH）和海表盐度
（SSS），从浅到深自回归重建三维温度场。默认目标水平网格为
`289×193`，Transformer 隐藏维度为 128，空间 token 网格为 `37×25`。

![B4 U-Net 架构图](../analysis/b4_unet_architecture/b4_unet_from_b3_edited_v5.png)

## 1. 核心思路

B4 U-Net 由三个连续部分组成：

1. **海表编码器**：分别在 SST 与 SSH/SSS 的原生网格上提取特征，再按真实
   经纬度重网格到共同的 `289×193` 网格；随后逐级下采样到 `145×97`、
   `73×49` 和 `37×25`，同时保存三条 U-Net skip。
2. **空间与垂向 Transformer**：将 `37×25` 特征展平为 925 个空间 token，
   生成海表 memory；深度解码器以 BOS 或上一深度温度场为输入，利用因果
   深度注意力和 memory cross-attention 从浅到深生成每层隐状态。
3. **共享 U-Net 解码器**：把每层的 925 个 token 还原为 `37×25` 特征图，
   依次融合 `73×49`、`145×97`、`289×193` 三条海表 skip；只在最终
   `289×193` 尺度预测温度，不再融合中间尺度的预测图。

海表 memory 和三条 skip 对同一天的所有目标深度共用；深度之间的差异主要
由上一深度温度场、深度位置编码和因果深度注意力共同表达。

## 2. 输入、输出和默认符号

| 符号 | 含义 | 默认形状 |
| --- | --- | --- |
| `B` | batch size | — |
| `D` | 需要重建的深度层数 | 全深度时为 28；也可选择部分深度 |
| `d` | token 隐藏维度 | 128 |
| `N` | 空间 token 数 | `37×25=925` |
| SST | 海表温度 | `(B,1,480,320)` |
| SSH/SSS | 海表高度与海表盐度 | `(B,2,192,128)` |
| `M` | 海表多源 memory | `(B,925,128)` |
| 输出温度场 | 从浅到深排列的温度 | `(B,D,289,193)` |

默认空间范围为 0–24°N、105–121°E。目标深度采用数据集中的真实、不等距
深度值，而不是把深度层当成等间隔序号。

## 3. 海表编码器：从原生网格到 37×25 瓶颈

### 3.1 为什么先分支编码

SST 的原始尺寸是 `480×320`，SSH/SSS 的原始尺寸是 `192×128`。两类数据
不仅数组大小不同，像元中心所代表的经纬度位置也不同，因此不能直接按数组
下标拼接。模型先让每一类变量在自己的原生网格上完成浅层特征提取，再统一
到共同经纬度网格，避免在特征提取前过早抹平不同数据源的原生结构。

### 3.2 SST 原生分支

SST 的处理过程为：

```text
(B,1,480,320)
  -> ResidualConvBlock: 1 -> 32 channels
  -> ResidualConvBlock: 32 -> 64 channels
  -> (B,64,480,320)
```

两层残差卷积均不改变原生空间尺寸；保留旧版浅层的 32、64 通道宽度，
但 SST 在坐标重网格前不再提前下采样。

### 3.3 SSH/SSS 原生分支

SSH 和 SSS 在通道维合并后输入同一分支：

```text
(B,2,192,128)
  -> ResidualConvBlock: 2 -> 32 channels
  -> ResidualConvBlock: 32 -> 64 channels
  -> (B,64,192,128)
```

该分支保持 `192×128` 的原生空间尺度，只提取并变换通道特征。

### 3.4 坐标感知的双线性重网格

两路特征分别被重网格到：

```text
289×193
```

这里不是按照数组长宽比例直接执行普通 resize，而是按物理经纬度建立采样
坐标：

1. 在目标区域端点之间生成 `289×193` 个目标经纬度点；
2. 根据源特征的尺寸和 cell-centred 假设计算源网格中心范围；
3. 将目标经纬度换算为源特征图中的归一化坐标 `[-1,1]`；
4. 使用双线性 `grid_sample` 在源特征图上取值。

双线性采样使用目标点周围四个源网格点的加权值。采用它有四个主要原因：

- **保证物理位置对齐**：相同数组下标不一定对应相同经纬度；先建立经纬度
  映射才能正确融合 SST、SSH 和 SSS。
- **比最近邻连续**：不会产生明显块状边界，更适合连续海洋变量及其特征。
- **稳定且无额外参数**：重网格过程不需要再学习一套几何映射，减少把空间
  错位误当成可学习特征的风险。
- **保持端到端可训练**：采样坐标本身固定，但双线性采样对输入特征可微，
  梯度仍可传回两个原生 CNN 分支。

边界采用 `padding_mode="border"`，超出源 cell-centre 有效范围的少量目标
边界点使用最近的边界值，避免额外的零值边缘。若源数据纬度索引为从北到南，
归一化纬度坐标会反向；纬度方向必须与数据文件的实际排列一致。

相比旧版在 `145×97` 融合，新版把融合和首条 skip 提前到 `289×193`，
保留更细的海表特征，但增加了计算与激活显存成本。

### 3.5 多源融合和三条 skip

重网格后的两路特征形状均为 `(B,64,289,193)`。它们先在通道维拼接为
128 通道，再经过 `1×1` 卷积压缩到 64 通道，并用残差卷积完成局部融合：

```text
SST feature     (B,64,289,193) --+
                                  +-> concat -> 1×1 conv -> residual conv
SSH/SSS feature (B,64,289,193) --+
```

融合结果既是共享编码金字塔的起点，也是第一条 skip。随后经过三个
`DownBlock`：

| 编码阶段 | 操作 | 输出 | 是否保存为 skip |
| --- | --- | --- | --- |
| 多源融合 | 拼接、`1×1` 卷积、残差卷积 | `(B,64,289,193)` | 是 |
| 下采样 1 | `3×3`、stride-2 卷积 + 残差卷积 | `(B,64,145,97)` | 是 |
| 下采样 2 | `3×3`、stride-2 卷积 + 残差卷积 | `(B,128,73,49)` | 是 |
| 下采样 3 | `3×3`、stride-2 卷积 + 残差卷积 | `(B,128,37,25)` | 否，作为瓶颈 |

由于卷积使用 `kernel_size=3`、`stride=2`、`padding=1`，奇数尺寸按向上取整
缩小：`289×193 -> 145×97 -> 73×49 -> 37×25`。通道数在前两级保持
64，到 `73×49` 增至 128 后保持不变，
使瓶颈通道与 Transformer 的 `d_model=128` 完全一致。

## 4. 海表空间编码：生成共享 memory

瓶颈 `(B,128,37,25)` 被展平并转置为：

```text
(B,128,37,25) -> (B,925,128)
```

卷积特征本身不显式包含 token 的绝对位置，因此模型直接定义了一张可学习
参数图：

```text
E_xy_base: (1,128,16,16)
  -> bicubic interpolation
E_xy_grid: (1,128,37,25)
  -> flatten + transpose
E_xy: (1,925,128)
```

这里不是先把某个输入特征“编码成 `16×16`”。`16×16` 是位置编码参数在模型
中保存和训练时的**基础网格尺寸**：初始化时就创建
`128×16×16=32,768` 个可学习参数。每次前向传播时，再通过双三次插值
（bicubic）把这张参数图调整到实际 Transformer 网格 `37×25`。

插值得到的同一份 `E_xy` 用在两个位置：

1. 加到海表 CNN 瓶颈 token 上，再送入 Spatial Transformer 生成 memory；
2. 加到每一个深度的 BOS/上一深度内容 token 上，构成 Depth Transformer
   的 query。

`E_xy` 在所有深度之间共享权重。模型中只有一个
`self.spatial_position.embedding` 参数张量；一次前向传播先把它插值成
`(1,925,128)`，随后在深度维广播。因此，同一个水平位置 `(x,y)` 在不同
深度使用同一空间位置向量，并不会为每个深度单独学习一张 `E_xy`。来自
surface memory 和所有深度 query 的梯度都会共同更新这一个参数张量。

这里共享的是“水平位置含义”；不同深度仍由各自的连续深度编码 `E_z(z_k)`、
上一深度内容 token 和因果注意力隐状态进行区分。

因此，双三次插值发生在**二维空间位置编码进入 Transformer 之前**，不是
U-Net 上采样，也不是海温场或海表特征的重采样。

这里的双三次插值与上一节的坐标重网格目的不同：

- 坐标重网格处理的是不同数据源的**物理经纬度对齐**，使用双线性
  `grid_sample`；
- 位置编码插值只是把一张较小的**可学习参数图**平滑调整到当前 token
  网格，使用 bicubic，不改变输入海洋变量。

`16×16 -> 37×25` 是当前实现中的一个可独立消融点。后续可以考虑直接学习
`37×25` 位置参数，从而完全去掉这一步插值；代价是位置参数从 32,768 增至
`128×37×25=118,400`，并且现有 checkpoint 的位置参数形状不再兼容。也可以
改为经纬度函数或坐标 MLP 生成位置编码。本文只记录这些可选方向，当前代码
仍采用 `16×16` 可学习参数加 bicubic 插值。

加入空间位置编码后，925 个 token 经过 4 层 Spatial Transformer Encoder。
自注意力使任意空间 patch 都能与其他 patch 交换信息，输出：

```text
surface memory M: (B,925,128)
```

`M` 对每个样本只编码一次。后续预测不同深度时读取的是同一份海表 memory，
不会为每个深度重复下采样或重新生成 memory。

## 5. 深度自回归编码与解码逻辑

### 5.1 三种 query 信息

第 `k` 个深度的 query 由三类信息逐元素相加，而不是在通道维拼接：

```text
Q_0 = E_BOS + E_z(z_0) + E_xy

Q_k = PatchEmbed(X_{k-1}) + E_z(z_k) + E_xy,  k >= 1
```

三项形状都为 `(B,925,128)`：

- `E_BOS`：可学习的 BOS token，表示第一层没有上一深度温度场；
- `PatchEmbed(X_{k-1})`：上一深度的真实值或预测值经过 `8×8` patch
  embedding 后的内容 token；
- `E_z(z_k)`：当前真实深度值的连续深度编码；
- `E_xy`：与海表 memory 相同的 `37×25` 空间位置编码。

### 5.2 上一深度温度场的 patch embedding

上一深度温度场大小为 `289×193`。模型在右侧和底部补零到 `296×200`，再
使用 `kernel_size=8`、`stride=8` 的卷积：

```text
(B,1,289,193)
  -> pad to 296×200
  -> 8×8 patch embedding
  -> (B,128,37,25)
  -> flatten
  -> (B,925,128)
```

因此 `output_patch_size=8` 直接控制的是“上一深度温度场如何变成 query
token”。它也间接要求海表 CNN 瓶颈必须是同一个 `37×25` 网格，代码会显式
检查两者是否一致。它并不是对 U-Net 的每一级 skip 分别做一次 patch
embedding。

### 5.3 `output_patch_size` 与 U-Net 尺度的区别

当前模型**没有“U-Net patch size”这个参数**。容易混淆，是因为两条路径最后
恰好都对齐到 `37×25`：

| 概念 | 当前设置 | 作用 |
| --- | --- | --- |
| 上一深度场 Patch Embedding | `kernel=8, stride=8` | 把 `289×193` 温度场切成 `37×25` 个 query token |
| U-Net 编码下采样 | `3×3` 卷积，`stride=2` | 每次把特征图约缩小一半，并学习局部特征 |
| U-Net 解码上采样 | 双线性插值到明确目标尺寸 | 依次恢复到 `73×49`、`145×97`、`289×193` |
| U-Net 局部卷积 | `1×1` 投影和 `3×3` 残差卷积 | 调整通道并融合上采样特征与 skip |

从目标网格看，U-Net 编码尺度近似按 `/2 -> /4 -> /8` 变化：

```text
289×193
  -> 全分辨率融合网格 289×193
  -> stride-2 DownBlock 145×97
  -> stride-2 DownBlock 73×49
  -> stride-2 DownBlock 37×25
```

而 `output_patch_size=8` 通过一次 `8×8`、stride-8 卷积直接得到：

```text
289×193 -> pad to 296×200 -> PatchEmbed -> 37×25
```

二者得到同一 `37×25` 是有意的网格对齐，但实现方式不同：U-Net 走多尺度
卷积金字塔，上一深度场走一次 Patch Embedding。skip 的大小来自 U-Net 的
编码层级，而不是由 `output_patch_size` 分别切出来。

当前构造函数还明确限制 `output_patch_size` 必须为 8。若只把它改为 16，
上一深度场会生成 `19×13` token，但海表 U-Net 瓶颈仍是 `37×25`，两者会
因网格不一致而报错。要试验 patch size 16，需要同时重新设计海表编码金字塔
或增加对齐步骤，不能只改这一个数值。

### 5.4 旧版 patch-16 是怎样保持网格一致的

早期 B4 架构 v1 的 `output_patch_size=16`，但它并不是在当前结构上只修改
一个参数。旧版同时把海表 CNN 金字塔加深一级，使两条路径都在 `19×13`
对齐。

上一深度温度场路径为：

```text
289×193
  -> pad to 304×208
  -> 16×16、stride-16 Patch Embedding
  -> 19×13 = 247 tokens
```

旧版海表编码路径为：

```text
145×97
  -> 73×49
  -> 37×25
  -> 19×13 bottleneck
```

因此 patch-16 旧版比 patch-8 版本多一个 `37×25 -> 19×13` DownBlock，并把 `37×25` 也保存
为 skip。解码时再完整反向恢复：

```text
19×13
  -> upsample + 37×25 skip -> 37×25 head
  -> upsample + 73×49 skip -> 73×49 head
  -> upsample + 145×97 skip -> 145×97 head
  -> skip-free upsample -> 289×193 head
```

四个 head 的预测统一上采样到 `289×193`，再用四个 softmax 权重融合；权重
初始值各为 `1/4`。此前 patch-8 架构 v2 去掉了最深的编码/解码层和
`37×25` 预测 head，保留 `73×49`、`145×97`、`289×193` 三个 head；
架构 v3 曾只保留 `289×193` head。上一版 v4 恢复三个 head，同时增加
`289×193` 编码层与 skip；当前 v5 保留这个完整金字塔和旧版浅层宽度，
但只在最终 `289×193` 尺度设置一个输出 head。

| 对比项 | 旧版 patch-16（架构 v1） | 此前 patch-8（架构 v2） |
| --- | --- | --- |
| 上一深度 Patch Embedding | `16×16, stride=16` | `8×8, stride=8` |
| Transformer 网格 | `19×13` | `37×25` |
| token 数 `N` | 247 | 925 |
| 海表共享金字塔 | `145 -> 73 -> 37 -> 19` | `145 -> 73 -> 37` |
| skip 网格 | `145×97`、`73×49`、`37×25` | `145×97`、`73×49` |
| 输出 head | 4 个：`37`、`73`、`145`、`289` | 3 个：`73`、`145`、`289` |
| 位置编码 | `16×16 -> 19×13` | `16×16 -> 37×25` |
| 总参数量 | 4,285,224 | 3,325,350 |

这些版本的 `d_model=128`、4 层 Spatial Transformer、3 层 Depth Transformer、
因果深度注意力和 memory cross-attention 逻辑没有改变。v3 相比 v2
只改输出 head，默认参数量从 3,325,350 降为 3,325,153。v4 曾将浅层
通道收窄为 32，参数量为 3,298,534；当前 v5 恢复旧版 64 通道浅层宽度，
参数量为 3,549,793。高分辨率的激活也会显著增加显存需求。

patch-16 的优点是 token 少，空间注意力和 cross-attention 的计算量更低；
缺点是 Transformer 只在 `19×13` 的较粗网格上建模。patch-8 把 token 网格
提高到 `37×25`，更接近 B3 的设置并保留更多空间结构，但注意力计算和显存
开销更大。

### 5.5 连续深度编码

GLORYS 深度并非等间距。模型先对真实深度值做 `log1p` 压缩和归一化，再用
两层 MLP 映射为 128 维向量。这样 `0.49 m`、`5.08 m`、`47.37 m`、
`266.04 m` 等层使用的是各自真实深度信息，而不只是“第几个输出层”。

### 5.6 因果深度注意力

Depth Transformer 包含 3 个解码层。对每个固定水平 patch，模型把由浅到深
的 token 看成一条深度序列，并使用因果 mask：第 `k` 层只能关注第 0 到
第 `k` 层，不能读取未来更深层的信息。

这使模型学习“上层结构如何传递到下层”，同时保证训练时的信息流与推理时
从浅到深生成的顺序一致。

### 5.7 对海表 memory 的 cross-attention

每个深度解码层的处理顺序为：

```text
causal depth self-attention
  -> surface-memory cross-attention
  -> feed-forward network
```

对当前深度，925 个 query token 可以读取全部 925 个海表 memory token，
因此并非只能读取同一水平位置。memory 作为 cross-attention 的 key 和 value，
深度隐状态作为 query。每个解码层都包含残差连接、LayerNorm 和 dropout。

输出隐状态为：

```text
(B,D,925,128)
```

### 5.8 Teacher Forcing 与推理

训练时，`X_{k-1}` 可以使用上一深度真值，也可以使用模型上一层预测值。
teacher-forcing ratio 决定每个样本使用真值的概率；完全 teacher forcing 时可
并行构造所有深度 query。推理时没有真值，始终把上一层预测结果重新做
`8×8` patch embedding，作为下一深度的输入，直到生成最后一层。

## 6. U-Net 解码器：每一层 skip 的准确尺寸

Transformer 输出先从 token 形式还原为：

```text
(B,D,925,128) -> (B,D,128,37,25)
```

解码时暂时把 batch 和深度合并为 `B×D`，因此同一组卷积参数服务于所有
深度。编码器 skip 原本只有 `(B,C,H,W)`，模型会沿深度维复制，使同一天的
海表 skip 被每个目标深度共享。

### 6.1 解码阶段 1：37×25 -> 73×49

```text
Transformer bottleneck: (B×D,128,37,25)
  -> 双线性上采样到 73×49
  -> 1×1 通道投影: 128 -> 128
  -> 与 (B×D,128,73,49) skip 在通道维拼接
  -> (B×D,256,73,49)
  -> ResidualConvBlock: 256 -> 128
  -> decoded feature: (B×D,128,73,49)
```

这一阶段补回编码器在 `73×49` 尺度保留的中尺度空间结构。

### 6.2 解码阶段 2：73×49 -> 145×97

```text
decoded feature: (B×D,128,73,49)
  -> 双线性上采样到 145×97
  -> 1×1 通道投影: 128 -> 64
  -> 与 (B×D,64,145,97) skip 在通道维拼接
  -> (B×D,128,145,97)
  -> ResidualConvBlock: 128 -> 64
  -> decoded feature: (B×D,64,145,97)
```

这一阶段融合的是多源海表特征在共同经纬度网格上的较高分辨率表示。

### 6.3 最终阶段：145×97 -> 289×193

```text
decoded feature: (B×D,64,145,97)
  -> 双线性上采样到 289×193
  -> 1×1 通道投影: 64 -> 64
  -> 与 (B×D,64,289,193) skip 在通道维拼接
  -> ResidualConvBlock: 128 -> 64
  -> 1×1 通道投影: 64 -> 32
  -> ResidualConvBlock: 32 -> 32
  -> 唯一输出 head: (B×D,1,289,193)
```

最终阶段使用融合后的 `289×193` 海表特征作为全分辨率 skip，再通过卷积
结合上一尺度的解码特征生成预测。

### 6.4 为什么 U-Net 使用双线性上采样

U-Net 解码端的双线性上采样与输入端的经纬度重网格不是同一个操作。这里
使用 `F.interpolate(..., size=目标尺寸, mode="bilinear",
align_corners=False)`，原因是：

- `37×25`、`73×49`、`145×97`、`289×193` 都含奇数边长，简单乘 2 不会
  始终得到 skip 的精确尺寸；直接指定目标大小可以保证拼接前完全对齐。
- 双线性上采样本身无参数，避免转置卷积可能产生的棋盘格伪影。
- 上采样只负责平滑恢复空间采样点，后续的 `1×1` 投影和残差卷积负责学习
  通道组合、局部纹理和误差修正。
- 相比最近邻，上采样后的特征连续性更好；相比高阶插值，计算更轻且不容易
  在温度异常边缘产生过冲。

这里所谓“恢复高分辨率”并不意味着插值凭空创造细节。真正的细节来源包括
`73×49`、`145×97`、`289×193` 三条编码器 skip，以及上采样后的可学习卷积。

## 7. 单 head 输出

`73×49` 和 `145×97` 阶段只融合特征，不提前预测温度。最终 `289×193`
特征融合了全部三条 skip，再由唯一的 `1×1` head 输出当前深度温度场。
不再有中间预测图、`scale_logits` 或多尺度预测加权。不同深度仍共用
解码器和最终 head 参数。

## 8. 完整张量流

```text
SST (B,1,480,320)
  -> native CNN -> (B,64,480,320)
  -> coordinate-aware bilinear regrid -> (B,64,289,193) --+
                                                              |
SSH/SSS (B,2,192,128)                                        |
  -> native residual CNN -> (B,64,192,128)                    |
  -> coordinate-aware bilinear regrid -> (B,64,289,193) ------+
                                                              |
                        concat + convolutional fusion --------+
                                                              v
full-resolution skip (B,64,289,193)
  -> DownBlock
high-resolution skip (B,64,145,97)
  -> DownBlock
middle-resolution skip (B,128,73,49)
  -> DownBlock
bottleneck (B,128,37,25)
  -> flatten + spatial position
  -> 4-layer Spatial Transformer
surface memory M (B,925,128)

BOS / PatchEmbed(previous temperature)
  + continuous depth embedding
  + spatial position embedding
  -> 3-layer causal Depth Transformer
     - depth self-attention
     - cross-attention to M
     - FFN
decoded tokens (B,D,925,128)

decoded tokens
  -> reshape (B×D,128,37,25)
  -> upsample + concat 73×49 skip -> (B×D,128,73,49)
  -> upsample + concat 145×97 skip -> (B×D,64,145,97)
  -> upsample + concat 289×193 skip -> (B×D,64,289,193)
  -> 1×1 projection + residual convolution -> (B×D,32,289,193)
  -> single 289×193 output head
  -> temperature field (B,D,289,193)
```

## 9. 默认结构参数

| 配置 | 默认值 |
| --- | ---: |
| Transformer hidden dimension | 128 |
| Attention heads | 8 |
| Spatial Transformer layers | 4 |
| Depth Transformer layers | 3 |
| FFN hidden dimension | 512 |
| Dropout | 0.1 |
| Native CNN base channels | 32 |
| Fusion channels | 64 |
| Previous-field patch size（不是 U-Net patch size） | 8 |
| Learnable position base grid | `16×16` |
| Transformer token grid | `37×25` |
| Transformer token count | 925 |
| U-Net encoder downsampling | 三个 `3×3, stride=2` DownBlock |
| U-Net decoder upsampling | 指定目标尺寸的双线性插值 |
| U-Net skip grids | `73×49`、`145×97`、`289×193` |
| Full-resolution skip | 有，64 通道 |
| Output heads | 仅 `289×193` |

## 10. 实现文件对应关系

| 文件 | 作用 |
| --- | --- |
| `models/b4_cplus/model.py` | 串联海表编码、空间 memory、深度自回归和 U-Net 重建 |
| `models/b4_cplus/unet.py` | 原生网格 CNN、共享编码金字塔、三条 skip、单 head U-Net 解码 |
| `models/b4_cplus/regrid.py` | 基于物理经纬度的双线性特征重网格 |
| `models/b4_cplus/embeddings.py` | `8×8` patch embedding、二维位置编码和连续深度编码 |
| `models/b4_cplus/decoder.py` | 因果深度 self-attention、memory cross-attention 和 FFN |

最容易混淆的四点是：

1. `output_patch_size=8` 只直接作用于上一深度温度场的 patch embedding，
   当前模型不存在单独的“U-Net patch size”；U-Net 使用 stride-2 卷积形成
   多尺度金字塔；
2. `16×16` 是可学习二维位置编码的基础参数网格，进入 Transformer 前通过
   bicubic 插值成 `37×25`；它不是输入特征的 patch grid；
3. 输入端的双线性 `grid_sample` 是按经纬度完成跨数据源对齐，解码端的双
   线性 `interpolate` 是为了把 U-Net 特征恢复到指定张量尺寸；
4. 海表 memory 与三条 skip 都只编码一次，并在所有目标深度之间共享；每个
   深度都会运行同一套 U-Net 解码权重，但输入的深度隐状态不同。
