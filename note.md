<h1 align="center">Paper Reading Notebooks </h1>

<table>
	<tr>
	    <th>Filed</th>
	    <th>Paper</th>
	    <th>Date</th>  
	</tr >
    <tr >
	    <td rowspan="2"><b>CV</b>: Image</td>
	    <td>VTs-Drloc</td>
	    <td>22-11-24</td>
	</tr>
	<tr>
	    <td>SPT_LSA_ViT</td>
	    <td>22-11-25</td>
	</tr>
	<tr >
	    <td rowspan="3"><b>CV</b>: 3D Construction</td>
	    <td>TransformerFusion</td>
	    <td>22-11-08</td>
	</tr>
	<tr>
	    <td>Neural Deformation Graphs</td>
	    <td>22-11-09</td>
	</tr>
	<tr>
	    <td>Optimize Non-Rigid Tracking</td>
	    <td>22-11-10</td>
	</tr>
	<tr>
        <td rowspan="1"><b>NLP</b>: Retrieval</td>
	    <td>Neural Corpus Indexer</td>
	    <td>22-12-21</td>
	</tr>
	<tr>
	    <td rowspan="1"><b>Audio</b></td>
	    <td>Whisper</td>
	    <td>22-11-17</td>
	</tr>
</table>

# CV - Image

<h3>Efficient Training of Visual Transformers with Small-Size Datasets</h3>

- 【NeurlIPS2021】 [ArXiv](https://arxiv.org/abs/2106.03746)  [Code](https://github.com/yhlleo/VTs-Drloc)
- 简介：使用小数据集优化训练Visual Transformer，训练加速，泛化能力增强。
- 关键技术：
  1. 验证实验SOTA VTs(CvT、Swin、T2T)在小数据集上效果不好
  2. VT由于缺少卷积归纳偏置，设计自监督代理任务，从图片中提取额外的信息学习空间关联，增加dense relative localization loss($L_{drloc}$)，即插即用。
- Limitation：fine-grained嵌入网格效果不好

<div align="center">
  <img src="Image/22-11-24VTs-Drloc.png">
</div>
<br>

---
<h3>Vision Transformer for Small-Size Datasets</h3>

- 【2021】 [ArXiv](https://arxiv.org/abs/2112.13492)  [Code](https://github.com/aanna0701/SPT_LSA_ViT)
- 简介：使用SPT+LSA解决由于Vision Transformer缺少局部归纳偏置不能在小数据集上训练的问题
- 关键技术：
  1. Shifted Patch Tokenization：利用邻接像素空间关系，扩大感受野
  2. Locality Self-Attention Mechanism：使用Diagonal Masking增加不同token之间的注意力分数 + 通过Learnable Temperature Scaling控制输出分布的平滑度
  
<div align="center">
  <img src="Image/22-11-25SPT_LSA_ViT.png">
</div>
<br>

# CV - 3D Reconstruction

<h3>TransformerFusion: Monocular RGB Scene Reconstruction using Transformers</h3>

- 【NeurlIPS2021】 [ArXiv](https://arxiv.org/abs/2107.02191)  [Code](https://github.com/AljazBozic/TransformerFusion)
- 简介：Transformer在单RGB Video室内场景三维重建中的应用
- 关键技术：
  1. Coarse-to-fine融合：coarse重建全局场景，fine只重建接近表面处的精细特征，最后将特征融合解码为更高分辨率场景。
  2. 多视图特征融合：每次最多使用K张图片训练，加载超过K张图片时去除attention权值最小的图片，一直保持使用K张图片； 
- Limitation: 遮挡、不完全场景、透明物体重建鲁棒性差。未来研究方向可以使用自监督损失，通过稀疏卷积和局部几何先验获得更高分辨率的几何保真。
  
<div align="center">
  <img src="Image/22-11-08TransformerFusion.png">
</div>
<br>

---
<h3>Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction</h3>

- 【CVPR2021】[Paper](https://aljazbozic.github.io/neural_deformation_graphs/) [Code](https://github.com/AljazBozic/NeuralGraph)
- 简介：使用GNN进行非刚性4D重建
- 关键技术：
  1. 全局+局部优化，损失分别计算；全局优化所有帧变形图，局部多MLP表示框
  2. 使用单帧图像多视图一致+变形表面一致来计算循迹和变形
- Limitation: 输入特定为64^3的SDF网格；纹理特征不鲁棒；未来可开展稀疏3D卷积和其他特征如颜色重建损失计算工作。

<div align="center">
  <img src="Image/22-11-09NDG.png">
</div>
<br>

---
<h3>Learning to Optimize Non-Rigid Tracking</h3>

- 【CVPR2020】[ArXiv](https://arxiv.org/abs/2003.12230)
- 简介：RGBD非刚性循迹图网络的收敛优化
- 关键技术：
  1. 使用CNN端到端学习深度特征匹配，使得高斯牛顿求解器可以解决大非刚性变形场景
  2. ConditionNet预处理求解器，增加PCG求解速度
- Limitation：3D场景遮挡问题建议直接从3D数据中提取特征；场景流真实数据难获取，建议在合成数据集上学习
- Trick：数据增强；深度图滤波预处理

<div align="center">
  <img src="Image/22-11-10Optim_NRT.png">
</div>
<br>

---
# NLP - Retrieval

<h3>A Neural Corpus Indexer for Document Retrieval</h3>
- 【NeurlIPS2022】[ArXiv](https://arxiv.org/abs/2206.02743v1)
- 简介：基于Transformer的sequence-to-sequence架构，给定qurey生成相关文档id
- 关键技术：
  1. 和DSI一样，是端到端的文档检索模型
  2. prefix-aware weight-adaptive (PAWA) 解码器生成文档id
  3. 基于对比学习的一致性正则损失
- Limitation：模型过大不利于部署；检索速度有待提高；model-based难以进行新文档更新
- 参考：[沐神论文精读](https://www.bilibili.com/video/BV1Se411w7Sn/?spm_id_from=333.788&vd_source=486265fa677326a8f53894f05277bfb9)

<div align="center">
  <img src="Image/22-12-21NCI.png">
</div>
<br>

---
# Audio

<h3>Robust Speech Recognition via Large-ScaleWeak Supervision</h3>

- OpenAI [Arxiv](https://cdn.openai.com/papers/whisper.pdf)
- 简介：基于Transformer通过大尺度弱监督学习自动语音识别（ASR，Automatic Speech Recognition）模型，模型可以不微调直接进行zero-shot迁移。
- 关键技术：
  1. 数据预处理：从网络上收集了68万小时的多语言（98 种语言）和多任务（multitask）监督数据对Whisper进行了训练。
               预处理使用了三种自动过滤方法：检测并删除机器生成的转录；使用语音检测器确保语言和转录匹配；识别并删除低质量数据。
  2. 模型：基于encoder-decoder的Transformer架构，其中解码器通过训练不同特殊的token识别单个任务，以此实现多任务统一训练。
- Limitation：由于使用现成的Transfomer架构并没有进行过多改进，会出现错误结果。可以对现有模型的解码策略、微调、正则化、数据增强、数据多样性、增加预训练等进行改进。
- 参考：[沐神论文精读](https://www.bilibili.com/video/BV1VG4y1t74x/?spm_id_from=333.999.list.card_archive.click&vd_source=486265fa677326a8f53894f05277bfb9)
       [知乎](https://zhuanlan.zhihu.com/p/568173245)

<div align="center">
  <img src="Image/22-11-17Whisper.png">
</div>
<br>

---
