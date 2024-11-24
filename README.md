# High-throughput mesoscopic optical imaging data processing and parsing using differential-guided filtered neural networks
** We design a efficient deep differential guided filtering module (DDGF) by fusing multi-scale iterative differential guided filtering with deep learning, which effectively refines image details while mitigating background noise. Subsequently, by amalgamating DDGF with deep learning network, we propose a lightweight automatic segmentation method DDGF-SegNet, which demonstrates robust performance on our dataset, achieving Dice of 0.92, Precision of 0.98, Recall of 0.91, and Jaccard index of 0.86. **
## **一、课题调研**
**自动小鼠器官分割：基于深度学习的解决方案**

目前，在这样的小鼠横切面图像中，**器官是手动识别**的，这是一个**劳动密集型**的过程，**耗时且不健壮**，这导致需要使用大量的实验动物。针对这一问题，我们提出了一种基于深度学习的器官分割解决方案，利用该解决方案，我们可以对**感兴趣的关键器官实现高精度的自动器官分割**（骰子系数在0.83-0.95之间，具体取决于器官）。

近年来，计算机视觉和自动图像分析领域取得了重大进展，特别是在深度学习的推动下【 G. Litkens, T. Kooi, B. Bejnordi, A. Setio, F. Ciompi, M. Ghafoorian, J. Laak, B. v. Ginneken and C. Sánchez, "A survey on deep learning in medical image analysis," Medical Image Analysis, Vol 42, pp. 60-88, 2017.】。从识别猫和狗到高度准确的MRI或数字病理图像分类和识别，深度学习被证明是一种有效且可扩展的技术，可用于复杂的医学图像分析任务。

**Full-body deep learning-based automated contouring of contrast-enhanced murine organs**

**for small animal irradiator CBCT基于全身深度学习的小动物CBCT对比增强小鼠器官自动轮廓**

在一秒钟内实现小鼠腹部、胸部和骨骼结构的精确自动轮廓可以花很多小时（13,14）。这样的时间不适合大规模的小鼠辐射生物学研究，因为每天需要许多小鼠的分娩才能产生统计显著性的结果。

动物模型是生物医学和临床前研究的许多领域的支柱，而小鼠是研究人类疾病最常用的模式生物（2-5）。详细了解和表征小鼠模型被认为是提高人类受试者临床前结果可重复性的关键(6)。人工器官分割是最耗时、最费力的工作之一。根据部位的不同，简单的单一器官轮廓（大脑、心脏等）可能需要大约20分钟，而对于涉及许多器官或骨骼轮廓的高度复杂的病例，则需要进行全骨髓照射，此外，在体积CT扫描的每个切片上手工描绘器官不仅需要高度关注细节，还需要高水平的小鼠解剖专业知识。此外，由于个人偏见，人工分割存在主观性。

  

![](/files/01933cfd-4d15-7bb3-bf0b-a8b8f18cf5cc/image.png)

  
图1：分割文件（左）与人工智能轮廓文件（右）并置的 Human ground truth 3D投影。GTV是深绿色的，左肾是橙色的，右肾是金色的，脊髓是水蓝色的，肠，胃是蓝色的，肝是紫色的，肠是品红的，心是浅蓝色的，左肺是亮绿色的，右肺是黄绿色的，骨头是粉红色的。

![](/files/01933cff-00be-7bb3-bf0b-b21a23a0f4f9/image.png)

图2:15只小鼠测试组中1只小鼠的冠状面和矢状面二维幻灯片。ground truth 的人类分割显示为一个粗壮的黄色轮廓。人工智能自动分割显示在突出显示的颜色中。GTV是深绿色的，左肾是橙色的，右肾是金色的，脊髓是水蓝色的，肠，胃是蓝色的，肝是紫色的，肠是品红的，心是浅蓝色的，左肺是亮绿色的，右肺是黄绿色的，骨头是粉红色的。

**Exploring Automated Contouring Across Institutional Boundaries:A Deep Learning Approach with Mouse Micro-CT Datasets**

**探索跨越机构边界的自动轮廓：使用小鼠微ct数据集的深度学习方法**

![](/files/01933d03-f170-7bb3-bf0b-bfd909aa209a/image.png)

**Deep learning based multi-organ segmentation and metastases segmentation in whole mouse body and the cryo-imaging cancer imaging and therapy analysis platform (CITAP)**

**基于深度学习的全鼠多器官分割和转移瘤分割及冷冻成像癌症成像与治疗分析平台（CITAP）**

![](/files/01933d05-dbac-7bb3-bf0b-c3579440ef3d/image.png)

**图6所示。小鼠全身器官分割和转移分布的三维可视化**。肺和肝分别用红色和蓝色表示。尺寸&lt; 0.5mm、0.5mm - 2mm和&gt;2mm的转移灶分别为黄色、红色和绿色。我们分别发现78例（1例为>2mm， 21例为0.5mm-2mm， 56例为&lt;0.5mm）和24例（1例为&gt;2mm， 11例为0.52mm， 12例为<0.5mm）肺转移灶和肝转移灶。

## 二、课题梳理

### **基于fMOST系统成像的小鼠全身多器官组织分割重建研究**

![](/files/01935c88-f092-7bb3-bf0e-501f003ad373/image.png)

在猪体内培养出实体人源器官登上 Cell 子刊封面，反映了**动物模型是许多生物医学和临床前研究领域的支柱**，其中**小鼠是研究人类疾病最常用的动物,详细了解和描述小鼠模型被认为是提高人类应用可重复性的关键**。对于广泛的研究领域，从癌症到器官病变研究，从放射研究到药物输送和纳米颗粒摄取，对获取的图像数据进行定量和比较分析**需要对小鼠解剖结构进行分割**，主要器官和其他感兴趣的结构的描绘允许从图像数据中提取定量信息，例如器官形状和大小、药物摄取、来自生物标志物的信号或转移分布。

传统上，这项任务是通过在体积扫描的每个切片中用多边形勾勒出器官轮廓来手动完成的。这是一个**劳动密集**型的过程，耗时且不健壮，并**需要解剖学和成像方式的专业知识量**。为了减少手动分割所需的时间，我们希望设计一种**器官分割解决方案，对感兴趣关键器官实现高精度自动器官分割。**

![](/files/01935c8d-fc66-7bb3-bf0e-5d81d925a9d1/image.png)

同时，器**官含有多种血管、胞体等微观结构，参与各种生理功能**，**为研究器官病变过程中的微观结构变化，有必要单细胞分辨成像方法获得整个器官的血管、胞体的三维形态。**

**感兴趣区域**：肝内重建血管、胆管和淋巴管的结合。门静脉呈青色。**肝静脉**呈黄色。胆管呈紫色。肝动脉呈红色。肝内淋巴管以橙色标记。肾脏由肾单位和集合管组成，肾单位包括**肾小球**、肾小管及球旁器。肾小球集中分布与皮质，由毛细血管簇构成，外面包裹着包曼氏囊。小鼠的**十二指肠**长度很短，肠绒毛密集，呈叶片状，杯状细胞较少，黏膜下层可见大量布伦内氏腺。

![](/files/01935c92-aa0d-7bb3-bf0e-64bdc12f4cd9/image.png)

**研究现状**：

同类技术中**AIMOS**做的比较成熟，但是是**低分辨率的CT成像数据**；

**MOST团队**当前工作对**单细胞分辨率的眼球血管**、**心脏血管**、**肝脏血管**（同时可视化肝脏内多种脉管结构）进行了重建，揭示各种器官的真实形态，**其他器官还在探索…**

![](/files/01935d0c-32f1-7bb3-bf0e-80e002adfad6/image.png)

**研究内容及技术方案**

希望建立一种小**鼠全身（全器官&lt;脑、心脏、肝脏、肺、肾和脾&gt;及组织）自动分割方案**

利用**fMOST亚微米分辨率的数据优势，分割重建可视化外轮廓以外的器官内腔以及组织层面的结构、细胞与器官之间的细节**

![](/files/01935d0d-8dfc-7bb3-bf0e-8f977b93695c/image.png)

## **目前要做的：**

**1.结合****fMOST亚微米分辨率****数据需求梳理主流器官分割技术如AIMOS、SAM****、****U-Net等**

**2.对技术路线本身要有独创性的智力贡献，如数据准备的处理、人在回路等**

**3.找器官数据集，学会用别人数据，借用标签，做预实验**

**4.调研现有工作是否有全标签数据集**
