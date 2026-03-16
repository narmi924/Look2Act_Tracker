论文是 **MPIIGaze / “Appearance-Based Gaze Estimation in the Wild” (Zhang et al., 2015)，**你当前系统是“普通 RGB webcam + face/landmark + eye crop + 3D gaze regression + PnP head pose + ray-plane + 9点仿射校准 + EMA 平滑 + 实时桌面交互”的完整链路，目标强调 CPU-only、实时部署和真实桌面使用。

## **1. 一句话判断**

这篇论文对 Look2Act 的最大价值，不在于你直接照搬它的网络，而在于它为你提供了三样非常关键的“研究支架”：**真实世界桌面 gaze 的数据集叙事、跨域泛化问题定义、以及 head-pose-aware 的标准化建模链条**。

更具体地说，它最适合帮助你的部分是：**dataset motivation / related work / experiment 设计 / 投稿叙事**，其次才是方法层面对“头姿 + 眼图像 + 归一化空间”的启发。它不太适合作为你当前 Look2Act 的直接工程蓝图，因为它的输出是**归一化空间中的 gaze angle**，而你的系统最终要落到**屏幕点、几何映射和交互层**。

## **2. 面向 Look2Act 的精确摘要**

这篇论文解决的核心问题是：**过去 appearance-based gaze estimation 大多在受控实验室条件下评估，缺少真实日常 laptop 使用环境下的数据与方法验证**；因此作者构建了 MPIIGaze，包含 15 名参与者、跨 3 个多月、共 213,659 张图像，强调真实光照、真实日常使用、真实设备差异与外观变化。

它和你的系统高度相关，原因不是“你们都做 gaze”这么泛，而是更具体的三点：

第一，它和你的 **桌面/笔记本真实使用场景** 高度一致。Look2Act 本身就是普通笔记本摄像头、桌面屏幕、非红外、面向真实交互的系统；MPIIGaze 也是用 laptop 作为采集平台，在自然日常环境中记录 gaze。

第二，它和你 README 中的 **head pose + eye crop + camera/screen geometry** 思路接近。论文方法是：face/landmark detection → generic face model 拟合 → head pose estimation → 数据归一化 → eye image + head angle 输入 CNN → 输出 gaze angle。你的系统链路是：face/landmark → eye crop → PnP head pose → gaze regression → 坐标变换 → ray-plane 求交 → calibration。两者在前半段很接近，但在后半段分叉明显：论文停在 gaze estimation，Look2Act 继续走到屏幕点与交互。

第三，它和你 README 中最远的部分，是 **screen projection / interaction layer / realtime deployment**。MPIIGaze 不是系统部署论文，不关心端到端 FPS、CPU-only、屏幕物理尺寸到像素映射，也不关心 dwell click、时序平滑或桌面控制。你的 README 在这些系统层部分已经走得比它更完整。

## **3. 可迁移到我项目中的核心点（按优先级排序）**

### **3.1 可以直接借鉴的 3–5 个点**

**(1) “in-the-wild laptop gaze” 的问题定义与叙事。**  
这篇论文是你写 Introduction / Related Work 时非常好的支点：它明确指出，实验室数据不足以代表真实日常 laptop 使用，真实场景的光照与外观变化显著更复杂。你做 Look2Act 时完全可以用它来支撑“真实桌面 gaze 交互仍然困难”的论证。

**(2) 头姿参与 gaze 估计是必要的。**  
作者的多模态 CNN 将 **head angle vector h** 与眼图像特征联合建模，并在实验中明确显示：去掉 head pose 输入会使性能变差，说明 pose-independent/person-independent 场景里头姿是关键因素。这个结论和你 README 里单独做 PnP head pose 的设计高度对齐。

**(3) 归一化空间的想法。**  
论文通过固定相机距离、固定 focal length、头坐标系对齐等方式，把原始观测归一化到统一空间，再学习 eye appearance 到 gaze 的映射。这不要求你完全照搬，但它给了你一个很好的研究表述：你可以把 Look2Act 的 eye crop、head pose、坐标变换写成“降低外观变化、将估计问题转移到统一几何空间”的一部分。

**(4) 跨数据集 / 跨域评估意识。**  
他们专门比较 cross-dataset 与 within-dataset，并指出跨域时性能显著下降，核心原因之一是外观与光照分布差异。这个思路对你后续做 “公开数据预训练 + 自采数据微调 + 实机评估” 非常有价值。

**(5) 光照条件误差分析的展示方式。**  
他们不仅给出平均误差，还把误差随 face region 灰度强度、左右光照差异的变化画出来。你完全可以借这种图表思路，为 Look2Act 增加更像论文的误差分析，而不只是报 overall mean error。图 9 的思路很值得借。

### **3.2 可能需要改造后再借鉴的 2–4 个点**

**(1) 归一化后的 gaze angle 回归。**  
论文输出的是归一化空间中的 2D yaw/pitch gaze angle，而你当前 README 的主线是直接输出 3D 单位 gaze 向量，再通过几何模型投到屏幕。你可以借它“归一化+回归”的思想，但不要机械换成 2D angle output，否则会打乱你现有的 3D→ray-plane 链路。

**(2) 左右眼镜像统一建模。**  
作者通过水平翻转和 around y-axis 镜像处理，把左右眼统一到单一回归函数里。你的 README 目前是左右眼 batch=2 再推理，然后后续融合。你可以把论文思路作为一个对照实验：单眼统一建模 vs 双眼分别推理 / 双眼平均。

**(3) 通用人脸形状模型。**  
论文用 generic mean facial shape 做 3D 头姿估计，这是现实约束下的务实方案。你现在 PnP 也是基于固定 3D face model 点集，这和它相通。但要注意：这只是 practical approximation，不要在论文里把它写得像“精确个体3D头模”。

### **3.3 不建议我借鉴或当前阶段不适合借鉴的点**

**(1) 直接照搬 LeNet 式小网络。**  
这篇论文是 2015 年，网络非常早期，输入只有 60×36 灰度眼图，2 个卷积层 + FC。它的历史意义大于结构价值。你当前轻量 CNN 设计已经更贴近你的部署目标。

**(2) 把它的结果数值拿来和你的结果直接横比。**  
它的评估空间、标签定义、归一化方式、训练集选择都和你的 Look2Act 不同。你可以引用它说明“真实世界很难”，但不要暗示你的 7.3° 与它的 6.3°/13.9° 是同口径比较。

**(3) 把 MPIIGaze 当成与你的真实桌面交互完全同分布的数据。**  
它虽然是 laptop in-the-wild，但目标仍是受提示的 fixation 数据采集，不是持续真实桌面交互中的自然 gaze-control 流。这个 gap 很大，不能硬说“与我的应用完全一致”。

## **4. 方法与公式层面的提炼**

### **4.1 关键变量定义**

从论文可提炼的核心变量是：

* ( e )：归一化后的 eye image
* ( h )：2D head angle vector
* ( g )：2D gaze angle vector（yaw, pitch）
* ( r )：3D 头部旋转
* ( t )：眼部位置 / 眼角中点在相机坐标系中的位置  
这些变量构成它的标准化 gaze estimation 表述。

你的系统已有相似变量，只是表示更偏 3D：

* ( \\mathbf{g}\_{local} \\in \\mathbb{R}^3 )：眼局部坐标系 gaze unit vector
* ( R, t )：由 PnP 得到的旋转和平移
* ( \\mathbf{d}*{cam} = R \\mathbf{g}*{local} )：变换到相机坐标系的 gaze direction
* 屏幕平面 ( (P\_0, N) )
* 屏幕交点 ( P = O + \\tau D )  
这套链路在你 README 中已经写得很清楚。

### **4.2 核心建模思路**

论文的核心不是一个复杂公式，而是一条很标准的建模链：

1. 从原图得到 face/landmark
2. 用 generic face model 估计 head pose
3. 通过几何归一化减少不必要的外观变化
4. 用 CNN 学习 ( (e, h) \\to g ) 的映射。

这对你可转写成一个更适合 Look2Act 的“简化科研表达”：

\[  
\\hat{\\mathbf{g}}*{local} = f*\\theta(I\_{eye}, \\mathbf{h})  
]

其中 (I\_{eye}) 是眼部裁剪图像，(\\mathbf{h}) 是头姿特征（可显式输入，也可通过前置 head pose 模块提供）。然后：

\[  
\\hat{\\mathbf{g}}*{cam} = R \\hat{\\mathbf{g}}*{local}  
]

\[  
P\_{screen} = O + \\tau \\hat{\\mathbf{g}}*{cam}, \\quad \\tau = \\frac{N^\\top(P\_0-O)}{N^\\top \\hat{\\mathbf{g}}*{cam}}  
]

最后再由校准映射：

\[  
\\mathbf{p}*{cal} = A\[\\mathbf{p}*{raw};1]  
]

这组式子是你现在系统真正能支撑的，不是硬凑。

### **4.3 适合我项目的数学表达**

#### **A. 可以真实写进我项目方法章节的公式**

**(1) gaze regression + 头姿辅助输入**  
\[  
\\hat{\\mathbf{g}}*{local} = f*\\theta(I\_{L}, I\_{R}, \\mathbf{h})  
]  
如果你当前实现还没有显式把 (\\mathbf{h}) 喂给网络，也可以写成更保守的版本：  
\[  
\\hat{\\mathbf{g}}*{local} = f*\\theta(I\_{L}, I\_{R})  
]  
然后在 Discussion 里说 head pose is estimated and used in downstream geometric projection，而不是说网络显式融合了 head pose。因为你的 README 目前看，head pose 是独立模块，不是明确的网络输入。

**(2) 相机坐标系变换**  
\[  
\\hat{\\mathbf{g}}*{cam} = R \\hat{\\mathbf{g}}*{local}  
]  
这是你现有链路的核心。

**(3) 射线-平面求交**  
\[  
\\tau = \\frac{N^\\top(P\_0-O)}{N^\\top D}, \\qquad P = O + \\tau D  
]  
你 README 已明确给出。

**(4) 仿射校准**  
\[  
\\mathbf{p}*{cal} = A\[\\mathbf{p}*{raw};1]  
]  
这是你当前校准模块的真实数学表达。

#### **B. 只能作为 related work 理解的公式**

**(1) MPIIGaze 的 gaze angle regression**  
\[  
\\hat{g} = f(e, h)  
]  
这里的 (g) 是归一化空间中的 yaw/pitch，不是你系统最终屏幕点，也不是你当前 3D 单位向量输出，适合拿来解释“相关工作如何表述 gaze estimation”，不适合冒充你的方法。

**(2) 归一化相机 / 固定距离 / 固定焦距的标准化过程**  
这套过程适合写成“受相关工作启发，我们采用/考虑统一观测空间以减弱头姿与尺度变化”，但你如果没有真的实现严格的归一化渲染/warp，就不要把他们那套标准化写成你的已实现方法。

#### **C. 不建议我照搬的公式**

**(1) 论文中最终用的 2D L2 gaze angle loss**  
\[  
L = |\\hat{g} - g|\_2^2  
]  
你当前 README 已是 3D 单位向量 + angular loss，这和你的几何链更一致。除非你真的换输出定义，否则没必要退回去。

### **4.4 顺手指出你 README 里一个可改进点**

你 README 在 “Look2Act vs MPIIGaze” 的相关工作表格里写 “MPIIGaze 单眼图像 → 2D gaze，GPU，\~4.5°，离线” 这一行，我建议你谨慎。MPIIGaze 这篇论文本身更准确的定位是：**提出 in-the-wild dataset + multimodal CNN benchmark**，其文中关键结果包含跨数据集 13.9°、同数据集 leave-one-person-out 6.3° 等；你表格里那种单一 “\~4.5°” 数值很容易和别的 setting 混淆，最好改成更明确的“reported under specific setting / not directly comparable”。

## **5. 和我当前系统架构的映射**

### **face / landmark**

**直接相关。**  
MPIIGaze 方法从 face detection 和 facial landmark detection 起步，并依赖 landmarks 做 head pose 与 eye normalization。你的系统同样从 MediaPipe FaceMesh 开始，这一段高度对齐。

### **eye crop / face crop**

**直接相关。**  
论文使用归一化后的 eye image 作为主要输入；你的系统使用 128×128 左右眼 crop。这是最接近的模块之一。区别在于：论文强调 normalised eye image compatibility across datasets；你的 eye crop 目前更偏工程裁剪与部署。

### **gaze regression**

**直接相关，但表示不同。**  
论文是 2D gaze angle regression；你是 3D gaze unit vector regression。两者都属于 appearance-based gaze estimation，但中间表示不同。这个差异决定了它更适合给你“思路和叙事”，不是直接换架构。

### **head pose**

**高度相关。**  
MPIIGaze 明确把 head angle 作为输入，并验证其必要性。你的系统独立估计 head pose 并用于几何变换。这一环是最值得引用和对齐的。

### **geometry / ray-plane / coordinate transform**

**间接相关。**  
论文用几何做归一化，但并不走你这种 gaze ray 与 screen plane 求交的完整链。它和你的关系是：前者提供“统一几何空间”的研究思想，后者是你自己的系统落地路径。

### **calibration**

**几乎无关。**  
MPIIGaze 强调 calibration-free/person-independent 学习方向，但不是在做你这种 screen-point affine calibration 模块。它可以作为“为什么我们仍然需要后置轻量校准”的反衬依据，但不能直接指导你的 9 点仿射设计。

### **smoothing / temporal stability**

**几乎无关。**  
论文不关注时序滤波、稳定性或交互 jitter；你的 EMA 是系统层设计。

### **screen projection / interaction layer**

**基本无关。**  
这篇论文停在 gaze estimation benchmark，不涉及 mm→pixel 映射、dwell click、cursor control 或 HCI interaction layer。

## **6. 对我论文写作最有用的引用建议**

### **6.1 最适合引用的位置**

最适合放在：

* **Introduction**：说明真实日常 laptop gaze estimation 的困难性与重要性
* **Related Work – Datasets / In-the-wild gaze estimation**：说明 MPIIGaze 的代表性
* **Related Work – Appearance-based + head pose-aware methods**：说明多模态 gaze estimation 的经典路线
* **Discussion / Limitation**：说明跨域、光照、外观变化仍是关键挑战

不太适合放在你的 **Method** 里作为“本方法直接来源”，除非你真的采用了它的 normalisation + head-angle-as-input 方案。

### **6.2 推荐我如何描述它（学术中文）**

可用表述 1：  
“Zhang 等提出了 MPIIGaze 数据集，首次系统性地面向日常 laptop 使用场景研究 appearance-based gaze estimation，显著提升了数据在光照、外观及采集时空条件上的真实性，为真实环境下的 gaze estimation 研究提供了重要基准。”

可用表述 2：  
“该工作进一步表明，跨数据集泛化在真实场景中仍然具有较大挑战，头部姿态信息与外观建模的联合利用对于提升 unconstrained gaze estimation 的鲁棒性具有重要意义。”

### **6.3 推荐我如何描述它（学术英文）**

Usable sentence 1:  
“Zhang et al. introduced MPIIGaze, one of the earliest large-scale in-the-wild laptop gaze datasets, highlighting the substantial appearance and illumination variability encountered in daily-use scenarios.”

Usable sentence 2:  
“Their results showed that cross-dataset generalization remains difficult under unconstrained conditions, and that combining eye appearance with head pose cues is beneficial for person-independent gaze estimation.”

### **6.4 1–2 句可直接改写后使用的 related work 句式**

句式 1：  
“与早期主要在受控实验室环境下评估的视线估计方法不同，MPIIGaze 将研究重点推进到日常 laptop 使用场景，揭示了光照变化、外观差异及跨域偏移对 appearance-based gaze estimation 的显著影响。”

句式 2：  
“MPIIGaze 所提出的多模态建模思路表明，仅依赖眼部外观难以充分应对 unconstrained 场景下的视线估计任务，头部姿态信息在 person-independent 设置中具有重要辅助作用。”

### **6.5 容易被误引的地方**

你最容易误引的有三处：

第一，不要把 MPIIGaze 误写成“一个屏幕交互系统”或“实时 gaze 控制框架”。它不是。  
第二，不要把它的结果和你的系统精度做直接 apples-to-apples 比较。  
第三，不要把它的“calibration-free/person-independent”叙事误写成“完全不需要任何后置映射或系统校准”。论文关心的是 estimator 学习，不是最终桌面控制系统。

## **7. 对我实验设计最有用的启发**

### **7.1 我可以从这篇论文中借哪些实验设置**

**(1) cross-dataset / cross-domain 视角。**  
你未必要真的在 MPIIGaze 上完整复现，但你可以设计：

* 公开数据预训练 → 自采 Look2Act 数据测试
* 自采 Look2Act 训练 → 跨 session / 跨 user / 跨 device 测试  
这会比只报随机划分 test 更像一篇论文。MPIIGaze 的核心贡献之一就是把“跨域评估”放到台面上。

**(2) leave-one-person-out 或至少 cross-user holdout。**  
你 README 已经规划了 cross-user generalization，这和 MPIIGaze within-dataset leave-one-person-out 很对齐。建议把它从“计划实验”尽快变成真实结果。

**(3) 光照相关误差分析。**  
借鉴 Figure 9 的思想，把误差按：

* 平均亮度
* 左右亮度不平衡
* 是否戴眼镜  
分桶分析。这会极大提升你实验部分的论文感。

### **7.2 我可以新增哪些对照 / 消融**

**最值得加的三个：**

1. **有/无 head pose 信息**  
不一定非得改网络，可以先做系统级对照：
* full pipeline
* 去掉或固定 head pose 几何补偿  
看看 pixel error / angular error 如何变化。MPIIGaze 已给出理论动机。
2. **公开数据预训练 vs 仅自采训练**  
如果你后面引入 MPIIGaze、UnityEyes 或其他公开数据，这个对照很关键。
3. **跨光照 / 跨 session / 跨时间段**  
MPIIGaze 的采集时间跨日常使用时段，你也可以按白天/晚上、室内强光/弱光等划分测试子集。

### **7.3 我可以补哪些图表 / 误差分析**

* error vs brightness
* error vs head yaw/pitch
* error vs distance to screen
* cross-session error bar
* user-wise generalization chart

其中前两个最直接受 MPIIGaze 启发，后三个更贴近你的系统目标。

### **7.4 它是否启发我补某一项**

最强烈启发的是：

* **cross-user**
* **cross-device / cross-domain**
* **calibration 前后的误差对比叙事中的“前置 estimator 难度”**
* **stability 之外的 robustness 评估**

它对 latency / usability 启发较少，因为那不是这篇论文的重点。

## **8. 图表与可视化建议（文本描述版本）**

### **建议图 1：公开数据与自采数据的 domain gap 图**

**表达什么信息：**  
展示公开数据集（如 MPIIGaze）与 Look2Act 自采桌面数据在采集场景、光照、头姿、目标空间、部署目标上的异同。  
**放置位置：** 主文 Related Work 或 Method 前半段。  
**图结构描述：**  
左侧为 MPIIGaze 样本特征摘要块：真实 laptop、长期采集、光照变化大、输出 gaze estimation；右侧为 Look2Act：普通 webcam、真实桌面交互、3D→screen projection、校准与部署；中间用箭头标注“shared challenges”和“remaining gaps”。

### **建议图 2：Look2Act 与 MPIIGaze 方法链对齐图**

**表达什么信息：**  
说明哪些模块继承了经典 in-the-wild gaze estimation 路线，哪些是你系统新增的部署层。  
**放置位置：** 主文 System Overview。  
**图结构描述：**  
上方一条链：face/landmark → head pose → normalized eye image → gaze estimation；  
下方一条链：face/landmark → eye crop → 3D gaze regression → coordinate transform → ray-plane intersection → affine calibration → smoothing → interaction；  
相同模块用相同颜色，高亮你的新增系统模块。

### **建议图 3：误差随光照条件变化图**

**表达什么信息：**  
借鉴 MPIIGaze Figure 9，证明真实光照变化对系统影响显著，以及你的某个设计是否缓解了这一点。  
**放置位置：** 主文实验部分。  
**图结构描述：**  
横轴为 face brightness 或左右亮度差，纵轴为 angular/pixel error，两条曲线分别代表 baseline 与你的方法，或 calibration 前后。

### **建议图 4：cross-user / cross-session 泛化柱状图**

**表达什么信息：**  
强化“不是随机划分凑出来的结果，而是更接近真实部署条件的评估”。  
**放置位置：** 主文实验部分。  
**图结构描述：**  
每个柱表示一个 held-out user/session/device 的平均误差，附标准差。

## **9. 风险与边界**

### **哪些地方我容易“看懂了但写不出来”**

**(1) 归一化空间的严格定义。**  
MPIIGaze 的 camera normalization 不是一句“把眼睛裁出来”就能替代的，它涉及固定距离、旋转对齐、焦距与分辨率统一。如果你没完整实现，不要在方法里写得太满。

**(2) cross-dataset generalization 的严谨叙述。**  
你可以借它的问题意识，但如果没真正做跨数据集实验，就不要写成“我们验证了跨数据集泛化”。可以说“受相关工作启发，我们进一步考虑跨用户/跨设备鲁棒性”。这更安全。

### **哪些地方容易“看起来高级但不适合我的项目”**

**(1) 把 gaze angle normalization 那套完整搬到当前工程里。**  
你现在的强项是 3D gaze + geometry + calibration + realtime system。如果贸然为了贴论文而大改表示层，风险很高，而且未必更适合桌面交互。

**(2) 把 person-independent calibration-free 当作你当前主打贡献。**  
Look2Act 当前实际上依赖后置 9 点仿射校准来达到更好的桌面点位精度，这和“完全 calibration-free”不是一回事。

### **哪些 claim 我不能乱说**

* 不能说 “MPIIGaze 证明了普通 webcam 已可直接高精度桌面控制”
* 不能说 “我们的方法和 MPIIGaze 同一设定下优于其结果”
* 不能说 “我们实现了 MPIIGaze 风格的标准化 gaze estimation” 除非你真的做了那套 warp/normalization
* 不能说 “我们的系统无需校准”——你 README 明确仍有 optional 9-point affine calibration。

### **哪些方法如果没有额外实验支撑就不要写成我的贡献**

* “head pose 显式融合提升性能”——如果你的网络并未显式输入 pose，就别写成已证实贡献
* “跨域泛化更强”——没有跨数据集或至少跨设备对照不要写
* “对光照鲁棒”——没有 brightness/illumination 分桶分析不要写
* “校准负担更低”——没有 3/5/9 点对照不要写。你 README 已有这项计划实验，建议尽快落地。

## **10. 最终落地建议**

### **“我现在就该做”的 1–3 个动作**

**(1) 立刻把 MPIIGaze 放进你的论文叙事骨架里，但定位为 dataset/benchmark/in-the-wild evidence，不是方法直接来源。**  
这是马上能产生价值的。

**(2) 追加一个与 MPIIGaze 强相关的实验子节：cross-user + 光照分桶误差分析。**  
这比继续微调小网络结构更能提升论文质量。

**(3) 修改 README/论文草稿中关于 MPIIGaze 的相关工作表述，把模糊的单一精度数字改成更谨慎的 setting-aware 描述。**  
这一点很重要，避免后面投稿时被 reviewer 抓口径问题。

### **“以后可以做但现在不要分散精力”的 1–3 个动作**

**(1) 完整复现 MPIIGaze 的归一化 gaze-angle pipeline。**  
研究上有价值，但现在会分散你主线精力。

**(2) 为了贴近这篇论文而重写现有网络输出定义。**  
没必要。

**(3) 真正做跨公开数据集统一 benchmark。**  
这是后续扩展项，不是你当前最短板。当前更短板的是实验设计的扎实程度和叙事规范性。

### **总评分**

**8.5 / 10**

理由：  
它对 Look2Act **当前阶段** 的价值非常高，但主要高在**研究定位、related work、数据动机、实验设计和论文叙事**；对你现有工程实现的“直接可复制方法”价值中等，不是那种一篇看完就能直接改代码拿结果的论文。

\---

# **附加分析：这是一篇数据集 / benchmark / 奠基性代表论文**

## **A1. 这篇论文的数据集对我的 Look2Act 最有价值的地方是什么？**

最有价值的不是“数据量大”本身，而是它给了你一个很强的论据：**真实日常笔记本使用场景中的 gaze estimation，和实验室数据集不是一回事。**  
MPIIGaze 的采集跨多个时间段、场景与光照条件，图 3、图 4、图 5 都在强调这种现实复杂性。它能帮助你为 Look2Act 这种真实桌面部署系统建立正当性：为什么不能只拿受控数据说故事。

第二个价值是：它把 **laptop-based gaze** 这个场景抬到了研究层面，这点和你天然同路。Look2Act 做的是普通 webcam 桌面交互，不是车载、不是手机、不是专用 IR tracker，因此 MPIIGaze 是非常合适的背景工作。

## **A2. 它的数据采集条件、标注方式、场景分布，和我的真实桌面系统之间有哪些 domain gap？**

有，且不小：

**(1) 任务形态 gap**  
MPIIGaze 是提示式 fixation 采集：每 10 分钟弹出随机 20 个屏幕点，用户看点并按空格确认。你的真实系统是连续在线 gaze-driven interaction。前者是监督采集任务，后者是实时交互任务。

**(2) 输出目标 gap**  
MPIIGaze 关心 gaze estimation；Look2Act 关心 screen-point projection、校准后精度和交互可用性。  
也就是说，它是 estimator dataset，不是 full interaction dataset。

**(3) 几何假设 gap**  
MPIIGaze 采集时有每台设备相机内参与 screen plane 几何标定；你的 README 目前使用的是简化相机内参和已知屏幕尺寸/位置假设。这意味着它的标签几何基础更严格，而你的系统更偏部署务实路线。

**(4) 用户规模 gap**  
MPIIGaze 有 15 人、213,659 图像；你当前自采只有 3 个用户、10 个 session、2193 样本。它很适合作为“公开大规模预训练或背景论据”，但不能掩盖你自采规模仍小的事实。

## **A3. 我是否适合把它作为：预训练来源 / baseline 对照背景 / “真实世界困难性”的证据**

**预训练来源：适合，但要谨慎。**  
它适合作为公开数据预训练来源或至少背景动机，因为场景是 laptop in-the-wild，和你最接近的一类公开数据之一。但要注意标签与输入标准化方式可能和你当前管线不完全一致，真正拿来训时需要统一表示。  
结论：**适合做预训练来源的候选，但不是“拿来就训”的零改造来源。**

**baseline 对照背景：非常适合。**  
你完全可以在论文里说：公开 benchmark 已表明 in-the-wild gaze estimation 很困难，而我们的工作进一步走向实际桌面交互部署。

**“真实世界困难性”的证据：非常适合。**  
这是它最强的用途之一。它的数据与实验就是在证明现实环境的外观、光照和域偏移问题。

## **A4. 如果我要在自己的论文里写“公开数据预训练 + 自采数据微调 + 真实部署评估”，这篇论文能提供哪些论据？**

它能提供四类论据：

**(1) 公开数据必要性。**  
真实 gaze 数据采集昂贵且复杂，跨时间、跨光照、跨设备覆盖难；MPIIGaze 通过长期 laptop 采集才得到较大规模数据。你可以据此说明为什么需要公开数据辅助训练。

**(2) 域偏移客观存在。**  
它的 cross-dataset 结果明确说明，训练域与测试域不同会带来明显性能下降。你因此有充分理由采用“公开预训练 + 自采微调”。

**(3) 真实部署仍需自采微调。**  
MPIIGaze 也显示 domain-specific training 会更好。这个结论正好支撑你的微调策略。

**(4) 公开数据与最终部署评估不能互相替代。**  
MPIIGaze 是 estimator benchmark；你做的是完整系统。因而“公开数据预训练 + 自采微调 + 实机部署评估”是合理组合，而不是多此一举。

## **A5. 3 条最适合写进 related work / dataset motivation 的句子**

句子 1：  
“MPIIGaze 首次系统性地将 appearance-based gaze estimation 从受控实验室环境推进到日常 laptop 使用场景，显著揭示了真实环境中光照变化、外观差异与数据域偏移带来的挑战。”

句子 2：  
“与受控条件下的数据集相比，MPIIGaze 在采集时间、环境与外观分布上更贴近真实桌面使用，因此常被视为 in-the-wild webcam gaze estimation 的代表性基准之一。”

句子 3：  
“MPIIGaze 的跨数据集与同数据集评估结果表明，公开数据可为通用 gaze estimator 提供重要训练基础，但面向具体部署场景的 domain-specific adaptation 或自采微调仍然具有必要性。”

