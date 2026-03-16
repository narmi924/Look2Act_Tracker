这篇是 **UnityEyes / “Learning an appearance-based gaze estimator from one million synthesised images” (2016)**。对 Look2Act 来说，它的价值非常明确：**不是直接给你一个最终部署模型，而是给你一套很强的数据策略论据——为什么 synthetic data 必要、如何覆盖真实数据难以覆盖的 gaze/head pose/illumination 空间、以及为什么“合成预训练 \+ 真实数据适配”是合理路线。**

---

## **1\. 一句话判断**

这篇论文对 Look2Act 的最大价值，在于 **数据策略与训练叙事**，不是系统级方法本身。它最适合帮助你的部分是：**synthetic data 的正当性、domain gap 叙事、augmentation / pretraining 合理性说明、以及为什么真实 gaze 数据永远不够全。**

更直白地说，这篇论文不是让你把 Look2Act 改成 Unity 渲染器项目，而是让你在论文中更有底气地说：**公开真实数据 \+ 合成数据 \+ 自采桌面数据** 这三类数据各有角色，单靠其中任何一类都不够。它和你当前 README 最接近的是 **eye crop / gaze regression / 训练数据策略**，最远的是 **ray-plane、screen projection、calibration、interaction layer**。

---

## **2\. 面向 Look2Act 的精确摘要**

这篇论文解决的核心问题是：**appearance-based gaze estimation 需要大量带标签训练数据，但真实采集的数据即使很大，仍然覆盖不足，而且采集昂贵、耗时、难以系统控制 head pose / gaze / illumination 分布。** 为了解决这个问题，作者提出 UnityEyes：基于统计形状模型、真实感渲染、HDR lighting 和可控 gaze/head pose 参数的大规模眼区图像合成框架，可在 commodity GPU 上高速生成大量带精确标签的训练数据。

它为什么和你的 Look2Act 相关：

第一，它正面回答了你后面一定会遇到的数据问题：**自采桌面 gaze 数据规模小、采集成本高、分布容易偏、极端条件覆盖不足**。UnityEyes 的整篇论文，本质上就是在证明“如果只靠真实采集，很难把 gaze/head pose/illumination 空间覆盖完整”。

第二，它与你当前 Look2Act 的主干有天然接口。你目前走的是 **face/landmark → eye crop → gaze regression → geometry projection**，而 UnityEyes 正是围绕 **眼区图像 \+ gaze labels \+ landmarks metadata** 这个训练前半段展开。它不会替代你的后半段几何链，但很适合作为前半段 estimator 的数据来源或动机来源。

第三，它和你当前系统最远的地方，是它不关心最终桌面交互。它做的是 **training data generation framework**，不是完整 screen-point system。它不讨论 PnP、ray-plane、9 点校准、cursor 控制，因此不能被你写成“Look2Act 的整体方法来源”。

---

## **3\. 可迁移到我项目中的核心点（按优先级排序）**

### **3.1 可以直接借鉴的 3–5 个点**

**(1) synthetic data 用于补覆盖，而不是替代真实数据。**  
这篇论文最值得借的不是“合成数据万能”，而是“真实数据再大也不够全面，合成数据可以有针对性地密集采样 head pose、gaze angle、illumination 空间”。这很适合你的 Look2Act 数据策略叙事。

**(2) 明确把 illumination variation 当成关键误差源。**  
论文反复强调 lighting variation 是 appearance-based gaze estimation 的主要误差来源之一，因此他们用 HDR panorama、directional light、曝光变化来覆盖更多照明条件。对你非常有用，因为你的桌面真实系统同样会被光照变化严重影响。

**(3) gaze / head pose 分布可以被“目标化设计”。**  
UnityEyes 允许指定 gaze 分布和 camera/head pose 参数范围，这个思想很适合你。你未来完全可以做“面向桌面场景的 synthetic targeting”，比如重点覆盖 Look2Act 常见的正前方、近距离、小中等 head pose 分布，而不是泛泛渲染。

**(4) synthetic data 可输出精确 metadata。**  
论文强调除图像外，还能输出 gaze direction、shape parameters、lighting info、2D/3D facial landmarks 等 JSON metadata。这一点对你很重要，因为它说明 synthetic 不只是“有图”，而是可以配套中间监督或辅助几何标签。

**(5) 数据量与覆盖度确实影响性能。**  
图 14 直接显示：随着渲染图像数量增大，pixel error 和 gaze error 都下降。这给你一个很有用的论据：数据规模与覆盖并非装饰，而是真实影响 gaze estimator 的泛化性能。

### **3.2 可能需要改造后再借鉴的 2–4 个点**

**(1) UnityEyes 的 eye-region only 路线。**  
它非常适合你的当前 eye-crop 主线，但你已经读过 full-face 论文了，所以更适合借成：

* synthetic eye-region 作为主训练数据  
* 真实人脸 / 真实上下文 作为目标域补偿  
  也就是说，不要把它理解成“只要合成眼睛就够了”。

**(2) 用 k-NN 做 gaze estimation 的实验设定。**  
论文为了证明数据本身价值，甚至用很简单的 k-NN 就在 MPIIGaze 上做到了竞争性结果。这很适合作为“数据质量证明”，但不适合你当前系统直接照搬成主方法。你可以借它说明数据覆盖的重要性，而不是把 k-NN 当最终模型。

**(3) 只做 generic rendering environment。**  
论文特意强调没有做 dataset-specific targeting。对你来说，这反而提示一个更现实方向：你未来如果真的用 synthetic data，**最好做面向桌面 webcam 场景的定向合成**，而不是完全 generic。

### **3.3 不建议我借鉴或当前阶段不适合借鉴的点**

**(1) 把 UnityEyes 写成“真实桌面 gaze 数据集替代品”。**  
不能这么写。它是合成数据框架，不是真实桌面使用数据。

**(2) 直接把它当成你当前系统精度的主证据。**  
它最多能证明数据策略合理，不能直接证明你的 Look2Act 屏幕交互性能。

**(3) 现在就大规模投入精力自建复杂渲染器。**  
对你当前阶段不划算。你更应该先把 synthetic/public/self-collected 三类数据在叙事和实验协议上组织好，而不是自己去重造 UnityEyes。

---

## **4\. 方法与公式层面的提炼**

### **4.1 关键变量定义**

这篇论文不是典型“方法公式很多”的估计器论文，它更偏 **生成模型与渲染参数化**。但仍有几组对你有用的变量：

* (s \\in \\mathbb{R}^{3n})：眼区 3D 形状向量  
* (M\_s \= (\\mu, \\sigma, U))：PCA 形状模型  
* (\\alpha \\in \\mathbb{R}^m)：形状基系数  
* (s(\\alpha) \= \\mu \+ U \\operatorname{diag}(\\sigma)\\alpha^T)：生成新的眼区形状  
* (\\theta\_p, \\theta\_y)：眼球 pitch / yaw 分布中心  
* (\\delta\\theta\_p, \\delta\\theta\_y)：gaze 分布范围  
* (\\phi\_p, \\phi\_y)：camera / head pose 参数  
* (\\delta\\phi\_p, \\delta\\phi\_y)：head pose 范围  
  这些变量最重要的意义在于：**合成数据不是“乱生成”，而是对形状、gaze、pose、lighting 进行可控采样。**

### **4.2 核心建模思路**

这篇论文的核心思路不是估计器 (f\_\\theta(\\cdot)) 本身，而是下面这条链：

1. 构建具有真实感和可变性的眼区生成模型  
2. 对 gaze、head pose、illumination、shape 进行参数化采样  
3. 快速渲染大量带精确标签的训练图像  
4. 用这些数据训练或支持 appearance-based gaze estimator。

对你来说，可转写成一句非常实用的训练叙事：

\[  
\\text{synthetic data} \\rightarrow \\text{cover broad appearance/pose/gaze space}  
\\rightarrow \\text{public real data} \\rightarrow \\text{bridge to real observations}  
\\rightarrow \\text{self-collected data} \\rightarrow \\text{adapt to final desktop deployment}  
\]

这比只说“我们用了公开数据和自采数据”更完整。

### **4.3 适合我项目的数学表达**

#### **A. 可以真实写进我项目方法/实验章节的公式**

如果你未来论文里只想吸收它的“数据策略”，最安全可写的是下面这种非常简洁的训练表述：

\[  
\\mathcal{D}*{train} \= \\mathcal{D}*{syn} \\cup \\mathcal{D}*{pub} \\cup \\mathcal{D}*{self}  
\]

其中：

* (\\mathcal{D}\_{syn})：合成数据，用于覆盖更广 gaze / pose / illumination 空间  
* (\\mathcal{D}\_{pub})：公开真实数据，用于学习真实图像分布下的通用 gaze 表征  
* (\\mathcal{D}\_{self})：自采桌面数据，用于适配最终目标场景

如果你要进一步写训练流程，可以写成：

\[  
\\theta\_0 \= \\arg\\min\_\\theta \\mathcal{L}(\\theta; \\mathcal{D}*{syn} \\cup \\mathcal{D}*{pub})  
\]

\[  
\\theta^\* \= \\arg\\min\_\\theta \\mathcal{L}(\\theta; \\mathcal{D}\_{self}, \\theta\_0)  
\]

这对应“先预训练，再目标域微调”的最安全表达。前提是你真的这样做。  
这类公式不是原文公式，但它是基于原文思想、且适合你项目落地的表达。

#### **B. 只能作为 related work 理解的公式**

UnityEyes 论文里的 PCA 形状模型：

\[  
s(\\alpha)=\\mu \+ U \\operatorname{diag}(\\sigma)\\alpha^T  
\]

这很适合你理解 synthetic eye-region generation 的原理，也适合在 Related Work 里说明“合成数据如何建模 shape variability”，但不适合写成你的方法，除非你真的做了生成建模。

#### **C. 不建议我照搬的公式**

它关于角膜折射、shader、几何程序化 eyelid 动画的公式，不建议你照搬到自己的方法章节。原因很简单：那是渲染器内部细节，不是你的 gaze 系统贡献。

---

## **5\. 和我当前系统架构的映射**

### **face / landmark**

**弱到中等相关。**  
UnityEyes 可以输出 landmarks metadata，这对你前端 face/eye alignment 有启发，但它本身不是 face landmark 方法论文。

### **eye crop / face crop**

**高度相关。**  
它最直接服务于 eye-region appearance modeling，这和你当前 eye crop 主线非常贴近。

### **gaze regression**

**中等相关。**  
它主要是在为 gaze regression 造数据，而不是提出新 gaze network。对你来说是“训练前置支撑”，不是“模型主线替代”。

### **head pose**

**中等相关。**  
论文明确把 camera / head pose 当作合成参数之一，可连续控制。这对你很有价值，因为它说明 head pose 覆盖不足可以通过合成补。

### **geometry / ray-plane / coordinate transform**

**几乎无关。**  
这不是它关心的问题。

### **calibration**

**无关。**

### **smoothing / temporal stability**

**无关。**

### **screen projection / interaction layer**

**无关。**

结论很明确：**它服务于你系统的“数据层和估计器前半段”，几乎不服务于“几何层和交互层”。**

---

## **6\. 对我论文写作最有用的引用建议**

### **最适合引用的位置**

最适合放在：

* **Introduction**：真实 gaze 数据采集难、覆盖不足、成本高  
* **Related Work – Synthetic data / learning-by-synthesis**  
* **Method – Training data strategy / Pretraining motivation**  
* **Experiment – Data ablation / Data source comparison**  
* **Discussion / Limitation**：合成到真实仍有 domain gap

### **推荐我如何描述它（学术中文）**

表述 1：  
“Wood 等指出，尽管真实 gaze 数据集规模不断扩大，但其在头部姿态、注视方向及眼区外观分布上的覆盖仍然有限，而大规模带标签数据的真实采集又具有显著的时间与成本开销。”

表述 2：  
“为缓解这一问题，该工作提出 UnityEyes 合成框架，通过可控的 3D 眼区生成模型与实时渲染机制高效合成大规模训练数据，从而补充真实数据难以覆盖的姿态与光照变化。”

表述 3：  
“该工作进一步表明，合成数据的关键价值并不在于替代真实图像，而在于通过密集覆盖 gaze / head pose / illumination 空间，为 appearance-based gaze estimation 提供更丰富的先验训练样本。”

### **推荐我如何描述它（学术英文）**

Usable sentence 1:  
“Wood et al. argued that even large real gaze datasets remain limited in their coverage of head pose, gaze direction, and eye appearance, while collecting accurately labeled real-world gaze data is expensive and time-consuming.”

Usable sentence 2:  
“They introduced UnityEyes, a real-time synthesis framework for generating large-scale labeled eye-region images with controllable gaze, head pose, and illumination variations.”

Usable sentence 3:  
“The work suggests that synthetic data is particularly useful for densifying the coverage of appearance and geometric variations that are difficult to collect exhaustively in real settings.”

### **1–2 句可直接改写后使用的 related work 句式**

句式 1：  
“Learning-by-synthesis 相关研究表明，真实 gaze 数据即使规模较大，仍难以充分覆盖复杂光照、广泛头姿与极端注视方向，而合成数据能够以较低成本对这些关键变化因素进行密集采样。”

句式 2：  
“因此，合成数据更适合被视为真实数据的补充而非替代，其主要作用在于为 gaze estimator 提供更广的预训练覆盖范围，并为后续在公开真实数据及目标场景数据上的适配奠定基础。”

### **容易被误引的地方**

你最容易误引的有四处：

1. 不要写成“UnityEyes 证明 synthetic data 可以完全替代真实数据”。  
2. 不要写成“只用 synthetic data 就足够支持实际桌面部署”。  
3. 不要把它的 k-NN 结果写成“比所有深度方法都更强”。它的重点是数据质量与覆盖，而不是 k-NN 本身。  
4. 不要把它当成你的系统方法来源，它本质上是数据生成框架。

---

## **7\. 对我实验设计最有用的启发**

### **我可以从这篇论文中借哪些实验设置**

**(1) 数据源消融。**  
这是你最该借的实验思路。你可以做：

* only self-collected  
* public only  
* synthetic \+ public  
* public \+ self-collected  
* synthetic \+ public \+ self-collected

这组实验会直接回答：每类数据到底在 Look2Act 中起什么作用。

**(2) 数据量分析。**  
论文图 14 的思路很值得借：随着训练样本增加，误差是否下降。  
你可以做更现实的版本：

* 10%  
* 30%  
* 50%  
* 100% synthetic/public/self data  
  观察性能趋势。

**(3) shape / appearance variability 的价值分析。**  
论文专门比较了有无 shape variation 的 UnityEyes 数据，发现有变化更好。你可以借成：

* 基础 augmentation  
* 加强 illumination augmentation  
* 加入 synthetic pretraining  
  比较各自收益。

### **我可以新增哪些对照 / 消融**

优先级最高的三个：

1. **公开预训练 vs 公开+合成预训练**  
2. **公开+合成预训练后，是否再用自采微调**  
3. **不同 illumination augmentation 强度下的泛化性能**

### **我可以补哪些图表 / 误差分析**

* data source ablation 柱状图  
* training data size vs error 曲线  
* illumination-condition 分桶误差图  
* synthetic/public/self 三类样本分布示意图  
* failure cases：真实中哪些情况 synthetic 覆盖不到（发丝遮挡、妆容、眼镜强反光等）

### **它是否启发我补 cross-user / cross-device / calibration / latency / stability / usability 中的某一项**

最直接启发的是：

* **cross-user / cross-domain**  
* **illumination robustness**  
* **数据源与泛化能力的关系**

对 calibration、latency、usability 的直接启发不大。

---

## **8\. 图表与可视化建议（文本描述版本）**

### **图 1：Look2Act 的三类数据协同图**

**表达什么信息：**  
说明 synthetic / public / self-collected 三类数据在训练链中的不同角色。  
**放置位置：** 主文 Method 或 Dataset Strategy。  
**图结构描述：**  
左侧三个模块：

* synthetic：广覆盖、精确标签、可控分布  
* public real：真实纹理与真实噪声  
* self-collected：目标桌面场景对齐  
  中间箭头汇入 gaze estimator pretraining / fine-tuning，右侧输出 Look2Act deployment。

### **图 2：训练数据规模与误差关系图**

**表达什么信息：**  
借鉴原文图 14，说明样本数增加与误差下降的关系。  
**放置位置：** 主文 Experiment。  
**图结构描述：**  
横轴为训练样本数量，纵轴为 angular error / pixel error，多条曲线对应不同数据源组合。

### **图 3：domain gap 结构图**

**表达什么信息：**  
说明 synthetic 到 public real，再到 self-collected 的差异层级。  
**放置位置：** 主文 Related Work / Discussion。  
**图结构描述：**  
三层同心圆或阶梯图：  
最内层 synthetic（可控但不真实）  
中层 public real（真实但不完全目标域）  
最外层 self desktop data（最贴目标部署但规模有限）

### **图 4：synthetic failure cases vs real target cases**

**表达什么信息：**  
说明 synthetic 的价值与边界，不把它神化。  
**放置位置：** 附录或 Discussion。  
**图结构描述：**  
左边放 synthetic 可覆盖的变化：gaze/head pose/illumination；右边列 synthetic 难覆盖的真实复杂因素：妆容、发丝遮挡、强反光眼镜、摄像头压缩噪声、个体习惯性姿态。

---

## **9\. 风险与边界**

### **哪些地方我容易“看懂了但写不出来”**

**(1) synthetic data 的角色边界。**  
你很容易理解“合成数据很有用”，但在论文里最容易写过头。正确写法应该是：  
**synthetic data 用来补 coverage、做 pretraining、增强对 pose/gaze/illumination 的覆盖；真实数据仍然负责对齐真实纹理分布与最终目标域。**

**(2) domain gap 的双重性。**  
一方面 synthetic 能补 coverage；另一方面它和真实图像之间有 gap。  
如果你没把这个双重性写清楚，审稿人会觉得你要么神化 synthetic，要么没有真正理解它的局限。

### **哪些地方容易“看起来高级但不适合我的项目”**

**(1) 自己去构建完整 3D morphable eye renderer。**  
研究上很酷，但对你当前主线不是最高 ROI。

**(2) 用复杂渲染细节包装自己论文。**  
像角膜折射、眼睑程序动画这些内容，看起来很专业，但和你当前 Look2Act 的贡献主线无关。不要被带偏。

### **哪些 claim 我不能乱说**

* 不能说 synthetic data 可以替代真实桌面数据  
* 不能说 UnityEyes 与 Look2Act 目标域一致  
* 不能说只靠合成预训练就足够支持最终部署  
* 不能说 augmentation 一定提升系统性能，除非你自己做了 ablation  
* 不能说数据更多就一定更好，应该说“在覆盖关键变化因素时通常更有利”

### **哪些方法如果没有额外实验支撑就不要写成我的贡献**

* “synthetic pretraining 显著提升泛化”  
* “illumination-targeted augmentation 显著提升真实桌面鲁棒性”  
* “三类数据协同训练优于双类数据”  
* “synthetic 有助于 extreme gaze / pose coverage”  
  这些都要靠你自己的实验，不要只靠 UnityEyes 帮你说。

---

## **10\. 最终落地建议**

### **“我现在就该做”的 1–3 个动作**

**(1) 在你的论文结构中，正式加入一个“数据策略”小节。**  
不要再只写模型和实验，要明确写 synthetic / public / self-collected 的角色分工。

**(2) 设计一组最小可行的数据源消融实验。**  
哪怕先做：

* public only  
* public \+ self  
* synthetic/public \+ self  
  也比没有强很多。

**(3) 在 related work 里把 UnityEyes 定位成 synthetic data / learning-by-synthesis 的代表文献。**  
不是方法主线文献，而是数据策略文献。

### **“以后可以做但现在不要分散精力”的 1–3 个动作**

**(1) 自建高真实度 eye-region renderer**  
**(2) 做复杂 domain adaptation 模块**  
**(3) 对 synthetic data 做大规模 targeted generation pipeline**  
这些都值得，但不是你当前最急需的。

### **总评分**

**8.8 / 10**

原因：  
这篇论文对 Look2Act **当前阶段** 的直接工程帮助中等，但对你未来写论文时的 **数据策略、预训练合理性、domain gap 背景、augmentation 正当性** 帮助非常大。  
你现在最缺的不是又一个网络结构，而是一套更像研究工作的训练叙事；这篇论文正好补这一块。

---

# **附加分析：这是一篇 synthetic data / domain gap / augmentation 相关论文**

## **A1. 这篇论文如何解释“真实 gaze 数据不够、覆盖不足、采集贵”的问题？**

这篇论文对这个问题的解释是非常系统的，核心逻辑有三层：

### **第一层：真实数据即使很多，也仍然覆盖不足**

论文明确说，像 TabletGaze 有 10 万张、MPIIGaze 有约 21.4 万张，但即使如此，**它们在 head pose 和 eye-region appearance variability 上仍有限**。也就是说，问题不是简单的“样本数不够”，而是 **关键变化空间覆盖不够密**。

### **第二层：真实采集昂贵、耗时、难控制**

论文直接指出，采集大量 ground-truth gaze training data 是 **time-consuming and costly**。更重要的是，真实采集还受限于实验条件：要叫受试者进实验室、要摆设备、要设置 marker、要受屏幕大小与位置限制，因此 gaze/head pose 分布很难系统覆盖。

### **第三层：极端情况在真实采集中很难完整覆盖**

论文强调，synthetic 的一个关键优势是可以精确控制 gaze direction、head pose、lighting。现实里很难专门大量采集“极端 gaze angle、瞳孔完全遮挡、复杂反光、特殊光照”的样本，但 synthetic 可以密集生成这些边界条件。

所以这篇论文给你的最好总结句就是：  
**真实 gaze 数据的问题不是“完全没有”，而是“贵、慢、难控制，而且在关键变化维度上永远不够全”。**

---

## **A2. 它和我当前“公开数据预训练 \+ 自采数据微调”的策略能否形成理论支持？**

**能，而且能形成很强的理论支持。**

但要分清支持的是哪一层：

### **支持 1：为什么需要预训练**

UnityEyes 说明了 synthetic data 能补 pose / gaze / illumination 覆盖，因此它非常适合作为预训练阶段的一个数据来源或理论依据。  
也就是说，它支持“先让模型看到更广的视觉变化”。

### **支持 2：为什么公开真实数据仍然必要**

UnityEyes 并没有否定真实数据；相反，它是在 synthetic 数据上拿 MPIIGaze 做测试，说明真实 benchmark 仍是评估标准。  
所以它支持“synthetic 不能替代真实公开数据”。

### **支持 3：为什么目标域微调仍然合理**

论文展示了 UnityEyes 在 in-the-wild 图像上仍有失败案例，例如妆容、头发遮挡等未建模变化。这恰恰说明：**即使 synthetic 很强，目标域微调依然必要。**  
这正好支持你的“公开数据预训练 \+ 自采数据微调”。

因此，这篇论文和你的策略是高度契合的。最安全的表达不是“它证明了我的策略最优”，而是：  
**它为这类分阶段数据策略提供了合理的理论与经验背景。**

---

## **A3. 我能否把它写成：数据策略依据 / domain gap 背景 / augmentation / pretraining 合理性说明**

### **1\. 数据策略依据**

**可以，而且非常适合。**  
这是它最强的用途之一。

### **2\. domain gap 背景**

**可以。**  
尤其适合用来说明 synthetic 与 real 之间有 gap，同时 real data 自身也存在覆盖不足的问题。  
也就是说，它不是只讲“gap exists”，还讲“coverage matters”。

### **3\. augmentation / pretraining 合理性说明**

**可以，但要谨慎。**  
更准确的说法应该是：

* synthetic data 提供一种 **structured augmentation / synthetic pretraining** 的路径  
* 它支持“通过人工控制的变化扩展训练分布”  
  但不要把 UnityEyes 简化成普通的图像增强论文。它比普通 augmentation 更接近 **data generation / learning-by-synthesis**。

---

## **A4. 一个适合我项目写法的训练叙事：synthetic / public / self-collected 各自扮演什么角色**

这是最适合你直接吸收进 Look2Act 的版本：

### **训练叙事（推荐写法）**

**1\. Synthetic data：覆盖器 / 先验构建器**  
作用：

* 密集覆盖 gaze direction、head pose、illumination 等关键变化因素  
* 提供大规模、精确标签、低成本样本  
* 用于预训练或数据增强，帮助模型建立基础 gaze appearance prior

**2\. Public real data：现实性桥梁 / 通用表征来源**  
作用：

* 引入真实成像噪声、真实人脸纹理、真实设备与场景变化  
* 缓解 synthetic-to-real gap  
* 用于学习更接近现实世界的通用 gaze representation

**3\. Self-collected data：目标域对齐器 / 最终部署适配器**  
作用：

* 对齐你的 Look2Act 最终使用条件：普通桌面 webcam、特定屏幕几何、真实交互方式、特定用户操作习惯  
* 用于微调与最终评估  
* 不是为了“数据量最大”，而是为了“目标域最贴近”

### **最精炼的一句话**

**synthetic 用来补覆盖，public real 用来补真实性，self-collected 用来补目标域一致性。**

这个说法很适合你后面写 Method 或 Dataset Strategy。

---

## **A5. 如果要写进我论文方法/实验，最安全的写法，不要让我过度声称**

这是最关键的部分。下面给你最安全的写法边界。

### **写进方法时，最安全的写法**

你可以写：

为缓解真实 gaze 数据采集成本高、覆盖有限的问题，训练阶段采用多源数据策略。合成数据用于补充 gaze direction、head pose 与 illumination 的覆盖范围；公开真实数据用于提供更接近真实成像条件的表征先验；自采桌面数据用于适配最终部署场景。

这句话是安全的，因为它描述的是 **设计动机与角色分工**，没有夸张说“显著提升”。

如果你真的做了预训练 / 微调，再写：

模型先在合成数据与公开数据上进行预训练，再在自采桌面数据上进行微调，以兼顾广覆盖训练与目标场景适配。

这也安全，前提是你真的这样做。

### **写进实验时，最安全的写法**

你可以写：

为评估不同数据源对模型性能的影响，进一步比较了 synthetic/public/self-collected 数据单独使用及组合使用时的表现。

这也是安全表述，因为它只是说明实验设计。

### **不要这样写**

* “synthetic data 显著解决了 domain gap”  
* “UnityEyes 证明了 synthetic training 足够支撑真实部署”  
* “我们的方法受益于 synthetic pretraining，因此具有更强泛化性”  
  除非你自己做了严格实验。

### **最建议你的论文口径**

把 UnityEyes 用在这三件事上：

1. **解释为什么数据覆盖问题客观存在**  
2. **解释为什么 synthetic / public / self 三类数据可以形成互补**  
3. **解释为什么预训练与多源数据策略是合理的**

不要把它用来直接替你证明结果。那是你实验自己的任务。

