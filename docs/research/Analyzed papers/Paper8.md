这篇是On-Device Few-Shot Personalization for Real-Time Gaze Estimation。

---

## **1\. 一句话判断**

这篇论文对 Look2Act 当前阶段的最大价值，不在于替代你的 3D 几何主链路，而在于把你现在“9 点仿射校准”这件事，升级为一个更学术的 **few-shot personalization / calibration** 研究问题。它最适合帮助你补强的部分是：**校准建模、少样本个性化实验设计、related work 叙事、以及未来从显式仿射校准走向更智能个性化校准的路线**。

---

## **2\. 面向 Look2Act 的精确摘要**

这篇论文解决的问题是：**如何在不依赖额外硬件、且需要 on-device 实时运行的前提下，用极少量校准点实现个性化 gaze estimation**。论文认为传统个性化方案要么需要较多 calibration points，要么需要在线微调模型参数，难以满足实时、端侧、低负担需求；因此提出了基于 embedding 的 supervised few-shot personalization，以及基于 teacher-student 的 unsupervised personalization。

它和你的系统相关，主要不是因为它和你一样做 3D gaze 或几何链条，而是因为它直接命中了你 README 里已经存在的痛点：你当前通过 **9-point affine calibration** 补偿误差，但也明确写了它存在姿态敏感、距离敏感、时间漂移、单用户绑定等局限，并且未来想探索 **3 点甚至 1 点快速校准、few-shot 个性化、隐式校准**。这和本文的问题意识高度一致。

和你当前 README 最接近的模块是：

* calibration  
* cross-user generalization  
* few-shot calibration  
* CPU/on-device real-time deployment

最远的模块是：

* 你的 3D gaze direction regression  
* PnP head pose  
* ray-plane intersection  
* screen geometry / coordinate transform

因为这篇论文本质上是 **2D on-screen gaze estimation \+ 个性化**，它不是你这条 3D 几何主线的直接方法来源。

---

## **3\. 可迁移到我项目中的核心点（按优先级排序）**

### **可以直接借鉴的 3–5 个点**

**(1) 把 calibration 从“仿射后处理”升级为“few-shot personalization”问题定义。**  
这是你最该借的。你现在的 9 点仿射写法更像工程补偿；这篇论文能帮你把它写成：给定少量用户特定校准样本，对基础 gaze estimator 做个体化适配。

**(2) 强调少量 calibration points 的价值，而不是默认点越多越好。**  
文中核心贡献之一就是用 2–5 个点就比传统需要 13 点以上的方案更有效。你完全可以借这个研究问题来设计 0/3/5/9 点对照。

**(3) 借它的“校准特征缓存”思想。**  
这篇论文把 calibration 阶段和 inference 阶段分开，校准时提取用户 calibration features，后续推理时直接读取缓存，降低端侧开销。这个思想对你的 CPU-only Look2Act 很契合，即便你不复现它的 few-shot 网络，也可以借鉴“校准参数/特征缓存”的系统表达。

**(4) 借它反驳 SVR/传统回归在极少样本下不稳的论点。**  
你 README 已经把 SVR calibration、few-shot calibration 列为相关方向；这篇论文给了你一个很明确的学术支点：传统浅层个性化方法在 \<5 点时泛化差。

**(5) 借它的实验叙事：base model → personalized model → few-shot gains → runtime cost。**  
这很适合你以后论文的实验结构。

### **可能需要改造后再借鉴的 2–4 个点**

**(1) embedding-based few-shot personalization 框架。**  
可以借思想，但不能直接搬。因为它基于 **2D screen gaze regression**，而你是 **3D gaze \+ 几何映射 \+ 2D affine correction**。你若真做，需要重写成“对 raw projected screen point 做个性化修正”或“对 gaze embedding 做 user-conditioned correction”。

**(2) direction classifier 辅助任务。**  
论文为 query–calibration pair 增加了相对方向分类，增强对标注噪声的鲁棒性。这个思路可借，但你不必照搬 4 象限分类本身，更适合改成：raw gaze point 相对 calibration anchor 的象限/方向辅助监督。

**(3) unsupervised personalization。**  
可以写进 discussion / future work，尤其适合你想做“用户正常使用中自动校准”的方向；但当前不适合当你主方法。

**(4) device-specific affine parameters。**  
论文里 SAGE 为不同设备引入 device-specific 参数并作用到输入 landmark 与输出 gaze 上。你可以借来思考 cross-device calibration，但不适合直接塞进你当前毕业设计主线。

### **不建议我借鉴或当前阶段不适合借鉴的点**

**(1) 直接把 Look2Act 改成 2D gaze-on-screen 主模型。**  
不建议。你现在项目的独特性恰恰在于 **3D gaze \+ geometry**，这是你系统定位的一部分。

**(2) 直接宣称 few-shot embedding personalization 一定优于 affine calibration。**  
你现在没有做对应实验，不能这么写。

**(3) 直接复现其完整 supervised/unsupervised personalization 网络作为当前毕设主贡献。**  
太重，且和你现有代码链路不连续。

---

## **4\. 方法与公式层面的提炼**

这一部分我分成三层：  
A. 可以真实写进你项目方法章节  
B. 只能作为 related work 理解  
C. 不建议照搬

### **4.1 关键变量定义**

结合论文与你项目，建议用下面这套符号：

* ( \\mathbf{g}\_{pred} \\in \\mathbb{R}^3 )：模型预测的 3D gaze direction  
* ( \\mathbf{p}*{raw} \= (x*{raw}, y\_{raw}) \\in \\mathbb{R}^2 )：通过几何投影得到的未校准屏幕点  
* ( \\mathbf{p}*{gt} \= (x*{gt}, y\_{gt}) \\in \\mathbb{R}^2 )：真实屏幕注视点  
* ( \\mathcal{C} \= {(\\mathbf{p}*{raw}^{(i)}, \\mathbf{p}*{gt}^{(i)})}\_{i=1}^{K} )：K 个 calibration samples  
* ( f\_\\theta(\\cdot) )：基础 gaze estimator  
* ( \\Phi(\\mathbf{p}\_{raw}; \\mathcal{C}) )：利用少量校准样本进行个性化修正的 calibration mapping  
* ( \\mathbf{p}\_{cal} )：校准后屏幕点

这套写法可以把你现有仿射、未来 few-shot、甚至 polynomial correction 都纳入一个统一框架。

---

### **4.2 核心建模思路**

这篇论文最核心的思想可以被你转写成：

先训练一个 person-independent base estimator，再利用少量用户特定校准样本构建一个 personalization / calibration module，在推理阶段对新用户进行快速适配，而不是在线微调整个主干网络。

这对你特别合适，因为你现在已经有：

* base model：GazeNet 输出 3D gaze  
* geometry：把 gaze 映射到屏幕点  
* calibration：做 affine correction

所以你完全可以把现有方法学化成：

\[  
\\mathbf{g}*{pred} \= f*\\theta(\\mathbf{x})  
\]

\[  
\\mathbf{p}*{raw} \= \\Pi(\\mathbf{g}*{pred}, \\mathbf{R}, \\mathbf{t}, \\text{screen geometry})  
\]

\[  
\\mathbf{p}*{cal} \= \\Phi(\\mathbf{p}*{raw}; \\mathcal{C})  
\]

其中 ( \\Pi ) 表示你现有的几何投影链路，( \\Phi ) 表示校准映射。这个统一写法非常适合你的论文方法章节。

---

### **4.3 适合你项目的数学表达**

#### **A. 可以真实写进你项目方法章节的公式**

**(1) 统一的 calibration mapping 表达**

\[  
\\mathbf{p}*{cal} \= \\Phi(\\mathbf{p}*{raw}; \\mathcal{C})  
\]

解释：  
给定未校准 gaze projection 和少量 calibration samples，学习或拟合一个用户特定映射函数，将原始预测映射到更准确的屏幕坐标。

这句是你最该写的，因为它既能覆盖当前 affine，也能为 future work 留口子。

**(2) 当前 affine calibration 的学术化写法**

你 README 里已有：

\[  
\\mathbf{p}*{cal} \= A*  
*\\begin{bmatrix}*  
*x*{raw}\\  
y\_{raw}\\  
1  
\\end{bmatrix},  
\\quad A \\in \\mathbb{R}^{2\\times 3}  
\]

## **\[**

## **A^\* \= \\arg\\min\_A \\sum\_{i=1}^{K}**

## **\\left|**

## **A**

## **\\begin{bmatrix}**

## **x\_{raw}^{(i)}\\**

## **y\_{raw}^{(i)}\\**

## **1**

## **\\end{bmatrix}**

\\mathbf{p}\_{gt}^{(i)}  
\\right|\_2^2  
\]

这个你已经能真实支撑。

**(3) 更一般化的 polynomial / regression correction 表达**

若你未来做 9-point \+ polynomial：

\[  
\\mathbf{p}*{cal} \= \\Psi(\\phi(\\mathbf{p}*{raw}))  
\]

其中  
\[  
\\phi(\\mathbf{p}\_{raw}) \= \[x,; y,; x^2,; xy,; y^2,; 1\]^T  
\]

然后最小二乘拟合：  
\[  
W^\* \= \\arg\\min\_W \\sum\_{i=1}^{K} |W \\phi(\\mathbf{p}*{raw}^{(i)}) \- \\mathbf{p}*{gt}^{(i)}|\_2^2  
\]

这个适合写成“扩展校准模型”或实验对照，不适合直接宣布为主方法，除非你真做。你 README 里已经规划了 9-point \+ polynomial 对照，所以这不是乱写。

**(4) few-shot personalization 的简化版写法**

如果你想把这篇论文思想变成适合 Look2Act 的“未来方法”表达，可以写成：

\[  
\\mathbf{p}*{cal} \= \\Phi*\\omega(\\mathbf{p}\_{raw}, \\mathcal{C})  
\]

其中 ( \\Phi\_\\omega ) 是一个由少量校准样本条件化的个性化修正器。  
它不同于固定仿射矩阵，因为它可以从用户少量样本中学习非线性误差模式。

这句很适合写到方法扩展或 discussion。

---

#### **B. 只能作为 related work 理解的公式**

论文 supervised few-shot personalization 的完整 loss 是：

* query gaze regression loss  
* 相对方向分类 loss  
* 联合训练

你可以理解其结构是“回归 \+ 辅助方向分类”，但不建议你把原式当成自己方法公式直接写进 main method，除非你真的做了这个网络。

它的本质可抽象为：  
\[  
L \= L\_{gaze} \+ \\lambda L\_{direction}  
\]

这句可用于 related work 或 method inspiration。

---

#### **C. 不建议照搬的公式**

**(1) 原论文完整的 query/calibration embedding 拼接回归公式**  
因为你的输入输出定义不同。

**(2) teacher-student unsupervised personalization 细节公式**  
当前与你的主线差得太远，只能 future work。

---

## **5\. 和我当前系统架构的映射**

### **face / landmark**

**弱相关。**  
论文的 SAGE 使用左右眼图像 \+ eye corner landmarks，并且用眼角点替代粗糙 head pose proxy。对你有启发，但不是这篇的主价值。你当前也有 landmark 和 face mesh。

### **eye crop / face crop**

**中等相关。**  
论文明确从 iTracker 的 face+eyes+face grid 改成更轻量的 eyes \+ eye landmarks，并删除 face input 来减少过拟合。对你“是否只保留双眼裁剪”这个设计是有支持作用的，因为你目前就是双眼裁剪主线。

### **gaze regression**

**中等相关，但输出空间不同。**  
论文是 2D gaze on screen；你是 3D gaze direction regression。可以借其“base estimator \+ personalization”结构，但不能把具体头尾层设计直接套上。

### **head pose**

**几乎无关核心主张。**  
这篇论文没有把 head pose 当成几何主链路。它更多是把 eye landmarks 当成更敏感的 pose proxy。对你来说，这不能替代 PnP。

### **geometry / ray-plane / coordinate transform**

**几乎无关。**  
这是你系统区别于它的关键。论文不走你这条显式 3D 几何映射链。

### **calibration**

**强相关。**  
这是本文与你最强的交点。你当前是 9-point affine；本文是 few-shot supervised/unsupervised personalization。它能把你的 calibration 从工程补偿提升成研究问题。

### **smoothing / temporal stability**

**无关。**

### **screen projection / interaction layer**

**弱到中等相关。**  
论文是 on-device real-time 2D screen gaze estimation，能支撑你 interaction layer 的部署叙事，但不能直接提供你的 projection 方法。

---

## **6\. 对我论文写作最有用的引用建议**

### **最适合引用的位置**

最适合放在：

* **Related Work**：personalization / few-shot calibration / on-device gaze estimation  
* **Introduction**：说明为什么“低负担个性化校准”是重要问题  
* **Discussion / Future Work**：从仿射校准走向 few-shot personalization / unsupervised calibration  
* **Experiment motivation**：为什么要比较 0/3/5/9 点

如果你当前论文主方法还是 3D \+ geometry \+ affine，就**不适合**把它引用成你方法直接来源，而更适合当“校准与个性化方向的重要相关工作”。

---

### **推荐如何描述它（学术中文）**

可写为：

He 等人提出了面向端侧实时 gaze estimation 的 few-shot personalization 框架，表明在仅使用少量 calibration points 的情况下，基于嵌入的个性化适配可以显著优于传统依赖较多校准点的个性化回归方法。该工作说明了 gaze 系统中的个体差异补偿并不必然依赖大量显式校准样本，而可以被建模为少样本个性化问题。

或者：

与将校准仅视为二维后处理映射不同，相关研究已开始将其表述为 few-shot personalization，即在保留基础 gaze estimator 的前提下，利用少量用户特定样本实现快速个体化适配。

---

### **推荐如何描述它（学术英文）**

He et al. formulated gaze calibration as an on-device few-shot personalization problem, showing that user-specific adaptation can be achieved with only a few calibration samples while maintaining real-time inference efficiency.

Their results suggest that personalization in gaze estimation should not be treated merely as a post-hoc coordinate correction step, but can be modeled as a lightweight user-conditioned adaptation module.

---

### **可直接改写后使用的 related work 句式**

**句式 1（中文）**  
“已有研究表明，视线估计中的个体差异可通过少样本个性化进一步缓解；与依赖较多校准点的传统回归式个性化方法相比，few-shot personalization 在低校准负担场景下更具潜力。”

**句式 2（中文）**  
“该方向提示我们，校准不仅可以被视为二维映射误差修正，也可以被建模为针对用户特征分布的快速适配过程。”

---

### **容易误引的地方**

你**不能**把这篇论文写成：

* 支持你的 3D gaze \+ PnP \+ ray-plane 方法  
* 证明 affine 一定不如 few-shot  
* 证明 webcam desktop 场景下 3 点一定够用

因为它的实验主体是手机/平板上的 **2D gaze estimation**，场景、设备、输出空间都和你不同。

---

## **7\. 对我实验设计最有用的启发**

### **我可以从这篇论文中借哪些实验设置**

**(1) 做 calibration point 数量曲线**

* 0 点  
* 3 点  
* 5 点  
* 9 点

这几乎是你最该立刻做的实验。你 README 里本来就有 0/3/5/9 的规划。

**(2) 比较不同校准模型**

* no calibration  
* affine  
* polynomial  
* few-shot regressor（若以后做）

**(3) 区分 base model 与 personalized model**  
这篇论文的好处在于它把“基础模型能力”和“个性化增益”拆开报告。你以后也该这样写，而不是把校准后结果和模型本体混在一起。

### **我可以新增哪些对照 / 消融**

**优先级最高：**

1. No calibration vs 3-point affine vs 5-point affine vs 9-point affine  
2. 9-point affine vs 9-point polynomial  
3. 同一点数下，不同头姿/距离偏移下的退化程度  
4. cross-user：训练用户与新用户是否需要校准  
5. cross-session：同一用户隔天复用校准参数是否退化

### **我可以补哪些图表 / 误差分析**

1. **校准点数–误差曲线图**  
   x 轴为 calibration points 数量，y 轴为 pixel/cm error  
2. **校准前后误差分布对比图**  
   最好按用户或 session 分组  
3. **校准迁移稳定性图**  
   校准时姿态 vs 使用时姿态偏移对误差影响  
4. **不同用户收益箱线图**  
   看 few-shot / affine 的个体差异

### **它是否启发我补某一项**

非常明显启发你补：

* **calibration**  
* **cross-user**  
* **cross-session**  
* **latency（若未来做 learned calibrator）**

比起 usability，这篇更像 calibration-personalization 论文；比起 geometry，这篇对你帮助没那么大。

---

## **8\. 图表与可视化建议（文本描述版本）**

### **图 1：Look2Act 中“基础预测 \+ 校准个性化”统一框架图**

**表达信息：**  
展示你的系统并不是“模型输出后随便做个仿射”，而是“base estimator \+ geometry projection \+ calibration adapter”的三段式结构。  
**结构描述：**  
Camera frame → face/eye processing → 3D gaze regression → geometry projection 得到 raw screen point → calibration module（affine / polynomial / future few-shot adapter）→ calibrated screen point。  
**位置：** 主文。

### **图 2：校准点数与误差关系曲线**

**表达信息：**  
展示 0/3/5/9 点下误差随点数下降，但收益可能递减。  
**结构描述：**  
横轴 calibration points，纵轴 mean pixel error / mean cm error；多条曲线对应 affine、polynomial、future few-shot。  
**位置：** 主文。

### **图 3：不同校准范式概念对比图**

**表达信息：**  
将“显式映射校准”和“few-shot personalization”区别开。  
**结构描述：**  
左侧：raw point → affine matrix → corrected point；右侧：query feature \+ calibration samples → user-conditioned correction → corrected point。  
**位置：** 附录或 discussion 图。

### **图 4：cross-session 校准漂移图**

**表达信息：**  
展示同一用户校准后隔时间、换姿态、换距离的误差变化。  
**位置：** 很适合附录，也可主文实验图。

---

## **9\. 风险与边界**

### **哪些地方你容易“看懂了但写不出来”**

**(1) embedding-based few-shot personalization 的网络细节**  
因为你当前代码主线并不是这种结构，硬写会导致答辩时讲不清“query feature、calibration feature、direction classifier”到底怎么用。

**(2) unsupervised personalization**  
很容易觉得高级，但你如果没做 teacher-student 或隐式用户嵌入，写进去会失真。

### **哪些地方容易“看起来高级但不适合我的项目”**

**(1) 直接把 calibration 说成 meta-learning / few-shot learning 方法**  
除非你真的做了 learned adapter，否则当前最稳妥的说法仍然是：

* 当前实现：affine calibration  
* 研究延展：few-shot personalization

**(2) 把本文的 2D 手机端结论直接迁移到你的 3D desktop webcam 系统**  
这会被抓住。

### **哪些 claim 不能乱说**

你不能写：

* “本文证明了 3 点校准足够用于所有 gaze 系统”  
* “few-shot personalization 必然优于 affine transformation”  
* “本工作借鉴该文实现了 on-device few-shot personalization”  
  除非你真的实现了。

### **哪些方法如果没有额外实验支撑就不要写成我的贡献**

* learned few-shot calibrator  
* unsupervised personalization  
* calibration-free adaptation  
* user embedding based correction

这些都只能写成：

* future work  
* inspiration  
* possible extension

---

## **10\. 最终落地建议**

### **“我现在就该做”的 1–3 个动作**

**(1) 立刻把你当前 calibration 章节改写成统一表述。**  
不要只写“9 点仿射校准”，改成：  
“Given K calibration samples, a user-specific mapping ( \\Phi(\\mathbf{p}\_{raw};\\mathcal{C}) ) is estimated to correct raw screen projections. In the current implementation, ( \\Phi ) is instantiated as a 2D affine transform fitted by least squares.”  
这一步收益极高。

**(2) 立刻做 0/3/5/9 点校准对照实验。**  
这是最像论文的动作，也和你的 README 完全一致。

**(3) 在 related work 中新增一个“小节：calibration and personalization”。**  
把这篇作为核心引用之一，说明你知道校准不仅是几何或仿射，也可以是少样本个性化问题。

### **“以后可以做但现在不要分散精力”的 1–3 个动作**

**(1) learned few-shot correction module**  
等你把当前 baseline、affine、polynomial 实验打稳后再说。

**(2) unsupervised / implicit calibration**  
适合 future work，不适合当前主线。

**(3) device-conditioned calibration**  
以后做 cross-device 才值得展开。

### **总评分（满分 10 分）**

**8.3 / 10**

原因很明确：  
它对你的 **3D 几何主链条** 帮助不大，所以不是满分；  
但对你的 **calibration 学术化、个性化叙事、实验设计、future work 路线** 非常有价值，尤其适合把你当前“9 点仿射校准”从工程技巧升级成研究问题。

---

# **附加分析 1：这篇论文中的 calibration 思想更偏什么？**

更偏：

1. **few-shot personalization**  
2. **显式校准**  
3. **映射误差修正**（但不是简单仿射，而是 learned correction）

不太偏：

* 系统级 usability 优化（虽然它关心少点数降低负担，但不是 HCI 论文主轴）  
* 纯隐式校准（因为 supervised 主方法仍要 label）

unsupervised personalization 属于它的扩展，但不是主线。

---

# **附加分析 2：它和我当前 README 中的 9 点校准 / affine transformation 有哪些关系？**

关系非常直接：

你的当前做法是：  
\[  
\\mathbf{p}*{cal} \= A\[x*{raw}, y\_{raw}, 1\]^T  
\]  
本质是 **基于少量标注样本拟合用户特定映射**。

这篇论文做的事情也是：

* 收集少量 calibration points  
* 利用它们构造用户特定适配  
* 在推理阶段用这些“个性化信息”修正预测

差别在于：

* 你是 **显式二维线性映射**  
* 它是 **embedding-conditioned learned personalization**

所以你可以把二者写成同一大类问题的两种实现：

* 传统映射式 calibration  
* 学习式 few-shot personalization

这是非常适合论文叙事的。

---

# **附加分析 3：我是否可以从这篇论文中提炼出一个更学术化的 calibration 数学表达？**

可以，而且很适合你。

建议你以后统一写成：

\[  
\\mathbf{p}*{cal} \= \\Phi(\\mathbf{p}*{raw}; \\mathcal{C})  
\]

其中 (\\mathcal{C}) 为少量用户校准样本。  
然后再说明：

* 当前实现：  
  \[  
  \\Phi(\\mathbf{p}*{raw}; \\mathcal{C}) \= A*  
  *\\begin{bmatrix}*  
  *x*{raw}\\y\_{raw}\\1  
  \\end{bmatrix}  
  \]  
* 扩展实现：  
  \[  
  \\Phi(\\mathbf{p}*{raw}; \\mathcal{C}) \= W \\phi(\\mathbf{p}*{raw})  
  \]  
  或  
  \[  
  \\Phi(\\mathbf{p}*{raw}; \\mathcal{C}) \= \\Phi*\\omega(\\mathbf{p}\_{raw}, \\mathcal{C})  
  \]

这样你就把：

* affine  
* polynomial  
* learned few-shot calibrator  
  统一进了一个学术框架里。

---

# **附加分析 4：适合我项目的校准建模写法**

### **4.1 从原始 gaze estimate 到 screen point 的映射**

你完整链条可以写成：

\[  
\\mathbf{g}*{pred} \= f*\\theta(\\mathbf{x})  
\]

\[  
\\mathbf{p}*{raw} \= \\Pi(\\mathbf{g}*{pred}, \\mathbf{R}, \\mathbf{t}, \\mathcal{S})  
\]

\[  
\\mathbf{p}*{cal} \= \\Phi(\\mathbf{p}*{raw}; \\mathcal{C})  
\]

其中：

* (f\_\\theta)：3D gaze estimator  
* (\\Pi)：几何投影函数  
* (\\mathcal{S})：屏幕几何参数  
* (\\Phi)：校准映射

### **4.2 affine / polynomial / regression correction 的表述**

**Affine**  
\[  
\\mathbf{p}\_{cal} \= A\[x,y,1\]^T  
\]

**Polynomial**  
\[  
\\mathbf{p}\_{cal} \= W\[x, y, x^2, xy, y^2, 1\]^T  
\]

**Regression correction / learned calibrator**  
\[  
\\mathbf{p}*{cal} \= \\Phi*\\omega(\\mathbf{p}\_{raw}, \\mathcal{C})  
\]

### **4.3 calibration samples 的损失函数形式**

如果是显式拟合：  
\[  
L\_{calib}(A)=\\sum\_{i=1}^{K}|A\\tilde{\\mathbf{p}}*{raw}^{(i)}-\\mathbf{p}*{gt}^{(i)}|\_2^2  
\]

如果是多项式：  
\[  
L\_{calib}(W)=\\sum\_{i=1}^{K}|W\\phi(\\mathbf{p}*{raw}^{(i)})-\\mathbf{p}*{gt}^{(i)}|\_2^2  
\]

如果是 learned few-shot：  
\[  
L\_{calib}(\\omega)=\\sum\_{i}|\\Phi\_\\omega(\\mathbf{p}*{raw}^{(i)},\\mathcal{C})-\\mathbf{p}*{gt}^{(i)}|\_2^2  
\]

这三层表述已经足够你方法章节、开题答辩、实验设计一起用了。

---

# **附加分析 5：哪些内容适合写进我的方法章节，哪些只适合写进 discussion / future work？**

### **适合写进方法章节**

* 统一 calibration framework：(\\mathbf{p}*{cal} \= \\Phi(\\mathbf{p}*{raw};\\mathcal{C}))  
* 当前 affine 的最小二乘拟合  
* 如果你真做了，再写 polynomial 对照  
* calibration 点数设计与误差评估指标

### **只适合写进 discussion / future work**

* embedding-based few-shot personalization  
* direction-classifier 辅助监督  
* unsupervised personalization  
* teacher-student adaptation  
* automatic / implicit calibration

---

# **附加分析 6：如果我做 3×3 vs 5×5 calibration，或 few-shot calibration，对照实验该怎么设计更像论文？**

这里我直接给你一个更论文味的实验矩阵。

### **实验 A：校准点数量对性能的影响**

* 0-point  
* 3-point  
* 5-point  
* 9-point（3×3）  
* 25-point（5×5，可选，若你想证明密集校准收益递减）

报告：

* Mean pixel error  
* Mean cm error  
* Median error  
* Calibration time  
* 用户负担（至少记录总点数与总耗时）

### **实验 B：校准模型对比**

在相同点数下比较：

* Affine  
* Polynomial  
* Future learned regressor（如果以后做）

### **实验 C：校准鲁棒性**

固定 9 点校准，测试：

* 同 session  
* 跨 session  
* 姿态偏移  
* 距离偏移

### **实验 D：cross-user \+ few-shot**

* Person-independent base model  
* base \+ 3-point calibration  
* base \+ 5-point calibration  
* base \+ 9-point calibration

如果以后你真做 learned few-shot，可再加：

* base \+ few-shot adapter (3 / 5 points)

### **论文写法上最像样的结论目标**

你最终不是要说“点越多越好”，而是要回答：

1. 少量点是否已经足够显著降低误差  
2. 更多点的收益是否递减  
3. 简单 affine 是否已能提供足够高的性价比  
4. 是否存在跨 session / 姿态漂移导致校准失效

这就很像论文，而不是软件测试。

