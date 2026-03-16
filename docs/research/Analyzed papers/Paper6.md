这篇是CalibMe: Fast and Unsupervised Eye Tracker Calibration for Gaze-Based Pervasive Human-Computer Interaction，**CalibMe 对 Look2Act 的价值很高，但价值不在于“你直接照搬它的采集方式”，而在于它能帮你把当前 README 里的 9 点 affine calibration 从“工程补丁”提升成“有明确建模对象、有质量控制、有 usability 叙事的校准模块”。**  
它和你最相关的不是 gaze regression 本体，而是 **校准样本采集、校准质量评估、校准覆盖率、重新校准触发、以及 3×3 / 5×5 / dense calibration 的实验设计**。你完全可以把它作为 Look2Act 校准章节和实验章节的关键参考文献。

## **1\. 一句话判断**

这篇论文对 Look2Act 的最大价值是：**它把 calibration 这件事从“收几个点做线性拟合”提升成了一个完整的 HCI 子问题：如何更快、更独立、更稳、更可评估地完成校准。**  
它最适合帮助你的部分是：**9 点校准的学术化表达、校准样本质量控制、校准覆盖率定义、以及 few-shot / fast calibration 对照实验设计。**  
它对你的 gaze CNN、3D gaze 向量、ray-plane 几何几乎没有直接方法贡献，但对你论文里的 **Method / Experiment / Discussion** 很有帮助。

## **2\. 面向 Look2Act 的精确摘要**

CalibMe 解决的问题不是“如何估计 gaze”，而是**如何让 regression-based gaze estimation 的 calibration 更快、更少依赖人工、更适合真实 HCI 使用**。论文指出，很多回归式 gaze 方法本身并不复杂，但校准流程长期缺乏改进：通常需要操作者协助、点数少、时间长、还需要人工检查点采得是否可靠。作者因此提出一种基于可自动检测 fiducial marker 的快速无监督采样方法，允许用户在移动 marker 或头部时连续收集大量眼-注视关系点，再做异常值剔除和自动保留评估点，从而把 calibration 同时变成“拟合 \+ 自评估”流程。

它和你的系统相关，主要相关在三层：

第一层，**你的 README 已经明确把 calibration 作为从 raw screen point 到 calibrated screen point 的后处理映射**，并且用 9 点仿射和最小二乘求解 2×3 矩阵 A。CalibMe 和你完全同属“回归映射式校准”范式，而不是几何免校准范式。

第二层，CalibMe 明确把 calibration 看作一个映射函数学习问题：由眼部观测变量到 gaze / point-of-regard 的函数拟合；你现在的 affine 模块本质上就是这个思想的简化版，只不过你的输入已经不是 pupil center，而是几何链路输出后的 raw screen point。也就是说，你们是在**不同层级做同类误差校正**。

第三层，它和你最远的地方在于：CalibMe 是**头戴式 eye tracker \+ field camera \+ marker-based collection**，你的 Look2Act 是**普通 webcam \+ 3D gaze \+ screen plane geometry \+ 屏幕像素域后校准**。所以你不适合把它写成你的系统蓝本，但很适合把它写成**校准方法和 usability 动机**的参考。

## **3\. 可迁移到我项目中的核心点（按优先级排序）**

### **可以直接借鉴的 5 个点**

1. **把 calibration 明确表述为一个映射函数学习问题。**  
   CalibMe 明说 calibration 的本质是“produce a function mapping the position of the user’s eyes to gaze”。你可以安全转写为：在 Look2Act 中，校准学习的是从未校准注视估计 (p^{raw}) 到屏幕真实点 (p^{gt}) 的校正映射。  
2. **把 9 点 calibration 视为一个合理 baseline，而不是默认标准答案。**  
   论文把 9-point 视为“时间与精度的折中基线”，而不是最终最优。这对你很重要，因为你 README 里现在默认 9 点是当前方案，但未来完全可以让 3 点、5 点、9 点、9 点+二次项成为正式对照。  
3. **引入 calibration coverage 的概念。**  
   他们不只看误差，还看“采样是否覆盖了预期交互区域”。这非常适合你，因为你当前 README 只写 mean residual \> 50px 重新校准，但还没有覆盖率指标。  
4. **引入 calibration sample quality control / outlier removal。**  
   他们提出了基于 pupil size ratio、pupil position range、算法 aware 的异常样本剔除。你虽然不是 pupil-based，但可以迁移成：校准阶段去掉 head pose 剧烈变化、blink、low face confidence、prediction jump 的样本。  
5. **把 calibration usability 写进论文贡献或 discussion。**  
   他们最强的点之一是“用户可以独立、快速完成校准”。你未来如果做桌面系统，这种 usability 叙事很重要。

### **可能需要改造后再借鉴的 3 个点**

1. **无监督连续采样式 calibration。**  
   你可以借思想，但不必照搬 ArUco marker。对桌面系统更合理的是：屏幕上移动 target、用户跟随；或屏幕显示短路径/螺旋轨迹，自动采连续样本。  
2. **曲面 calibration / parallax surface 的讨论。**  
   这在头戴式系统里很重要，因为 eye 与 field camera 有视差，校准面与物体面不重合会引入 parallax。你桌面系统没有完全相同的问题，但“校准时头姿 / 距离与使用时不一致会退化”这一点是强相关的。  
3. **自动预留 evaluation points。**  
   你可以把它改成：校准点用于拟合，额外隐藏点用于 hold-out 评估，从而自动判断是否重新校准。

### **不建议你当前阶段借鉴的点**

1. **ArUco marker \+ 手机外部标记作为主校准交互。**  
   这不适合 Look2Act 的“普通桌面 webcam 软件即用”主线。  
2. **直接复刻其头戴式 field-camera 采样逻辑。**  
   设备和坐标系都不一样。  
3. **把它硬写成 few-shot personalization 的代表。**  
   它更偏 fast explicit calibration / mapping correction / usability optimization，不是现代 ML 意义上的 few-shot personalization。

## **4\. 方法与公式层面的提炼**

### **关键变量定义**

对你最有价值的不是 CalibMe 的设备细节，而是它给出的 calibration tuple 形式。论文定义数据 tuple  
\[  
D \= {t, p\_x, p\_y, p\_w, p\_h, cm\_x, cm\_y}  
\]  
其中 (p\_x,p\_y) 是 pupil center，(cm\_x,cm\_y) 是 collection marker center，校准目标是学习一个从 pupil 到 gaze position 的映射。

在你的 Look2Act 里，最自然的迁移写法是把 tuple 改成：

\[  
D\_i \= {t\_i,\\ \\mathbf{g}\_i,\\ \\mathbf{h}\_i,\\ p\_i^{raw},\\ p\_i^{gt}}  
\]

其中

* (\\mathbf{g}\_i)：当前 gaze estimate 的中间特征，可选  
* (\\mathbf{h}\_i)：head pose，可选  
* (p\_i^{raw} \= (x\_i^{raw}, y\_i^{raw}))：几何链路输出的未校准屏幕点  
* (p\_i^{gt} \= (x\_i^{gt}, y\_i^{gt}))：校准目标点

### **核心建模思路**

CalibMe 本质上是：**收集很多校准对应点，然后拟合低阶回归函数做 gaze mapping。** 文中实验统一使用的是 **bivariate second-order polynomial regression**。

对你项目最合适的学术化表达有三层，按复杂度递增：

#### **A. 你现在就能写进方法章节的公式**

**1）仿射校准映射**

你 README 现在的写法是：  
\[  
\\mathbf{p}^{cal} \= A  
\\begin{bmatrix}  
x^{raw}\\  
y^{raw}\\  
1  
\\end{bmatrix}, \\quad A\\in \\mathbb{R}^{2\\times 3}  
\]  
这本身是对的，而且和你的现有系统完全一致。

更学术一点，可以写成：  
\[  
\\hat{\\mathbf{p}}*i \= f*\\theta(\\mathbf{p}^{raw}*i),\\qquad*  
*f*\\theta(\\mathbf{p}) \= A\\tilde{\\mathbf{p}},\\quad \\tilde{\\mathbf{p}}=\[x,y,1\]^T  
\]  
通过最小化  
\[  
\\mathcal{L}*{calib}(\\theta)=\\frac{1}{N}\\sum*{i=1}^N \\left| f\_\\theta(\\mathbf{p}^{raw}\_i)-\\mathbf{p}^{gt}\_i \\right|\_2^2  
\]  
求得 (\\theta=A)。

这就是你当前 9 点 affine 的标准论文写法。

**2）更一般的校准映射**

如果你要为 future work 铺路，可写：  
\[  
\\hat{\\mathbf{p}}*i \= f*\\theta(\\phi\_i)  
\]  
其中 (\\phi\_i) 可以是

* 仅 raw screen point：(\\phi\_i=\[x\_i^{raw}, y\_i^{raw}\])  
* 或拼接 head pose：(\\phi\_i=\[x\_i^{raw}, y\_i^{raw}, yaw\_i, pitch\_i, roll\_i\])

这样你就把 calibration 从“固定 affine”提升成“可替换的 correction function family”。

**3）样本筛选后的加权损失**

结合 CalibMe 的异常值处理思想，可写成：  
\[  
\\mathcal{L}*{calib}(\\theta)=\\frac{1}{\\sum\_i w\_i}\\sum*{i=1}^N w\_i \\left| f\_\\theta(\\mathbf{p}^{raw}\_i)-\\mathbf{p}^{gt}\_i \\right|\_2^2  
\]  
其中 (w\_i\\in{0,1}) 或 (w\_i\\in\[0,1\]) 反映样本质量。  
这非常适合你写成“校准样本质量控制”版方法。

#### **B. 适合写进方法或 future work 之间的公式**

**二次多项式校准**  
\[  
\\hat{x}=a\_1x+a\_2y+a\_3xy+a\_4x^2+a\_5y^2+a\_6  
\]  
\[  
\\hat{y}=b\_1x+b\_2y+b\_3xy+b\_4x^2+b\_5y^2+b\_6  
\]

这和 CalibMe 用的低阶 polynomial regression 路线一致。它非常适合你做 9-point affine vs 9-point polynomial 的对照。

但要注意：**9 个点去拟合 12 个参数会有点紧**，尤其 hold-out 评估时容易不稳。所以更安全的实验写法是：

* 9 点只做 affine baseline  
* 25 点或 dense path 才做 polynomial

#### **C. 只适合理解 related work 的部分**

CalibMe 里关于 parallax、pattern surface vs poster surface 的分析，在头戴式系统中非常重要；但你不要直接抄成你方法里的理论主轴。你的桌面系统没有同样的 eye-camera / field-camera parallax 结构。

## **5\. 和我当前系统架构的映射**

### **face / landmark**

几乎无关。CalibMe 不讨论你这种 webcam 人脸关键点链路。

### **eye crop / face crop**

弱相关。它的校准建立在 eye observation 到 gaze mapping，但不是基于你这种双眼 crop CNN。

### **gaze regression**

中等相关。论文明确说 regression-based gaze estimation 需要 calibration，且现有研究更多关注回归函数而非 calibration 过程本身。这个对你 Introduction 很有用。

### **head pose**

间接相关。CalibMe 讨论“允许头部旋转 / 深度变化”会形成非平面 calibration surface，并影响评估公平性。对你来说，这可转化为：**校准时头姿与使用时不一致会导致参数退化**。

### **geometry / ray-plane / coordinate transform**

弱相关。CalibMe 的核心不在 3D 几何，而在 calibration mapping。

### **calibration**

强相关。是这篇论文与你系统的最强对齐处。你 README 里第 6 节已经明确有 9-Point Affine Calibration、最小二乘求解、mean residual 和重新校准阈值，这与本文非常容易形成对话关系。

### **smoothing / temporal stability**

中等相关。它没有做 EMA，但它的 outlier removal、连续采样、marker movement speed 约束，本质上都在处理 temporal sample stability。

### **screen projection / interaction layer**

中等相关。它的贡献不在 projection，而在“让 gaze-based HCI 更可用”。这能支撑你桌面交互系统里的 usability 讨论。

## **6\. 对我论文写作最有用的引用建议**

### **最适合被引用的位置**

最适合放在：

* **Introduction**：为什么 calibration 仍然是 gaze HCI 的瓶颈  
* **Related Work**：显式校准、快速校准、无监督校准  
* **Method（Calibration 模块）**：说明你采用显式屏幕点校准的合理性  
* **Experiment**：3×3 / 5×5 / dense calibration 对照的动机  
* **Discussion / Future Work**：自动重校准、隐式个性化、在线 drift correction

### **推荐学术中文描述**

“CalibMe 将校准视为 gaze-based HCI 可用性的核心瓶颈之一，指出传统 N 点校准往往依赖操作者协助、采样点少且耗时。为此，该工作提出基于自动检测标记的快速无监督校准流程，通过连续采样、异常值剔除与自动保留评估点，提高了校准效率与可评估性。”

### **推荐学术英文描述**

“CalibMe treats calibration as a critical bottleneck in gaze-based HCI rather than a minor preprocessing step. It proposes a fast and unsupervised calibration procedure based on automatically detected collection markers, together with outlier removal and automatic reservation of evaluation samples.”

### **可直接改写的 related work 句式**

1. “已有研究指出，回归式 gaze estimation 的性能不仅取决于回归模型本身，还显著受限于 calibration 过程的效率、覆盖范围与样本质量；CalibMe 即从这一角度出发，对传统 N 点校准流程进行了系统改造。”  
2. “与仅比较回归误差的工作不同，CalibMe 同时强调 calibration accuracy、calibration coverage 与 calibration time，为本文的校准实验设计提供了直接参考。”

### **容易误引的地方**

你不要把它误写成：

* 现代 few-shot neural personalization 论文  
* 桌面 webcam calibration 论文  
* 你的 affine 校准的直接前身  
* 纯粹的 drift correction 论文

它更准确的定位是：**fast explicit calibration \+ mapping correction \+ usability-oriented calibration design**。

## **7\. 对我实验设计最有用的启发**

这篇论文对你实验最值钱的启发，基本都集中在 calibration 部分。

### **你可以直接借的实验设置**

1. **把 calibration time 纳入正式指标。**  
   不是只报精度。CalibMe 同时看误差和时间。你也应该这样做，否则 25 点可能误差更低但体验差。  
2. **把 calibration coverage 纳入正式指标。**  
   对桌面系统可定义为：校准点的凸包或网格覆盖面积占交互区域的比例，或 hold-out 测试点被包含的比例。  
3. **保留独立 evaluation points。**  
   不要只报 calibration residual，因为那会高估效果。CalibMe 自动预留 evaluation tuples 的做法就是为了解决这个问题。  
4. **显式比较 9-point baseline。**  
   CalibMe 直接拿 9 点做基线，这是你最容易对齐的地方。

### **你可以新增的对照 / 消融**

1. **3×3 vs 5×5 vs dense continuous path**  
   * 3×3 affine  
   * 5×5 affine  
   * 5×5 polynomial  
   * 连续路径采样 \+ affine  
   * 连续路径采样 \+ polynomial  
2. **sample filtering**  
   * 无筛选  
   * 置信度筛选  
   * head-pose-stable 筛选  
   * blink / large-jump 筛选  
3. **calibration feature family**  
   * raw point only  
   * raw point \+ head pose  
   * raw point \+ distance estimate  
4. **few-shot calibration**  
   * 1 点平移修正  
   * 3 点仿射近似  
   * 5 点仿射  
   * 9 点仿射

### **可以补的图表 / 误差分析**

* 校准点数 vs 平均误差曲线  
* 校准点数 vs 校准时间曲线  
* 校准点数 vs 稳定性标准差曲线  
* 屏幕空间热力图：不同区域误差  
* 校准后随时间漂移曲线：0 min / 10 min / 20 min

### **它最启发你补哪一类实验**

优先级是：  
**calibration / stability / usability** \> cross-user personalization \> latency

## **8\. 图表与可视化建议（文本描述版本）**

1. **图 1：Look2Act 校准建模图**  
   左侧是原始几何链路输出 (p^{raw})，右侧是真实屏幕点 (p^{gt})，中间是 calibration function (f\_\\theta)。  
   表达的信息：校准不是替代 gaze model，而是对几何输出残差做用户相关修正。  
   放主文。  
2. **图 2：3×3 / 5×5 / dense calibration 对比图**  
   三列分别显示采样点布局、平均误差、所需时间。  
   表达的信息：点数更多不一定最优，存在 accuracy-usability trade-off。  
   放主文。  
3. **图 3：屏幕误差热图（校准前后）**  
   上排 raw， 下排 calibrated。  
   表达的信息：校准如何纠正系统性偏差，尤其边角区域误差。  
   放主文。  
4. **图 4：校准覆盖率示意图**  
   显示校准样本分布、hold-out 点位置、未覆盖区域。  
   表达的信息：覆盖不足会带来局部外推失真。  
   放附录或主文补图都可以。  
5. **图 5：时间漂移与重校准触发图**  
   横轴时间，纵轴 hold-out error；阈值线表示触发 re-calibration。  
   表达的信息：校准不是一次性动作，而是会退化的系统状态。  
   放 discussion 或附录。

## **9\. 风险与边界**

1. **最容易“看懂了但写不出来”的地方**  
   是它关于 pattern surface / poster surface / parallax 的那套分析。你可以理解为“校准分布和评估分布不一致会偏”，但不要把头戴式 parallax 原理硬搬到你桌面 webcam 系统里。  
2. **最容易“看起来高级但不适合你的项目”的地方**  
   是外部 marker、手机辅助、无监督大规模连续采样。Look2Act 当前主线是普通桌面 webcam，主文里不宜突然变成“还要拿手机做辅助校准”。  
3. **不能乱说的 claim**  
   * 不能说 CalibMe 证明 affine 一定优于 polynomial  
   * 不能说它是 few-shot personalization SOTA  
   * 不能说它直接适用于桌面 webcam  
   * 不能说它验证了你的 3D 几何链路  
4. **没有额外实验就不要写成你的贡献**  
   * “更快校准”  
   * “更少点数也能达到同样精度”  
   * “支持在线重校准”  
   * “鲁棒于姿态变化的个性化校准”  
     这些都要靠你自己的对照实验。  
5. **顺手指出你 README 一个可加强点**  
   你现在的 calibration 数学已经够清楚，但仍偏“工程实现说明”。建议你在 README / 论文里把 calibration 的对象明确写成：  
   **“校准学习的是从系统性几何残差到真实屏幕点的用户特定修正映射。”**  
   这样会比“通过 affine 补偿系统误差”更学术，也更像方法模块。

## **10\. 最终落地建议**

### **我现在就该做的 3 个动作**

1. **把你当前 9 点 affine 公式改写成统一的 calibration mapping 形式**：  
   ( \\hat{p}=f\_\\theta(p^{raw}) )，其中 affine 是一个具体实例。  
2. **给校准模块补两个指标**：hold-out error 和 coverage。  
3. **立刻设计 3×3 vs 5×5 vs 9-point vs polynomial 的正式实验表。**

### **以后可以做但现在不要分散精力的 3 个动作**

1. 做屏幕上连续轨迹的快速校准交互。  
2. 做基于 head pose / usage data 的隐式重校准。  
3. 做真正意义上的 few-shot per-user personalization，把少量校准样本用于微调校正网络。

### **总评分**

**8.7 / 10**

原因：  
对 Look2Act 当前阶段的 **calibration 建模、实验设计、相关工作写作、系统可用性叙事** 非常有用；  
对你的 gaze backbone 和几何主方法帮助有限；  
但因为你当前 README 已经有明确 calibration 模块，所以它的“可转化率”很高。

---

# **附加要求：calibration / personalization / few-shot 相关分析**

## **1\. 这篇论文中的 calibration 思想更偏哪一类？**

它主要偏向下面三类，不太属于现代神经网络 personalization：

### **最主要**

**显式校准**  
因为它明确依赖用户注视已知 reference / collection marker，显式收集校准样本，再拟合映射函数。

### **次主要**

**映射误差修正**  
因为它本质是在学一个从 eye observation 到 point-of-regard 的回归映射，并通过更多样本与异常值控制提升该映射。

### **也很强**

**系统级 usability 优化**  
因为它强调更快、更独立、无需监督、自动评估、约 10 秒完成。

### **不是它的主标签**

* **隐式校准**：不是主轴，虽然文中 related work 提到 interaction / saliency-based recalibration。  
* **few-shot personalization**：严格说不算。它不是“少量样本个性化模型参数”，而是“少量或连续样本学习校准映射”。

## **2\. 它和我当前 README 中的 9 点校准 / affine transformation 有哪些关系？**

关系非常直接：

1. **同属显式校准范式**  
   你的 README 里：显示 9 个屏幕点，收集 ((raw,target))，最小二乘拟合 affine。CalibMe：显示/移动 marker，收集 ((eye, gaze))，拟合回归映射。两者都是显式 supervision 下的校准函数学习。  
2. **你的是它的简化桌面版**  
   你现在可以把 9-point affine 视为“静态少样本、低参数的显式校准实例”；CalibMe 则是“连续密集采样、带异常值处理和自动评估的显式校准扩展”。  
3. **它可以帮你解释 affine 的局限**  
   你 README 已写“仿射假设只能补偿线性误差，非线性畸变无法修正”。CalibMe 正好给你一个合理过渡：低阶多项式是比 affine 更一般的映射家族。  
4. **它也能帮你解释为什么 9 点不是终点**  
   因为 9 点只是时间/精度折中。你的论文完全可以把 9 点写成 current baseline，而不是 canonical best choice。

## **3\. 我是否可以从这篇论文中提炼出一个更学术化的 calibration 数学表达？**

可以，而且很适合你。

你现在的 README 写法已经是实现级正确版本；更学术化的版本建议写成两层：

### **一般形式**

\[  
\\hat{\\mathbf{p}} \= f\_\\theta(\\phi)  
\]  
其中 (\\phi) 是校准输入特征。

### **你的当前系统可实例化为**

\[  
\\hat{\\mathbf{p}} \= f\_\\theta(\\mathbf{p}^{raw}),\\qquad \\mathbf{p}^{raw}\\in\\mathbb{R}^2  
\]  
其中 (f\_\\theta) 可取 affine / polynomial / regression correction。

这样一写，你的 calibration 章节就从“硬编码 affine”升级成“校准函数族 \+ 当前选用实例”。

## **4\. 给我一个适合我项目的校准建模写法**

### **4.1 从原始 gaze estimate 到 screen point 的映射**

建议你在论文里这样写：

Look2Act 首先通过 gaze regression、head pose estimation 与 ray-plane intersection 获得未校准屏幕点  
\[  
\\mathbf{p}^{raw}\_i=(x\_i^{raw},y\_i^{raw}).  
\]  
由于个体差异、头姿偏差、屏幕几何近似误差以及视觉轴-光轴偏移，(\\mathbf{p}^{raw}) 与真实屏幕点 (\\mathbf{p}^{gt}) 之间通常存在系统性偏差。因此，引入用户特定校准函数  
\[  
\\mathbf{p}^{cal}*i=f*\\theta(\\mathbf{p}^{raw}\_i),  
\]  
以最小化校准样本上的映射残差。

### **4.2 affine / polynomial / regression correction 的表述**

**Affine**  
\[  
\\mathbf{p}^{cal}=A  
\\begin{bmatrix}  
x^{raw}\\y^{raw}\\1  
\\end{bmatrix},\\qquad A\\in\\mathbb{R}^{2\\times 3}  
\]

**Polynomial**  
\[  
\\mathbf{p}^{cal}=W,  
\\begin{bmatrix}  
x^{raw},y^{raw},x^{raw}y^{raw},(x^{raw})^2,(y^{raw})^2,1  
\\end{bmatrix}^T  
\]  
其中 (W\\in\\mathbb{R}^{2\\times 6})

**Regression correction（未来可选）**  
\[  
\\mathbf{p}^{cal}=f\_\\theta(x^{raw},y^{raw},yaw,pitch,roll,d)  
\]  
这句适合 discussion / future work，不要现在硬写成已实现。

### **4.3 calibration samples 的损失函数形式**

最稳妥的写法：

\[  
\\mathcal{L}*{calib}(\\theta)=\\frac{1}{N}\\sum*{i=1}^{N}  
\\left| f\_\\theta(\\mathbf{p}^{raw}\_i)-\\mathbf{p}^{gt}\_i \\right|\_2^2  
\]

如果你想体现异常样本剔除：

\[  
\\mathcal{L}*{calib}(\\theta)=\\frac{1}{\\sum\_i w\_i}\\sum*{i=1}^{N}  
w\_i\\left| f\_\\theta(\\mathbf{p}^{raw}\_i)-\\mathbf{p}^{gt}\_i \\right|\_2^2  
\]  
其中 (w\_i) 表示样本是否通过质量筛选。

### **4.4 质量评估写法**

你现在 README 里有 mean residual。可以扩展为：

\[  
e\_{fit}=\\frac{1}{N}\\sum\_i \\left| f\_\\theta(\\mathbf{p}^{raw}\_i)-\\mathbf{p}^{gt}\_i \\right|\_2  
\]

\[  
e\_{holdout}=\\frac{1}{M}\\sum\_j \\left| f\_\\theta(\\mathbf{p}^{raw}\_j)-\\mathbf{p}^{gt}\_j \\right|\_2  
\]

其中 (e\_{fit}) 是拟合误差，(e\_{holdout}) 是保留点误差。  
论文里优先报 **hold-out**，不要只报 fit residual。

## **5\. 哪些内容适合写进我的方法章节，哪些只适合写进 discussion / future work？**

### **适合写进方法章节**

* 9-point affine calibration 的统一映射表达  
* least-squares 求解  
* calibration loss / residual  
* 样本质量筛选规则  
* hold-out evaluation / recalibration threshold  
* 3×3 / 5×5 / 9-point 作为实验变量

### **只适合写进 discussion / future work**

* 隐式 calibration  
* 利用使用过程自动重校准  
* few-shot neural personalization  
* head pose / distance aware nonlinear correction network  
* 连续轨迹 / marker-based 无监督快速校准

这些东西你现在没有系统实现和实验，不要硬塞进 method。

## **6\. 如果我做 3×3 vs 5×5 calibration，或 few-shot calibration，对照实验该怎么设计更像论文？**

给你一个最像论文的设计。

### **实验目标**

回答三个问题：

1. **校准点数增加是否显著降低误差？**  
2. **更复杂的校准映射是否值得额外交互成本？**  
3. **少样本 personalization 在多大程度上能替代完整校准？**

### **实验设置**

#### **实验 A：校准点数对比**

同一批受试者、同一设备、同一距离、同一测试点集。

* No calibration  
* 1-point translation correction  
* 3-point affine  
* 5-point affine  
* 9-point affine  
* 25-point affine  
* 25-point polynomial

指标：

* mean pixel error  
* median pixel error  
* 90th percentile error  
* calibration time  
* hold-out error  
* user-rated burden（若你愿意做简单问卷）

#### **实验 B：few-shot personalization**

前提：有一个 person-independent 基础模型。

* zero-shot：无个人校准  
* 1-shot：1 个点只估计平移 bias  
* 3-shot：仿射近似  
* 5-shot：完整 affine  
* 9-shot：标准 affine  
* optional：5-shot nonlinear correction

这里最关键的是把 **few-shot** 定义清楚：  
对于你项目，few-shot 不一定要是微调模型，也可以是**少量校准样本拟合轻量 correction map**。这在论文里是完全成立的。

#### **实验 C：时变退化与重新校准**

* 校准后立即测试  
* 10 分钟后测试  
* 姿态变化后测试  
* 距离变化后测试  
* 重校准后再测

这能非常自然地连接 CalibMe 关于 calibration validity / recalibration 的讨论。

### **更像论文的统计呈现方式**

* 表 1：不同点数 / 方法的平均误差与时间  
* 图 1：点数 vs 误差  
* 图 2：点数 vs 时间  
* 图 3：不同方法在屏幕空间的误差热图  
* 图 4：few-shot 样本数 vs 性能曲线

### **一个很关键的实验细节**

**训练校准点与测试点分离。**  
不要只用校准点本身计算误差，否则你得到的是拟合能力，不是泛化能力。CalibMe 很强调 reserved evaluation points，这一点你必须学。

