这篇是 **Gaze360: Physically Unconstrained Gaze Estimation in the Wild (2019)**，它本质上是一篇 **大规模 3D gaze 数据集 \+ benchmark \+ temporal baseline \+ uncertainty modeling** 论文。对 Look2Act 来说，它的核心价值首先是 **dataset / benchmark / 真实世界困难性论证**，其次才是方法层面对 **3D representation、时序建模、跨域适配** 的启发。

## **1\. 一句话判断**

这篇论文对 Look2Act 的最大价值，不是给你一套可以直接替换当前系统的工程方法，而是给你一个更强的研究坐标系：**真实世界 gaze 不应只等于桌面正面、近距离、固定头姿，而应该考虑更宽头姿、更远距离、更复杂光照、更强 domain shift 的 3D gaze benchmark**。

它最适合帮助你的部分是：**数据集动机、公开预训练叙事、跨域泛化实验设计、3D gaze related work、真实部署困难性论证**。它和你当前 README 中最接近的是 **3D gaze direction regression / robustness / future cross-user-cross-device evaluation**，最远的是 **screen projection / calibration / desktop interaction layer**。

---

## **2\. 面向 Look2Act 的精确摘要**

这篇论文解决的核心问题是：**现有 gaze 数据集大多面向 physically constrained 场景，例如桌面或手机视线追踪，头姿与 gaze 范围有限，难以支撑真正 unconstrained 的 3D gaze estimation**。因此作者提出 Gaze360：238 名受试者，室内外环境，宽头姿、宽距离、连续视频、3D gaze 标注，并配套一个多帧时序模型和 uncertainty estimation。

它为什么和你的 Look2Act 相关，不是因为“都做 gaze”，而是因为它直接回答了你后面论文里一定会遇到的问题：  
**为什么只在自采桌面小数据上做评估不够？为什么要强调公开数据、域偏移、真实世界复杂性？为什么 3D gaze 比 2D screen point 更适合作为基础表示？**  
Gaze360 把这些问题系统地摆到了台面上。

和你当前 README 里的模块关系如下：

* **接近的模块**  
  * 3D gaze direction regression：高度接近  
  * robustness / cross-domain：高度接近  
  * potential temporal modeling：中高接近  
* **中等相关模块**  
  * head pose / facial context：中等相关  
  * face crop instead of pure eye crop：中等相关  
* **较远模块**  
  * ray-plane intersection  
  * screen coordinate projection  
  * affine calibration  
  * cursor interaction / dwell control  
    因为 Gaze360 停留在 “3D gaze estimator \+ benchmark” 这一层，而你已经走到了 “完整桌面交互系统”。

---

## **3\. 可迁移到我项目中的核心点（按优先级排序）**

### **3.1 可以直接借鉴的 3–5 个点**

**(1) 公开数据预训练 \+ 自采数据微调 \+ 实机部署评估 的整体叙事。**  
这是你最该吸收的。Gaze360 本身就强调：数据规模、场景多样性、跨域泛化，是 unconstrained gaze 的关键前提。你完全可以把它作为自己“公开数据提供基础表征，自采数据对齐目标场景，最终用真实桌面部署验证”的核心依据。

**(2) 3D gaze 表示优先于固定 2D 桌面坐标。**  
Gaze360 明确以 3D gaze 为数据与模型核心输出。这和你的 Look2Act 当前主线一致：先回归 gaze direction，再做 geometry projection。它能帮助你解释为什么不直接做 screen point regression。

**(3) cross-dataset evaluation 意识。**  
它专门做了跨数据集训练测试，并且结果显示：**Gaze360 训练出来的模型迁移到别的数据集上更强**。这对你非常重要，因为它能支撑“更丰富公开数据有助于提升泛化，而不是只提升同域分数”。

**(4) temporal information 的价值。**  
论文用 7-frame 输入 \+ BiLSTM，证明时序能改善 3D gaze 估计，尤其是帮助解决单帧歧义。你现在虽然主要是单帧 \+ EMA 平滑，但这篇论文可以支撑你后续从“后处理平滑”升级到“轻量时序建模”的研究方向。

**(5) uncertainty / confidence estimation 作为系统安全层。**  
Gaze360 用 pinball loss 预测 gaze uncertainty，并强调在大 head yaw、遮挡、远距离时不确定性应更高。这个思想对你这种桌面交互系统很有价值，因为后面你完全可以加一个“低置信度不触发点击 / 降低控制权重”的机制。

### **3.2 可能需要改造后再借鉴的 2–4 个点**

**(1) head crop 输入而非 eye crop 输入。**  
Gaze360 不依赖眼睛或脸的显式 detector，直接用 head crop 做估计，这是为了适应更宽视角、更强遮挡、更远距离场景。你当前系统是 eye crop 主导，这更适合桌面近距离精细指向。  
所以不能直接替换，但可以改造成：

* eye-only  
* eye+head / eye+face  
* fallback head-only 低置信模式  
  这样的分层方案。

**(2) 球坐标输出 ((\\theta,\\phi))。**  
它把 3D gaze 作为球坐标来回归，而你现在更偏单位向量表示。这个可以借来用于论文表述或某些 loss/visualization，但不必强行改你的主表示。

**(3) self-supervised domain adaptation。**  
论文用了 domain discriminator \+ 左右翻转一致性正则做无标注域适配。这个思路值得借，但现在不一定是你最优先做的，因为你当前更短板的是基础实验量与数据规模，而不是高级域适配模块。

### **3.3 不建议我借鉴或当前阶段不适合借鉴的点**

**(1) 把 Gaze360 的数据分布当成与你桌面系统完全匹配。**  
不对。Gaze360 强调的是 physically unconstrained、360° 范围、远距离、室内外、多人采集逻辑，这和你的桌面 webcam 正面交互系统并不等分布。

**(2) 直接拿它的 error 数值和你的屏幕精度横比。**  
它报告的是 angular error under 3D gaze benchmark，不是桌面校准后的像素误差，也不是同样 camera-screen setup 下的交互精度。不能混用。

**(3) 立即大改成 video-LSTM 主模型。**  
有研究价值，但你当前系统主线还在单帧实时桌面交互，贸然换时序主干会增加工程复杂度，可能分散当前节奏。

---

## **4\. 方法与公式层面的提炼**

### **4.1 关键变量定义**

从 Gaze360 最值得提炼的变量有：

* (p\_t)：target cross 的 3D 位置  
* (p\_e)：眼部位置  
* (g\_L \= p\_t \- p\_e)：在全局 Ladybug 坐标系中的 gaze 向量  
* (E)：观察相机的眼部坐标系  
* (g \= E \\cdot \\frac{g\_L}{|g\_L|\_2})：变换到局部眼坐标系下的单位 gaze 向量  
* ((\\theta,\\phi))：gaze 的球坐标角  
* (\\sigma)：预测的不确定性量化参数  
  这些变量里，对你最重要的是：**局部坐标系下的 3D gaze 表示、球坐标/单位向量互转、以及 uncertainty 作为附加输出**。

### **4.2 核心建模思路**

论文的核心链条是：

1. 构造一个大规模、宽姿态、宽环境的 3D gaze 数据集  
2. 用 head crop 序列输入模型  
3. 输出中心帧的 3D gaze 角度与误差 quantile  
4. 通过 cross-dataset evaluation 证明数据集与模型具有更强 in-the-wild 泛化性。

对你来说，真正可吸收的不是它的具体采集装置，而是下面这个研究表达：

\[  
\\text{diverse public 3D data} \\rightarrow \\text{robust gaze representation} \\rightarrow \\text{target-domain adaptation} \\rightarrow \\text{deployment evaluation}  
\]

这个逻辑和你后面要写的 Look2Act 论文路线高度兼容。

### **4.3 适合我项目的数学表达**

#### **A. 可以真实写进我项目方法章节的公式**

如果你继续沿用当前 Look2Act 链路，这篇论文最适合帮你补的是 **3D 表示与不确定性表达**，但不能替代你的屏幕几何链。

你可以写：

**(1) gaze regression**  
\[  
\\hat{\\mathbf{g}}*{local} \= f*\\theta(I\_{eye})  
\]  
或若后面加 face/head branch：  
\[  
\\hat{\\mathbf{g}}*{local} \= f*\\theta(I\_{eye}, I\_{face})  
\]

**(2) 坐标变换**  
\[  
\\hat{\\mathbf{g}}*{cam} \= R \\hat{\\mathbf{g}}*{local}  
\]

**(3) 射线-平面求交**  
\[  
P \= O \+ \\tau \\hat{\\mathbf{g}}*{cam}, \\quad*  
*\\tau \= \\frac{N^\\top(P\_0 \- O)}{N^\\top \\hat{\\mathbf{g}}*{cam}}  
\]

**(4) 校准映射**  
\[  
\\mathbf{p}*{cal} \= A\[\\mathbf{p}*{raw};1\]  
\]

**(5) 若以后加 uncertainty head，可写成**  
\[  
(\\hat{\\mathbf{g}}, \\hat{\\sigma}) \= f\_\\theta(I)  
\]  
其中 (\\hat{\\sigma}) 表示预测视线方向的不确定性，用于置信门控或低可信输出抑制。  
这一步受 Gaze360 启发，但你只有真的实现了再写进方法章节。

#### **B. 只能作为 related work 理解的公式**

Gaze360 的 pinball quantile loss 相关公式，尤其是对 ((\\theta,\\phi,\\sigma)) 的 10%/90% quantile 设计，更适合你作为相关工作理解或未来工作启发，而不是当前直接冒充为已实现方法。

#### **C. 不建议我照搬的公式**

它基于其数据采集装置定义的全局坐标系到眼坐标系变换，不适合你直接搬到 Look2Act，因为你的 camera/screen/subject setup 完全不同。  
你该借的是“3D gaze 局部表示”的思想，不是它具体数据采集几何。

---

## **5\. 和我当前系统架构的映射**

### **face / landmark**

**中低相关。**  
Gaze360 有意弱化对 face/eye detector 的依赖，更多依赖 head crop 与 backbone 直接学习。你当前系统则显式依赖 FaceMesh。这两者路线不同，但不冲突：Gaze360 更偏强鲁棒泛化，你更偏精细桌面几何。

### **eye crop / face crop**

**中等相关。**  
Gaze360 更像 head crop 路线，不是 eye crop 路线。它对你的启发是：未来可考虑 eye-only \+ head/face context，而不是永远只盯着局部眼图。

### **gaze regression**

**高度相关。**  
它是标准 3D gaze benchmark，这和你当前 3D gaze direction regression 完全在同一研究轨道上。

### **head pose**

**间接相关。**  
Gaze360 不像你一样显式走 PnP head pose \+ geometry 链，它更多通过数据多样性和 head crop 学隐式鲁棒性。所以它对你 head pose 的直接方法帮助有限，但对“宽 head pose 是必须覆盖的变量”这一点帮助很大。

### **geometry / ray-plane / coordinate transform**

**中低相关。**  
它的数据标注过程有明确的 3D 坐标与 gaze 向量表达，但模型论文主体不是 screen-geometry 系统。对你来说更多是 3D representation 的支持，而不是工程几何细节的模板。

### **calibration**

**几乎无关。**  
它不是桌面校准论文，不关心你的 9 点仿射校准。

### **smoothing / temporal stability**

**中等到高度相关。**  
这是它比前几篇更值得你关注的地方：它真正把时间维度放进模型，而不是后处理平滑。这是你未来可以升级的一条路线。

### **screen projection / interaction layer**

**基本无关。**  
没有桌面光标、点击、交互层设计。

---

## **6\. 对我论文写作最有用的引用建议**

### **最适合引用的位置**

最适合放在：

* **Introduction**：真实世界 gaze estimation 的困难性  
* **Related Work – Datasets / Benchmarks**  
* **Related Work – 3D gaze estimation in the wild**  
* **Experiment motivation**：为什么要考虑公开预训练 / 跨域泛化 / temporal information  
* **Discussion / Future Work**：为什么你的桌面系统仍然存在 domain gap，为什么后续值得做时序和 uncertainty

### **推荐我如何描述它（学术中文）**

表述 1：  
“Gaze360 构建了一个面向 physically unconstrained 场景的大规模 3D gaze 数据集，覆盖室内外环境、宽头姿与宽距离分布，为真实世界条件下的 gaze estimation 提供了更具代表性的 benchmark。”

表述 2：  
“该工作进一步表明，相较传统主要面向桌面或移动终端的受限数据集，更大范围的场景与姿态变化对于提升 gaze 模型的 in-the-wild 泛化能力至关重要。”

表述 3：  
“除单帧外，该工作还探索了基于多帧输入的时序 gaze estimation，并通过 uncertainty estimation 反映模型在遮挡、远距离和大视角条件下的可信度变化。”

### **推荐我如何描述它（学术英文）**

Usable sentence 1:  
“Kellnhofer et al. introduced Gaze360, a large-scale dataset for physically unconstrained 3D gaze estimation, covering indoor and outdoor scenes, wide head poses, and large subject-camera distances.”

Usable sentence 2:  
“Their cross-dataset evaluation suggests that training on broader and more diverse 3D gaze data leads to better generalization to unseen domains.”

Usable sentence 3:  
“They further showed that temporal modeling and uncertainty estimation are beneficial for gaze prediction in challenging real-world conditions.”

### **1–2 句可直接改写后使用的 related work 句式**

句式 1：  
“与主要面向桌面或手机等 physically constrained 场景的数据集不同，Gaze360 将 3D gaze estimation 推向了更宽头姿、更复杂光照及更大拍摄距离的真实环境 benchmark。”

句式 2：  
“Gaze360 的跨数据集评估结果表明，数据多样性对 gaze estimation 的跨域泛化具有关键影响，这为公开数据预训练与目标场景微调的训练范式提供了重要依据。”

### **如果这篇论文容易被我误引，请提醒我**

你最容易误引的地方有四个：

1. 不要把 Gaze360 说成“桌面 gaze interaction 数据集”。它不是。  
2. 不要把它的 360° wide-range 分布当成与你的桌面系统同分布。  
3. 不要拿它的 angular error 和你的 screen pixel error 直接比较。  
4. 不要把它的时序模型结果写成“对桌面屏幕指向一定更好”，因为那需要你自己的实验支撑。

---

## **7\. 对我实验设计最有用的启发**

### **我可以从这篇论文中借哪些实验设置**

**(1) cross-dataset / cross-domain 评估意识。**  
即使你现在还做不到完整跨数据集 benchmark，也至少应该做：

* 公开数据预训练 vs 不预训练  
* 自采数据微调 vs 不微调  
* cross-user / cross-session / cross-device holdout  
  这会让你的论文明显更像研究论文，而不是工程展示。

**(2) temporal vs single-frame 对照。**  
你当前系统已经有 EMA smoothing，可以把它升级成更规范的实验轴：

* 单帧预测  
* 单帧 \+ EMA  
* 轻量时序模型 / 帧堆叠  
  哪怕暂时不做 LSTM，也可以先做简单时间窗口平均，作为研究前哨。

**(3) uncertainty-aware 评估。**  
可以做：

* error vs confidence  
* 低置信样本剔除后性能  
* head pose / illumination / distance 分桶下置信度变化  
  这对桌面交互尤其有意义，因为系统不是“每次都必须输出动作”。

### **我可以新增哪些对照 / 消融**

优先级最高的是：

1. **公开数据预训练 vs 仅自采训练**  
2. **单帧 vs 加入轻量时序**  
3. **校准前 vs 校准后**  
4. **cross-user / cross-session**  
5. **大头姿 / 弱光 / 遮挡分桶误差分析**

### **我可以补哪些图表 / 误差分析**

* error vs head yaw / pitch  
* error vs distance to camera  
* error vs confidence  
* pretrain/fine-tune/deploy 三阶段性能图  
* 单帧 vs 时序 模型的稳定性对比图

### **它是否启发我补 cross-user / cross-device / calibration / latency / stability / usability 中的某一项**

最直接启发的是：

* **cross-user**  
* **cross-device / cross-domain**  
* **stability（通过 temporal information）**  
* **robustness（通过 uncertainty \+ wide-condition analysis）**

对 calibration 和 usability 的直接启发较少；对 latency 则提醒你时序建模会带来额外成本，需要自己权衡。

---

## **8\. 图表与可视化建议（文本描述版本）**

### **图 1：公开数据与自采桌面数据的 domain gap 结构图**

**表达什么信息：**  
说明 Gaze360 与 Look2Act 的共享点和差异点。  
**放置位置：** 主文 Related Work / Dataset Motivation。  
**结构描述：**  
左侧 Gaze360：室内外、宽头姿、远距离、3D gaze、连续视频；右侧 Look2Act：单用户桌面、普通 webcam、屏幕交互、几何投影与校准；中间标出“shared 3D gaze challenges”与“remaining domain gaps”。

### **图 2：公开预训练 \+ 自采微调 \+ 实机部署评估流程图**

**表达什么信息：**  
把你未来的训练评估范式画清楚。  
**放置位置：** 主文 Method / Training Strategy。  
**结构描述：**  
公开数据集阶段 → 学到 general gaze representation → 自采 Look2Act 微调阶段 → 几何投影与校准 → 实时桌面部署评估。

### **图 3：单帧 vs 时序模型 对比图**

**表达什么信息：**  
说明 temporal information 是否提升稳定性与准确性。  
**放置位置：** 主文 Experiment。  
**结构描述：**  
横轴为不同条件分桶（静止、头动、遮挡、弱光），纵轴为 angular/pixel error 和 temporal jitter，两组柱形分别代表 single-frame 与 temporal variant。

### **图 4：error-confidence 联动图**

**表达什么信息：**  
说明模型在大角度、遮挡、低质样本下能否给出更低可信度。  
**放置位置：** 主文实验或附录。  
**结构描述：**  
散点图或分桶图，横轴为 predicted confidence / uncertainty，纵轴为 actual error。

---

## **9\. 风险与边界**

### **哪些地方我容易“看懂了但写不出来”**

**(1) “in-the-wild” 与 “desktop real-world” 的关系。**  
你很容易觉得 Gaze360 很强，所以可以直接替你证明桌面系统。但实际上它证明的是 **更广义 unconstrained 3D gaze**，不是你的具体桌面交互任务。写作时必须把这个层次讲清楚。

**(2) temporal modeling 的迁移。**  
Gaze360 的时序输入是为 head crop 3D gaze benchmark 设计的，不等于你加个 LSTM 就一定提升桌面 screen-point accuracy。你需要自己的实验。

### **哪些地方容易“看起来高级但不适合我的项目”**

**(1) 直接把 head crop 替代 eye crop 主线。**  
对你的当前系统未必合适。桌面近距离指向通常更依赖眼部细粒度信息。

**(2) 现在就上 domain adversarial adaptation。**  
对研究上有价值，但当前不一定是最高 ROI。你更该先把基础实验做完整。

### **哪些 claim 我不能乱说**

* 不能说 Gaze360 与 Look2Act 场景完全一致  
* 不能说 Gaze360 的性能结果可直接说明你的桌面系统性能上限  
* 不能说 temporal model 一定适合实时桌面系统  
* 不能说 uncertainty 输出就等于系统可用性提升，除非你自己做了 interaction-level 验证

### **哪些方法如果没有额外实验支撑就不要写成我的贡献**

* “公开数据预训练显著提升泛化”  
* “时序建模显著提升稳定性”  
* “uncertainty 能有效过滤危险输出”  
* “宽姿态训练使系统对真实桌面更鲁棒”  
  这些都必须靠你的实验，不要只靠 Gaze360 帮你背书。

---

## **10\. 最终落地建议**

### **“我现在就该做”的 1–3 个动作**

**(1) 立刻把 Gaze360 放进你论文的数据集 / benchmark 相关工作核心位置。**  
它会显著增强你的 dataset motivation 与真实世界困难性论证。

**(2) 尽快规划“公开数据预训练 vs 自采微调”的实验框架。**  
哪怕先不真正训练完，也要先把实验协议写出来。

**(3) 在你的实验设计里补上至少一个 temporal/stability 方向的小对照。**  
哪怕只是单帧 vs EMA / 小时间窗平均，也比完全没有时序意识强很多。

### **“以后可以做但现在不要分散精力”的 1–3 个动作**

**(1) 完整复现 Gaze360 的 BiLSTM \+ uncertainty pipeline**  
**(2) 上更复杂的无监督域适配**  
**(3) 把目标拓展到大角度、远距离、非桌面场景**  
这些都值得做，但不是你当前 Look2Act 主线最优先的。

### **总评分**

**9.0 / 10**

原因：  
它对 Look2Act 当前阶段的价值非常高，尤其高在 **数据动机、benchmark 地位、预训练叙事、跨域泛化意识、真实世界困难性论证**。  
它对你的直接工程落地帮助不如几何建模论文高，但对你未来“怎么把项目写成像论文”非常关键。

---

# **附加分析：这是一篇数据集 / benchmark / 奠基性代表论文**

## **A1. 这篇论文的数据集对我的 Look2Act 最有价值的地方是什么？**

最有价值的地方不是“样本多”本身，而是它给了你一个非常强的论据：  
**高质量 gaze 研究不能只建立在小范围、固定视角、受控桌面数据上；真实世界 gaze 需要更大 head pose、更远距离、更复杂光照、更明显域偏移的 benchmark。**  
这会直接抬高你论文的研究视野。

第二个价值是，它把 **3D gaze** 而不是 **2D screen point** 作为核心表示，这和你当前 Look2Act 的主线高度兼容。对你来说，它能支撑“先学 gaze direction，再做几何映射”的合理性。

第三个价值是 cross-dataset evidence。表 3 很关键：**用 Gaze360 训练的模型迁移到 Columbia、MPIIFaceGaze、RT-GENE 上都更强**，这说明更丰富的公开 3D gaze 数据可以提供更好的通用表征。这个点对你后续写“公开预训练 \+ 自采微调”特别重要。根据表 3，Gaze360 训练后在 Columbia、MPIIFaceGaze、RT-GENE 测试上的平均角误差分别为 9.0°、12.1°、23.4°，优于其他训练源；再加上无监督域适配后分别进一步改善到 8.1°、9.9°、21.9°。

---

## **A2. 它的数据采集条件、标注方式、场景分布，和我的真实桌面系统之间有哪些 domain gap？**

有，而且很明显。

**(1) 采集装置 gap**  
Gaze360 用的是 Ladybug5 360° 多相机全景设备 \+ 移动 target board \+ AprilTag 标定体系；你的 Look2Act 是普通单目 RGB webcam。这个差异非常大。

**(2) 场景约束 gap**  
Gaze360 强调 physically unconstrained：1–3 米距离、室内外环境、宽头姿、多人同时采集、远距离、模糊、遮挡。你当前系统是桌面 webcam 近距离人机交互，用户通常面对屏幕，头姿范围和目标空间都更受限。

**(3) 标签定义 gap**  
Gaze360 的标签是 3D gaze direction，来源于 target board 与眼位几何；你的最终任务则是屏幕注视点，需要进一步经过 ray-plane intersection 与 calibration。也就是说，Gaze360 的标签更接近你的中间表示，不是最终交互标签。

**(4) 输入分布 gap**  
Gaze360 模型主要吃 head crop，甚至考虑部分眼睛不可见的情形；你的系统强调 face landmarks \+ eye crop 精细建模。两者关注的视觉证据分布不同。

**(5) 时序与交互目标 gap**  
Gaze360 有连续视频，但目标是 3D gaze benchmark；你的系统目标是实时桌面控制、稳定屏幕点和可用交互。  
所以它和你是“研究上相通，任务上不等价”。

---

## **A3. 我是否适合把它作为：预训练来源 / baseline 对照背景 / “真实世界困难性”的证据**

### **1\. 预训练来源**

**适合，而且很适合。**  
前提是你要明确：它更适合作为 **3D gaze representation 的预训练来源**，而不是直接拿来学桌面 screen point。  
因为它的标签层级与你的中间表示接近，且场景多样性强。

### **2\. baseline 对照背景**

**非常适合。**  
它是非常典型的 benchmark / background reference。你完全可以在 Related Work 里把它作为“in-the-wild 3D gaze benchmark”的代表。

### **3\. “真实世界困难性”的证据**

**非常适合，而且这是它最强的用途之一。**  
它的整个论文都在说明：真实场景下 wide pose、distance、occlusion、lighting、domain shift 会显著增加难度。  
这正好可以为你的 Look2Act 提供问题背景。

---

## **A4. 如果我要在自己的论文里写“公开数据预训练 \+ 自采数据微调 \+ 真实部署评估”，这篇论文能提供哪些论据？**

它至少能提供四类非常直接的论据：

**(1) 公开数据的必要性。**  
真实 gaze 数据很难采，尤其是 wide pose、室内外、长距离、多人、多时段的数据。Gaze360 为了得到这类数据，专门设计了复杂但高效的采集体系。这说明公开数据对学习通用表示非常重要。

**(2) 数据多样性比单域高精度更重要。**  
Gaze360 的 cross-dataset 结果表明，训练在更广分布数据上的模型，往往对别的数据域迁移更好。这是你做公开预训练最强的理论与实验证据之一。

**(3) 目标域微调依然必要。**  
论文进一步做了无监督域适配，并显示可以继续改善新域表现。这从侧面说明：即便公开数据很强，目标场景适配仍然重要。对你来说，这就是“自采微调”的合理性来源。

**(4) 真实部署评估不能被 benchmark 替代。**  
Gaze360 虽然 benchmark 很强，但作者仍然展示了 YouTube 视频和超市货架应用。这说明 benchmark 结果和实际应用验证是两件不同的事。对你来说，这正好能支撑“公开预训练 \+ 自采微调 \+ 桌面真实部署评估”三段式结构。

---

## **A5. 3 条最适合写进 related work / dataset motivation 的句子**

句子 1：  
“Gaze360 构建了面向 physically unconstrained 场景的大规模 3D gaze 数据集，覆盖室内外环境、宽头姿与较大拍摄距离，为真实世界 gaze estimation 提供了比传统桌面或移动端数据集更具挑战性的 benchmark。”

句子 2：  
“其跨数据集实验结果表明，训练于更大范围、更高多样性的 3D gaze 数据能够显著提升模型在 unseen domain 上的迁移性能，这为公开数据预训练提供了有力依据。”

句子 3：  
“尽管大规模公开数据有助于学习更通用的 gaze 表征，Gaze360 的域适配实验也表明，面向具体应用场景的进一步适配仍然必要，这支持了‘公开预训练 \+ 目标域微调 \+ 实际部署评估’的研究范式。”

