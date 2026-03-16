论文是 **“It’s Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation” (2017)**。

## **1\. 一句话判断**

这篇论文对 Look2Act 的价值，**主要不在于几何链条本身**，而在于它为你提供了一个很强的方法启发：**视线估计不一定只能盯着眼睛，整张脸里也包含对 gaze 很有帮助的信息，尤其是在复杂光照、极端 gaze 方向和大 head pose 下**。

对你当前项目最有帮助的部分依次是：**输入表示设计（eye-only vs full-face / multi-region）、鲁棒性叙事、3D gaze representation 的论文写法、实验分析图表设计**。它对你现有的 **ray-plane intersection / screen projection / calibration** 这条系统级几何链帮助较小，不是那种几何建模论文。你应该主要借它的 **表示与叙事**，其次才是局部方法细节。结合你 README，最接近的是前端感知与 gaze regression 输入设计，最远的是交互层与屏幕映射层。

---

## **2\. 面向 Look2Act 的精确摘要**

这篇论文解决的问题是：**过去很多 appearance-based gaze estimation 方法只用单眼或双眼图像，而作者要验证“只用整张脸”是否也能做得更好，尤其是在 2D 与 3D gaze estimation 两个任务上**。他们提出了 full-face CNN，并加入 spatial weights 机制，让网络自动学习“人脸不同区域对 gaze 估计的重要性不同”。

为什么和你的 Look2Act 相关：  
因为你当前 README 的主链还是典型的 **face/landmark → eye crop → gaze regression → geometry → screen projection**，本质上仍然是“眼部主导”的设计。该论文提醒你：**当场景进入真实桌面、复杂光照、头部偏转、非理想姿态时，仅依赖眼部局部纹理可能不够，脸部其余区域携带的 head pose、光照分布、脸部外观上下文也可能显著帮助 gaze estimation**。这对你的系统特别 relevant，因为你面向的是普通 webcam、桌面真实使用、非受控环境。

和你当前 README 中最接近的模块：

* **face / landmark**：高度接近。它以 full-face 为主输入，仍依赖 face alignment / landmarks。  
* **gaze regression**：高度接近。它也做 2D / 3D gaze 回归。  
* **head pose**：中等接近。它讨论 head pose 对性能的重要性，但不是一篇专门做 head pose estimation 的论文。  
* **geometry / ray-plane / screen projection**：中低接近。它在 3D 任务里会把 3D gaze 与 target plane 求交得到 2D 点，但这不是论文主创新点。  
  最远的是：  
* **calibration**  
* **smoothing / temporal stability**  
* **interaction layer / dwell click / desktop control**  
  这些几乎不是它关心的问题。

---

## **3\. 可迁移到我项目中的核心点（按优先级排序）**

### **3.1 可以直接借鉴的 3–5 个点**

**(1) full-face 作为 gaze estimation 输入表示的研究动机。**  
这是这篇论文最直接可迁移的点。你当前系统虽然有 face landmark 和 head pose，但真正喂给 gaze 模型的核心还是 eye crop。该论文给你一个非常自然的研究扩展方向：

* eye-only baseline  
* eyes \+ face multi-region  
* full-face only  
  你完全可以把它做成 Look2Act 的一组论文级对照。

**(2) 3D gaze estimation 比 2D screen mapping 更适合通用部署叙事。**  
论文明确区分了 2D gaze estimation 与 3D gaze estimation，并指出 3D formulation 更通用，因为它不依赖固定 camera-screen 关系；之后再通过几何把 gaze ray 与目标平面求交得到 screen point。这个逻辑和你当前 README 高度对齐。

**(3) “复杂条件下 full-face 更稳健”的叙事。**  
论文显示 full-face 方法在 illumination、极端 gaze direction、极端 head pose 下优势更明显。对你来说，这不是一句漂亮话，而是可直接转化为实验设计和 Discussion 结构。

**(4) 区域重要性图（region importance map）分析范式。**  
这对你很有价值，因为你后面写论文时不能只报 mean error。你可以借它的思路做：

* eye-only 模型关注什么  
* full-face / face-assisted 模型关注什么  
* 在不同 head pose / illumination 下注意区域怎么变  
  这类图会让你的论文明显更像研究，而不是工程报告。

**(5) 把 facial appearance 与 head pose 区分开来分析。**  
论文专门做了一个很好的实验：遮住眼睛，用仅 head pose 的 naive baseline / regression baseline 与 eye-blocked full-face 模型比较，结论是 full-face 带来的收益不只是“它偷偷学会了 head pose”，而是 facial appearance 本身也有价值。这个分析意识很重要。

### **3.2 可能需要改造后再借鉴的 2–4 个点**

**(1) spatial weights 机制。**  
这个机制本质上是在 feature map 上乘一个空间权重图：  
\[  
V\_c \= W \\odot U\_c  
\]  
思路可借，但不建议你现在原样照搬。它是 2017 年 AlexNet 时代的设计。你可以把它简化成“attention-like spatial modulation” 的实验思路，而不是机械复现。

**(2) full-face only 的极端设计。**  
论文主打“只用 full face”。但你的 Look2Act 当前系统已经稳定建立在 eye crop \+ geometry \+ calibration 这条链上。对你来说，更现实的改造方式是：

* 先做 **eye-only vs eye+face**  
* 再考虑 full-face only  
  而不是一下子把眼部彻底扔掉。

**(3) 不显式输入 head pose 的 full-face 方案。**  
论文在 full-face 3D case 中发现额外显式输入 head pose 特征没有进一步提升，因此仅用 image features。但这并不意味着你也应删掉 PnP。因为你的系统后半段的 **坐标变换与 ray-plane** 明确依赖 head pose / geometry。也就是说，论文是在“估计器输入”层面弱化 pose，而你在“系统几何链”层面不能弱化 pose。

### **3.3 不建议我借鉴或当前阶段不适合借鉴的点**

**(1) 用这篇论文来替代你的几何链设计。**  
不合适。它不是 geometry-first 论文，不会替代你现有 PnP → coordinate transform → ray-plane → affine calibration 这条链。

**(2) 把它写成“head pose 方法论文”。**  
不准确。它会分析 head pose，但核心贡献是 **full-face representation \+ spatial weights \+ robustness analysis**。

**(3) 直接照搬 AlexNet \+ 448×448 full-face 架构。**  
从你当前 Look2Act 的实时部署目标看，这样做很可能在 CPU-only 或轻量部署上代价过高。你的 README 明确强调实时与部署，这一点要守住。

---

## **4\. 方法与公式层面的提炼**

### **4.1 关键变量定义**

从这篇论文中提炼出的关键变量，你最该掌握的是：

* ( I )：输入图像  
* ( p )：2D 屏幕 / 平面 gaze location  
* ( g )：3D gaze vector  
* ( x )：3D reference point（如眼中心、两眼中心、或面部参考点）  
* ( M \= SR )：3D normalization conversion matrix  
* ( W )：spatial weight map  
* ( U )：卷积特征张量  
* ( V )：加权后的卷积特征张量

这组符号里，对你最有价值的不是 (W,U,V) 这些网络内部符号，而是：

* **2D vs 3D gaze task 的严格区分**  
* **reference point (x)**  
* **3D gaze vector 与 plane intersection 的写法**  
* **normalization space / camera coordinate / target plane 的关系**

### **4.2 核心建模思路**

论文的关键建模思路其实有两层：

第一层是任务定义层：

* 2D gaze estimation：  
  \[  
  p \= f(I)  
  \]  
  这里 (p) 是屏幕或目标平面上的 2D 点。  
* 3D gaze estimation：  
  \[  
  g \= f(I)  
  \]  
  这里 (g) 是 3D gaze vector。之后再与 target plane 求交得到 2D gaze location。

第二层是表示学习层：  
网络不再只看眼睛，而看 full-face，并通过 spatial weights 机制自动强调更有信息的 facial regions：  
\[  
V\_c \= W \\odot U\_c  
\]  
这说明它在“输入表示”上做文章，而不是在几何链上做大创新。

### **4.3 适合我项目的数学表达**

#### **A. 可以真实写进我项目方法章节的公式**

这篇论文里，最适合你写进 Look2Act 方法章节、且与你系统兼容的，是下面这组“3D estimator \+ 几何投影”表达。

**(1) 眼部/脸部观测到 gaze 向量**  
若你后续扩展成 eye+face 或 full-face 辅助输入，可写：  
\[  
\\hat{\\mathbf{g}}*{local} \= f*\\theta(I\_{eye}, I\_{face})  
\]  
如果保持你当前实现主线，也可写成更保守版本：  
\[  
\\hat{\\mathbf{g}}*{local} \= f*\\theta(I\_{eye})  
\]  
然后在扩展实验中定义：  
\[  
\\hat{\\mathbf{g}}*{local}^{face} \= f*\\phi(I\_{face})  
\]  
作为对照分支。这样表述比直接宣称你已经 full-face 要严谨得多。

**(2) 头姿变换到相机坐标系**  
结合你当前 README：  
\[  
\\hat{\\mathbf{g}}*{cam} \= R \\hat{\\mathbf{g}}*{local}  
\]  
其中 (R) 由 face landmarks \+ PnP 得到。

**(3) 与屏幕平面求交**  
\[  
P \= O \+ \\tau \\hat{\\mathbf{g}}*{cam}, \\quad*  
*\\tau \= \\frac{N^\\top(P\_0 \- O)}{N^\\top \\hat{\\mathbf{g}}*{cam}}  
\]  
这是你当前系统里已经有、且与该论文 3D formulation 逻辑一致的表达。

**(4) 校准映射**  
\[  
\\mathbf{p}*{cal} \= A\[\\mathbf{p}*{raw};1\]  
\]  
这虽然不是该论文核心，但和你的实际系统落地强相关，必须保留。

#### **B. 只能作为 related work 理解的公式**

**(1) 2D 直接映射**  
\[  
p \= f(I)  
\]  
它适合你在 Related Work 里解释“早期或固定 setup 下的 2D mapping 思路”，但不适合拿来当你的主方法，因为你现在明确在走 3D \+ geometry 这条更通用的路。

**(2) normalization matrix**  
\[  
M \= SR  
\]  
以及  
\[  
\\hat{g} \= Mg, \\quad g \= M^{-1}\\hat{g}  
\]  
这组式子很适合放在相关工作理解、或你论文中的“受启发的几何标准化叙事”里。但如果你没有真的做这套 perspective warp normalization，就不要冒充为你的实际实现。

**(3) spatial weights**  
\[  
V\_c \= W \\odot U\_c  
\]  
这个式子适合在你未来做“face-assisted branch”时参考，但当前阶段不必硬写成你主方法。

#### **C. 不建议我照搬的公式**

**(1) 它的 full-face-only regression 作为你唯一主线。**  
当前不建议。因为你已经建立了 eye crop \+ head pose \+ geometry 的稳定系统链，贸然改成 full-face only 容易把项目主线打散。

### **4.4 如果原论文公式太复杂，适合你写进自己论文的简化版**

你项目里最适合写的方法表达可以压缩成这四步：

\[  
\\text{Face / landmarks} \\rightarrow R,t  
\]

\[  
I\_{eye}, I\_{face} \\rightarrow \\hat{\\mathbf{g}}\_{local}  
\]

\[  
\\hat{\\mathbf{g}}*{cam} \= R \\hat{\\mathbf{g}}*{local}  
\]

\[  
P\_{screen} \= \\Pi(\\hat{\\mathbf{g}}*{cam}, \\Pi*{screen}), \\quad \\mathbf{p}*{cal}=A\[\\mathbf{p}*{raw};1\]  
\]

其中：

* (R,t)：由 face landmarks 和 PnP 得到的头部位姿  
* (I\_{eye}, I\_{face})：眼部与脸部图像观测  
* (\\hat{\\mathbf{g}}\_{local})：局部坐标系下预测 gaze direction  
* (\\Pi\_{screen})：屏幕平面  
* (P\_{screen})：未校准投影点  
* (\\mathbf{p}\_{cal})：校准后屏幕点

这套表达既能吸收这篇论文对 3D gaze representation 的启发，又不会破坏你自己的 Look2Act 主线。

---

## **5\. 和我当前系统架构的映射**

### **face / landmark**

**高度相关。**  
这篇论文 full-face 输入建立在 face crop / landmark alignment 基础上。你当前系统也以 FaceMesh 为前置。这个模块可以直接对齐。

### **eye crop / face crop**

**非常相关。**  
论文最核心的就是：过去大多只看 eye region，而 full-face 可能更强。对你来说，这正好映射到一个很自然的实验轴：

* eye-only  
* face-only  
* eye+face  
  你 current README 里这一块还有明显可扩展空间。

### **gaze regression**

**高度相关。**  
论文既做 2D 也做 3D gaze regression。你的系统主打 3D gaze direction regression。这里有非常直接的可对齐性，尤其是“3D 比 2D 更适合通用部署”的写法。

### **head pose**

**中等相关，但不是主创新。**  
它分析了 head pose 的作用，也说明 extreme head pose 下 full-face 更有帮助；但它不是一篇 head pose estimator 论文。对你来说，最有价值的是：它能支撑你在论文里强调 head pose 与 facial context 是 gaze estimation 的重要影响因素。

### **geometry / ray-plane / coordinate transform**

**中等相关。**  
论文在 3D formulation 里明确写到：给定 3D gaze vector 和目标平面姿态，可通过与 plane 相交得到 gaze location。这和你现有几何链方向一致。  
但注意：它只是把这件事当标准 3D gaze formulation，不是它的主要贡献。

### **calibration**

**弱相关。**  
它不讨论你这种 9 点仿射校准，也没有你的桌面交互级 calibration 设计。这里不要硬套。

### **smoothing / temporal stability**

**几乎无关。**

### **screen projection / interaction layer**

**弱到中等相关。**  
只在“3D gaze 最终可投影到目标平面”这个层面相关；不涉及你的桌面控制、光标移动、平滑、点击等。

---

## **6\. 对我论文写作最有用的引用建议**

### **最适合引用的位置**

最适合放在：

* **Related Work – Appearance-based gaze estimation**  
* **Related Work – Full-face / multi-region representation**  
* **Method motivation**  
* **Experiment / Discussion（极端 head pose、复杂光照下的鲁棒性）**

如果你后续真的做了 face-assisted 或 full-face 分支，它也可以少量进入 Method 里的设计动机部分。

### **推荐我如何描述它（学术中文）**

表述 1：  
“Zhang 等进一步突破了仅依赖眼部区域的传统 appearance-based gaze estimation 设定，提出基于整脸输入的 full-face gaze estimation 框架，并表明面部非眼部区域同样蕴含有助于视线估计的判别信息。”

表述 2：  
“该工作指出，full-face 表示在复杂光照、极端视线方向以及较大头部姿态变化条件下具有更好的鲁棒性，说明 gaze estimation 的有效线索并不限于眼周局部纹理。”

表述 3：  
“在 3D gaze estimation 设定下，该工作进一步强调了从通用三维视线方向出发、再结合目标平面几何求得屏幕注视点的建模路径，相较直接 2D 屏幕回归具有更强的设备与场景泛化潜力。”

### **推荐我如何描述它（学术英文）**

Usable sentence 1:  
“Zhang et al. showed that full-face appearance can provide discriminative cues beyond the eye region, and proposed a full-face CNN that improves both 2D and 3D gaze estimation.”

Usable sentence 2:  
“Their analysis suggests that non-eye facial regions become increasingly informative under challenging illumination, extreme gaze directions, and large head poses.”

Usable sentence 3:  
“For 3D gaze estimation, they also highlighted the practical advantage of regressing a 3D gaze direction and subsequently intersecting it with a target plane, rather than directly learning a fixed 2D screen mapping.”

### **1–2 句可直接改写后使用的 related work 句式**

句式 1：  
“与仅依赖单眼或双眼区域的传统 appearance-based gaze estimation 方法不同，full-face 方法表明整脸外观中包含可用于视线推断的重要上下文信息，尤其在复杂光照与较大头部姿态变化条件下更为明显。”

句式 2：  
“相关研究进一步指出，3D gaze direction regression 相较固定屏幕平面上的 2D 回归更具通用性，可在后续结合几何投影映射到任意目标平面，因此更适合作为真实部署系统的基础表示。”

### **容易被误引的地方**

* 不要把它误写成“几何建模论文”  
* 不要把它误写成“head pose 显式输入论文”，因为它甚至在 full-face 3D case 中发现额外加 pose feature 没提升  
* 不要把它说成“证明 full-face 可以替代所有几何模块”  
* 不要把它当作 calibration 或交互系统论文

---

## **7\. 对我实验设计最有用的启发**

### **我可以从这篇论文中借哪些实验设置**

**(1) 输入表示对照实验**  
这是你最该借的：

* eye-only  
* face-only  
* eye+face  
* full-face with simple attention / weighting  
  这组实验如果做出来，会直接增强你的 Method 与 Experiment。

**(2) 3D task vs 2D task 的叙事区分**  
即使你不真的训练一个 2D direct mapping 模型，也可以在论文中把“为什么最终采用 3D representation”写清楚，并在实验里给出后置投影误差作为对应指标。

**(3) 条件分桶鲁棒性分析**  
按以下因素分组：

* illumination  
* gaze direction  
* head pose  
  这是这篇论文最强的实验启发之一。

### **我可以新增哪些对照 / 消融**

**优先级最高：**

1. **eye-only vs eye+face**  
   这是最现实、最贴合你现阶段工程基础的一组。  
2. **有无遮挡的 facial context 对照**  
   比如只保留眼周 vs 扩展到眉毛鼻梁 vs 整脸，做分级输入范围实验。  
3. **极端 head pose 条件下的性能对照**  
   哪怕你先用离散分桶：小、中、大 yaw/pitch。  
4. **illumination 分桶误差分析**  
   可先用 face region mean intensity / 左右亮度差做 proxy。

### **我可以补哪些图表 / 误差分析**

* error vs head yaw/pitch 分桶图  
* error vs illumination imbalance  
* eye-only 与 eye+face / full-face 的 region importance 对比图  
* full-face 在 extreme pose 样本上的 qualitative case study

### **它是否启发我补 cross-user / cross-device / calibration / latency / stability / usability 中某一项**

最直接启发的是：

* **cross-user robustness**  
* **head pose robustness**  
* **illumination robustness**  
* **representation-level ablation**

对 calibration / latency / usability 的直接启发不大。

---

## **8\. 图表与可视化建议（文本描述版本）**

### **图 1：Look2Act 输入表示设计图**

**表达什么信息：**  
说明你为什么从 eye-only 扩展到 eye+face 或 full-face assisted representation。  
**放置位置：** 主文 Method。  
**图结构描述：**  
左侧为原始 webcam frame；中间分出三个分支：eye crop、face crop、head pose estimation；右侧为 gaze regression 模块和 geometry projection 模块；在 gaze regression 前标注“appearance cues from eyes” 与 “context cues from full face”。

### **图 2：不同输入表示在复杂条件下的性能对照图**

**表达什么信息：**  
说明 full-face 或 face-assisted 输入在 extreme head pose / poor lighting 下更稳。  
**放置位置：** 主文 Experiment。  
**图结构描述：**  
横轴为 head pose 或 illumination 分桶，纵轴为 angular error / pixel error；多条曲线分别是 eye-only、face-only、eye+face。

### **图 3：region importance / attention 可视化图**

**表达什么信息：**  
展示模型在不同 gaze direction 或 head pose 下依赖的人脸区域不同。  
**放置位置：** 主文实验或附录。  
**图结构描述：**  
上排是原始 face patch，下排是 importance heatmap；按水平 gaze、垂直 gaze、水平 head pose、垂直 head pose 分组展示。

### **图 4：3D gaze → screen point 的几何链条图**

**表达什么信息：**  
把这篇论文的 3D formulation 与你的系统链路统一起来。  
**放置位置：** 主文 Method Overview。  
**图结构描述：**  
面部观测 → head pose via PnP → local gaze vector → camera-frame gaze vector → ray-plane intersection → screen projection → affine calibration。

---

## **9\. 风险与边界**

### **哪些地方我容易“看懂了但写不出来”**

**(1) “full-face 为什么有效” 的机制解释。**  
你很容易理解“整张脸有用”，但如果没有实验支撑，写出来会变成空话。你需要至少有：

* 输入表示对照  
* 条件分桶分析  
* 可视化证据  
  否则只能写成相关工作观察，不能写成你自己的结论。

**(2) 3D representation 与几何投影的边界。**  
这篇论文会让你觉得“3D gaze \+ target plane intersection 很自然”，但你不能省略自己的系统具体假设，比如相机内参、屏幕平面定义、参考点定义、校准流程。

### **哪些地方容易“看起来高级但不适合我的项目”**

**(1) full-face only 作为当前主线。**  
你现在更适合做 face-assisted，而不是完全 full-face-only。因为 Look2Act 当前已经围绕 eye crop 和轻量部署做了不少工程积累。

**(2) spatial weights 大改网络。**  
从你的项目节奏看，现在最大的短板不是“缺一个 fancy 模块”，而是实验设计与论文叙事还不够扎实。

### **哪些 claim 我不能乱说**

* 不能说 “head pose 不重要，因为 full-face 已经隐式包含了它”  
  对你的系统来说这是错的，因为后续几何投影明确依赖 pose。  
* 不能说 “full-face 一定优于 eye-only”  
  你得自己验证，尤其在你的数据规模还不大时。  
* 不能说 “这篇论文证明 3D gaze 就能直接得到屏幕点，无需额外几何/校准”  
  这显然不对。  
* 不能说 “这篇论文主要解决了桌面 gaze interaction”  
  它不是系统交互论文。

### **哪些方法如果没有额外实验支撑就不要写成我的贡献**

* “full-face 表示提升了复杂光照鲁棒性”  
* “full-face 特别适合极端 head pose”  
* “face context 显著优于 eye-only”  
* “spatial weighting 带来注意力机制收益”  
  这些都必须靠你自己的实验支撑。

---

## **10\. 最终落地建议**

### **“我现在就该做”的 1–3 个动作**

**(1) 把这篇论文转化为你的一个明确实验计划：eye-only vs eye+face。**  
不要空想 full-face-only，先做最现实的辅助输入对照。

**(2) 在论文方法草稿里，把你的表示层与几何层分开写。**  
也就是明确区分：

* appearance representation  
* geometric projection  
  这会让整体结构比现在更科研化。

**(3) 新增一组 head pose / illumination 分桶误差分析。**  
这是这篇论文最值得你吸收的实验精神。

### **“以后可以做但现在不要分散精力”的 1–3 个动作**

**(1) 复现它的 spatial weights CNN**  
不是当前最优先。

**(2) 全面转成 full-face only 模型**  
对你现在系统风险太大。

**(3) 把 head pose 从系统里拿掉，完全交给网络隐式学习**  
不适合 Look2Act 当前项目阶段。

### **总评分**

**8.0 / 10**

原因：  
这篇论文对 Look2Act **表示设计、鲁棒性叙事、3D gaze 写法、实验分析** 很有价值；但对你当前最核心的系统几何链创新帮助有限。它更像是帮你把“估计器前半段”变得更像研究，而不是替你重写整套方法。

---

# **附加分析：这是一篇 head pose / geometry / 3D gaze / representation 相关论文**

## **A1. 这篇论文里 head pose 是什么角色？**

你的这道问题必须答得很清楚：**它不是单一角色，而是多重角色，但主角色不是“显式输入特征”。**

### **结论先说：**

* **输入特征**：在一般 2D/3D gaze literature 综述层面，是常见输入；但在这篇 full-face 3D 实验的最终方案里，作者明确说额外的 head pose feature 没带来提升，因此最终只用 image features。  
* **辅助监督**：不是。  
* **几何约束**：是，在 3D normalization 和 target plane intersection 里是几何前提。  
* **disentanglement 对象**：不是严格意义上的 disentanglement 论文。  
* **仅作为分析因素**：也是。作者专门分析了 head pose 条件下 full-face 的收益，尤其在 extreme pose 下更明显。

### **更准确地说：**

对这篇论文，head pose 的角色可归纳为：

1. **几何建模中的必要变量**  
   用于 3D gaze normalization、reference point 定义、camera / normalized coordinate 变换。  
2. **分析因素**  
   作者把 head pose 当作分桶分析变量，证明 full-face 在 extreme head pose 时更有优势。  
3. **潜在隐式信息来源**  
   full-face 中很多非眼部区域可能隐式编码 head pose 线索，因此作者发现显式再喂 head pose feature 并未继续提升。

所以如果你要一句话总结：  
**在这篇论文里，head pose 主要是几何链中的必要条件 \+ 鲁棒性分析因素，而不是最终模型中最核心的显式输入特征。**

---

## **A2. 它和我当前的四个模块有什么可对齐的地方？**

### **1\. 3D gaze direction regression**

**高度可对齐。**  
它明确支持 3D gaze vector regression，而不是只做 2D 屏幕点回归。这和你当前 Look2Act 的 3D gaze direction regression 非常对齐。你完全可以引用它来说明为什么你不直接把网络输出定义成屏幕坐标。

### **2\. head pose via PnP**

**中等可对齐。**  
论文在 3D case 中用 generic 3D face model 拟合 landmark 来得到 head pose，用于 normalization。你当前是 landmarks \+ PnP 求 (R,t)。虽然实现不同，但论文逻辑与你对齐：  
**head pose 是 3D gaze system 的基础几何量。**

### **3\. ray-plane intersection**

**高度可对齐，但它不是论文主创新。**  
论文明确说 3D gaze vector 与 target plane 相交即可得到 2D gaze location。你的系统已经在这么做，所以这篇论文可以作为你这条链的一个相关工作支撑。

### **4\. screen coordinate projection**

**中等可对齐。**  
它只把这一层当作几何后处理；而你把它扩展成了完整 screen-space interaction pipeline。所以可对齐的是“理论链条”，不可对齐的是“系统完整度”。

---

## **A3. 适合我项目写法的几何链条**

这是你可以直接吸收到自己论文里的版本：

### **几何链条（适合 Look2Act）**

1. **眼部/脸部观测**  
   从单目 RGB webcam 图像中提取 face landmarks、eye crops，以及可选的 face crop / full-face appearance。  
2. **头姿估计**  
   利用人脸关键点与通用 3D face model，通过 PnP 求得头部相对于相机的旋转 (R) 与平移 (t)。  
3. **gaze 向量估计**  
   基于眼部或眼脸联合外观输入，回归局部坐标系下的 gaze direction (\\hat{\\mathbf{g}}\_{local})。  
4. **坐标系变换**  
   利用头部旋转矩阵将局部 gaze 向量变换到相机坐标系：  
   \[  
   \\hat{\\mathbf{g}}*{cam} \= R \\hat{\\mathbf{g}}*{local}  
   \]  
5. **与屏幕平面求交**  
   将以眼球中心或面部参考点为起点、方向为 (\\hat{\\mathbf{g}}*{cam}) 的 gaze ray 与屏幕平面相交，得到未校准屏幕注视点：*  
   *\[*  
   *P*{raw} \= O \+ \\tau \\hat{\\mathbf{g}}\_{cam}  
   \]  
6. **屏幕坐标投影与校准**  
   将相交点投影为屏幕二维坐标，并通过仿射校准映射到最终屏幕点：  
   \[  
   \\mathbf{p}*{cal} \= A\[\\mathbf{p}*{raw};1\]  
   \]

这个链条比“只说 gaze regression”更完整，也比“只说几何”更贴近你系统。

---

## **A4. 一版“我能写进论文方法章节”的简化数学表达，包括符号定义**

下面这版是你可以直接往论文方法章节靠的简化写法。

### **符号定义**

* (I)：输入 RGB 图像  
* (I\_{eye})：眼部裁剪图像  
* (I\_{face})：脸部裁剪图像  
* (R,t)：头部相对于相机坐标系的旋转和平移  
* (\\hat{\\mathbf{g}}\_{local}\\in\\mathbb{R}^3)：局部坐标系下预测的 gaze unit vector  
* (\\hat{\\mathbf{g}}\_{cam}\\in\\mathbb{R}^3)：相机坐标系下的 gaze vector  
* (O)：gaze ray 的起点（如眼中心或参考点）  
* (\\Pi\_{screen}=(P\_0,N))：屏幕平面，其中 (P\_0) 为平面上一点，(N) 为法向量  
* (P\_{raw})：gaze ray 与屏幕平面的交点  
* (\\mathbf{p}\_{cal})：校准后的最终屏幕注视点

### **方法表达**

**(1) 外观到局部 gaze 向量**  
\[  
\\hat{\\mathbf{g}}*{local} \= f*\\theta(I\_{eye}, I\_{face})  
\]  
若当前实现没有 face 分支，可退化为：  
\[  
\\hat{\\mathbf{g}}*{local} \= f*\\theta(I\_{eye})  
\]

**(2) 局部坐标到相机坐标**  
\[  
\\hat{\\mathbf{g}}*{cam} \= R \\hat{\\mathbf{g}}*{local}  
\]

**(3) 与屏幕平面求交**  
\[  
P\_{raw}=O+\\tau \\hat{\\mathbf{g}}*{cam}, \\qquad*  
*\\tau=\\frac{N^\\top(P\_0-O)}{N^\\top \\hat{\\mathbf{g}}*{cam}}  
\]

**(4) 屏幕点校准**  
\[  
\\mathbf{p}*{cal}=A\[\\mathbf{p}*{raw};1\]  
\]

### **这版写法的优点**

* 吸收了这篇论文关于 **3D gaze formulation** 的优点  
* 保留了你 Look2Act 现有的 **PnP \+ ray-plane \+ calibration** 主线  
* 不会假装你已经实现了 full-face-only 或 normalization warp 那套复杂东西

---

## **A5. 如果这篇论文的方法其实不适合我当前项目，最适合借的是“叙事”还是“具体方法”？**

结论很明确：

**最适合借的是：叙事 \+ 实验设计 \+ 输入表示思路。**  
**次适合借的是：3D gaze representation 的写法。**  
**最不适合直接照搬的是：具体网络结构与 spatial weights 实现。**

更直白一点：

* 你该借的不是 “我也上 AlexNet+spatial weights”  
* 你该借的是 “为什么只看眼睛可能不够”  
* 以及 “为什么 3D gaze direction 比直接 2D screen mapping 更适合真实部署系统”

所以这篇论文对你当前项目最合适的使用方式是：

1. **作为 related work / method motivation 的关键文献**  
2. **作为 input representation ablation 的实验启发**  
3. **作为 head pose / illumination / extreme gaze robustness 的分析模板**

不是直接拿来重写你整个 Look2Act 模型。

