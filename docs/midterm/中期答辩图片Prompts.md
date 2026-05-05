## 图G-01 普通 RGB 摄像头桌面视线追踪应用场景示意图

```text
Style Reference & Execution Instructions:
1. Art Style:
Generate a top-tier scientific schematic with true paper-figure aesthetics: extremely clean 2D flat design, no photoreal rendering, no material texture, no dramatic lighting, no glossy UI, no cartoon style, no decorative illustration style. If any depth cue is needed, only use very restrained 2.5D isometric stacking for tiny sub-panels, never realistic 3D rendering.
2. Visual Language:
Use dashed bounding boxes to divide logical regions. Use thin dark-gray outlines, crisp solid arrows, precise alignment, large whitespace, and restrained pastel palette only: pale azure blue, pale mint green, pale lavender, pale warm yellow, with deep gray text and strokes.
3. Text Rules:
All visible text must be in simplified Chinese, except unavoidable technical abbreviations such as RGB, IR, and 3D. Do not render any unlabeled placeholder text. Do not render meta-words such as “zone”, “module”, “input”, “output”, “layout”, “step”, “caption”, or any English explanatory paragraph.
4. Composition Rules:
Strictly horizontal composition, left to right, with balanced margins and a journal-grade layout. Every icon and label must feel like it was drawn in TikZ or precision Illustrator rather than by a generic infographic generator.
5. Forbidden Style:
No gradients, no shadows, no lens flare, no shiny buttons, no fake dashboard UI, no skeuomorphism, no childish icon set, no hand-drawn feeling, no random floating decorations.

---BEGIN PROMPT---
[Figure Goal]
Draw a highly disciplined scientific scene figure that explains the practical usage scenario of desktop gaze tracking using only a normal RGB webcam, emphasizing low-cost deployment, unconstrained head pose, and screen-space gaze estimation.

[Global Layout]
Use a horizontal triptych composition with three major dashed bounding regions:
- Left region: user and hardware setup
- Middle region: visual sensing and gaze estimation cues
- Right region: screen mapping and interaction outcome
Add a compact comparison inset in the lower-left or lower-right corner, but keep it visually secondary.

[Region A: User and Device Setup]
Inside a pale slate-blue dashed box, draw:
- a simplified laptop or desktop monitor in frontal view
- a small webcam centered above the screen bezel
- a seated user in front view or slight three-quarter view, rendered as a clean neutral silhouette, not realistic
- the head and both eyes clearly visible
- a dashed distance indicator from webcam to user face
Render only these Chinese labels:
- “普通RGB摄像头”
- “桌面显示器”
- “用户面部与双眼”
- “自然头部姿态”
- “日常使用距离”

[Region B: Visual Sensing and Estimation]
Inside a pale mint-green dashed box, draw:
- a cropped face patch
- two eye ROI rectangles enlarged from the face
- a small head pose axis icon near the head with yaw / pitch / roll notation
- a clean gaze ray emerging from the eye center direction
Do not draw a neural network box here; this figure is about the application scene rather than internal architecture.
Render only these Chinese labels:
- “人脸区域”
- “眼部ROI”
- “头部姿态”
- “3D视线方向”

[Region C: Screen Mapping Result]
Inside a pale lavender dashed box, draw:
- a flat front-facing screen plane with a subtle coordinate cross or grid
- one highlighted gaze point on the screen
- a thin circular ring around the gaze point to suggest稳定注视
- a cursor symbol or small interaction marker near the gaze point
Render only these Chinese labels:
- “屏幕平面”
- “屏幕注视点”
- “二维屏幕坐标”
- “交互触发”

[Comparison Inset]
Create a very small secondary inset with two compact columns:
- left: low-cost RGB setup
- right: infrared eye tracker setup
The inset must remain minimal and elegant, not like a commercial comparison table.
Render only these Chinese labels:
- “普通RGB方案”
- “红外眼动仪方案”
- “低成本”
- “高成本”

[Connections]
Use solid dark-gray or muted blue arrows:
- from webcam and face toward the eye ROI
- from eye direction toward screen plane
- from screen plane toward gaze point
- from gaze point toward interaction cursor
No excessive arrow labels. If one tiny label is needed, only render:
- “几何映射”

[Typography and Spacing]
Use bold clean sans-serif Chinese text, perfectly legible, non-blurry, non-deformed. Keep every label horizontal. Use strong visual hierarchy and generous whitespace. Final result must feel like a clean systems paper figure, not a presentation infographic template.
---END PROMPT---
```

## 图G-02 开题目标到中期验证逻辑总览图

```text
Style Reference & Execution Instructions:
1. Art Style:
Generate an ultra-clean top-tier scientific logic diagram. Strictly 2D flat vector design. No realistic rendering, no glossy shading, no playful infographic style. The result should resemble carefully typeset academic figures produced with TikZ, Figma, or Illustrator by a skilled paper author.
2. Visual Language:
Heavy use of dashed bounding boxes for semantic grouping. Thin deep-gray connectors. Pastel fills only: light azure, light mint, light lavender, light warm yellow. Strict grid alignment and strong whitespace discipline.
3. Text Rules:
All visible text must be in simplified Chinese. Do not render any English sentence except unavoidable symbols if needed. Do not render meta-words such as “phase”, “box”, “container”, “layout”, “overview”, “zone”.
4. Structure Rules:
The figure must communicate progression, not decoration. Every block must have a clear semantic function and a restrained amount of text.
5. Forbidden Style:
No charts, no emoji-like icons, no thick rounded childish cards, no gradients, no 3D blocks, no fake UI windows.

---BEGIN PROMPT---
[Figure Goal]
Create a rigorous logic figure that summarizes how the project progresses from proposal-stage goals to midterm-stage evidence, and then to the remaining work. The figure should instantly communicate that the project has moved from “提出方案” to “验证路线成立”.

[Global Layout]
Use a left-to-right four-stage pipeline with a thin top title strip and a compact bottom conclusion strip.
Each stage is enclosed in a dashed pastel bounding box.
The four stages are:
1. 问题定义
2. 技术路线
3. 中期验证
4. 后续工作

[Top Title Strip]
Render only this title:
- “从开题方案到中期验证：重点在路线是否成立”

[Stage 1: 问题定义]
Inside a pale blue dashed box, place three vertically aligned nodes with simple geometric icons:
- “实时运行”
- “跨用户/跨设备泛化”
- “交互支持”
Keep the icons abstract and academic, not cute.

[Stage 2: 技术路线]
Inside a pale mint dashed box, place three vertically aligned nodes:
- “深度学习外观建模”
- “几何映射”
- “用户校准”
Show them as modular method components rather than a timeline.

[Stage 3: 中期验证]
Inside a pale lavender dashed box, place three nodes with restrained check symbols:
- “数据集已构建”
- “模型已训练评估”
- “系统原型已集成”
This stage should visually feel most stable and concrete.

[Stage 4: 后续工作]
Inside a pale warm-yellow dashed box, place three nodes with small progress markers:
- “校准映射优化”
- “长尾用户误差分析”
- “系统级测试与论文撰写”
Keep this stage clearly unfinished but organized.

[Connections]
Use solid straight arrows from stage to stage. Between stage 2 and stage 3, make the connector slightly visually emphasized to show that the route has now been verified.
Do not over-label arrows. If minimal arrow labels are included, only use:
- “提出方案”
- “形成证据”
- “继续收束”

[Bottom Conclusion Strip]
Use a very slim horizontal strip with one summary sentence:
- “中期阶段的核心结论：主路线已经走通，剩余工作集中在优化、验证与论文固化”

[Typography]
Use elegant bold sans-serif Chinese typography, dark gray text, perfect character correctness, strong baseline alignment, lots of negative space, and uncompromising visual cleanliness.
---END PROMPT---
```

## 图G-03 自建数据采集与数据处理流程图

```text
Style Reference & Execution Instructions:
1. Art Style:
Generate a research-grade pipeline figure with strict 2D flat design. No photorealism, no toy-like icons, no consumer-product infographic look. The image must feel like a methods figure from a high-quality systems or vision paper.
2. Visual Language:
Use modular dashed bounding boxes with distinct semantic grouping. Internal modules should be rectangular, consistently aligned, and minimally styled. Pastel palette only: pale blue for采集流程, pale green for预处理, pale lavender for输出结构, pale yellow for关键标签信息.
3. Text Rules:
All visible text must be simplified Chinese, except unavoidable technical abbreviations such as CSV, ROI, or ID when necessary. No English body text. No meta labels.
4. Layout Rules:
The figure should show a genuine full data lifecycle, not just a generic arrow chain. Include both online collection and offline preprocessing. Use strict spacing and visual rhythm.
5. Forbidden Style:
No 3D cylinders, no fake database clipart, no glossy cards, no gradients, no shadow-heavy icons.

---BEGIN PROMPT---
[Figure Goal]
Draw a full self-built dataset workflow for desktop gaze tracking, covering data collection, raw packaging, preprocessing, and standardized training data generation. The figure must clearly show why this dataset supports multi-user, multi-device, and multi-session learning.

[Global Layout]
Use a two-row layout with directional flow:
- Top row: online data collection
- Bottom row: offline preprocessing and dataset output
Use large dashed boxes to group the two rows.
Place a vertical bridge arrow from top-row raw data packaging to bottom-row preprocessing.

[Top Group: 在线采集流程]
Enclose in a pale blue dashed bounding box titled:
- “在线数据采集”
Inside it, arrange five horizontally aligned modules:
1. “用户注册与会话配置”
2. “相机与分辨率设置”
3. “人脸与头姿质量检查”
4. “屏幕目标点呈现”
5. “视线样本采集”
For each module, use highly simplified academic icons:
- user card / session tag
- camera and resolution panel
- face landmarks with pose axes
- screen with red target dot
- sample capture stream with repeated small frames
Add small auxiliary tags near this group:
- “多用户”
- “多设备”
- “多会话”

[Top Row Output Module]
At the far right of the top row, place one distinct module in pale yellow:
- “原始数据打包”
Inside or near it, show small stacked artifacts:
- face image
- left eye / right eye
- head pose record
- screen point label
- CSV / metadata package
Render only these small labels:
- “人脸图像”
- “左眼/右眼”
- “头部姿态”
- “屏幕注视点标签”
- “CSV”

[Bottom Group: 离线预处理流程]
Enclose in a pale mint dashed bounding box titled:
- “离线预处理”
Inside it, arrange four horizontally aligned modules:
1. “关键点检测”
2. “眼区裁剪”
3. “标签对齐”
4. “视线向量构建”
Show a clean transformation logic, not realistic images.
For “视线向量构建”, include a tiny coordinate cue and a compact unit-vector symbol.

[Final Output Group]
At the far right of the bottom row, place a pale lavender dashed output group titled:
- “标准化训练数据”
Inside it, show three neatly stacked output cards:
- “左眼样本”
- “右眼样本”
- “结构化标签”
Add three small side annotations:
- “统一尺寸”
- “统一格式”
- “可直接训练”

[Connections]
Use solid arrows between all modules.
Use one emphasized vertical arrow from “原始数据打包” down to the bottom preprocessing group.
If arrow labels are used, only allow:
- “原始采集数据”
- “预处理”
- “训练输入”

[Typography]
Chinese typography must be crisp and exact. Use deep gray text, thin separators, large whitespace, mathematically clean alignment, and a premium paper-figure feel.
---END PROMPT---
```

## 图G-04 GazeNetV2 模型结构与信息融合示意图

```text
Style Reference & Execution Instructions:
1. Art Style:
Generate a top-tier neural network architecture figure with a true academic paper aesthetic. Strictly 2D flat design with optional 2.5D isometric stacking only for feature tensors. No photoreal textures, no glowing neural network art, no AI-poster style.
2. Visual Language:
Use dashed bounding boxes to separate “输入”, “共享特征提取”, “特征融合”, and “回归输出”. Tensor blocks can be shown as thin stacked pastel plates in restrained isometric style. Use deep-gray lines, pastel fills, and precise solid arrows.
3. Text Rules:
All visible text must be simplified Chinese, except the following allowed technical terms when needed: CNN, FC, Dropout, 3D, L2 normalization, yaw, pitch, roll. Do not render any other English sentences.
4. Structural Fidelity:
The architecture must reflect the actual project design: left eye and right eye go through the same shared backbone; each branch outputs 256-d features; concatenation with 3-d head pose gives 515-d fused representation; then FC 515→128, ReLU + Dropout(0.3), FC 128→3, then L2 normalization.
5. Forbidden Style:
No decorative neuron webs, no meaningless matrices everywhere, no random floating tensors, no gradients, no shadows, no neon.

---BEGIN PROMPT---
[Figure Goal]
Draw a rigorous architecture diagram for GazeNetV2, clearly showing binocular shared-feature extraction and head-pose fusion for 3D gaze regression.

[Global Layout]
Use a left-to-right architecture layout with four dashed macro-groups:
1. 输入
2. 共享CNN特征提取
3. 特征融合与回归
4. 输出

[Group 1: 输入]
Inside a pale blue dashed box titled:
- “输入”
Arrange three aligned input nodes:
- top: “左眼图像 128×128”
- middle: “右眼图像 128×128”
- bottom: “头部姿态 yaw / pitch / roll”
For the two eye inputs, use two small square image placeholders with eye silhouettes.
For head pose, use a compact numeric vector card.

[Group 2: 共享CNN特征提取]
Inside a pale mint dashed box titled:
- “共享CNN特征提取”
Draw two parallel branches for left and right eye, both entering the same visually identical backbone block series.
The backbone must be shown as four repeated convolutional stages, each stage minimal and aligned:
- Conv-BN-ReLU-Pool
- Conv-BN-ReLU-Pool
- Conv-BN-ReLU-Pool
- Conv-BN-ReLU-Pool
Visually indicate weight sharing very clearly with one shared backbone label:
- “共享权重”
At the end of each branch, show one compact feature vector block labeled:
- “256维特征”
- “256维特征”
If using tensor depictions, use restrained isometric stacked plates only.

[Group 3: 特征融合与回归]
Inside a pale lavender dashed box titled:
- “特征融合与回归”
Show a concatenation node receiving:
- 左眼 256维特征
- 右眼 256维特征
- 头部姿态 3维
Then explicitly show these sequential blocks:
- “拼接后 515维”
- “FC 515→128”
- “ReLU + Dropout(0.3)”
- “FC 128→3”
Use a precise plus / concat logic symbol, not vague merging clouds.

[Group 4: 输出]
Inside a pale warm-yellow dashed box titled:
- “输出”
Show one final vector block labeled:
- “3D视线方向向量”
Below or beside it show:
- “L2 normalization”

[Side Annotation]
Add a slim annotation strip with only these three Chinese points:
- “双眼信息提升稳定性”
- “头姿信息提供几何上下文”
- “3D输出服务后续屏幕映射”

[Connections]
Use only solid straight arrows. Make the concat connection mathematically clear. No decorative flow arrows.

[Typography]
All Chinese characters must be accurate and well spaced. The whole composition must feel like a precise model figure from a serious paper, with strong hierarchy, compact logic, and no visual noise.
---END PROMPT---
```

## 图G-05 实时视线追踪系统端到端流程架构图

```text
Style Reference & Execution Instructions:
1. Art Style:
Generate an elite systems-architecture figure in strict 2D flat academic style. No interface mockup aesthetic, no product marketing, no material realism. Use only precise blocks, arrows, dashed semantic containers, and restrained 2.5D only where a stacked tensor or coordinate slab is truly necessary.
2. Visual Language:
Use large dashed bounding regions to separate “感知层”, “估计层”, “几何层”, “输出层”, “交互层”. Use pastel palette with strong restraint: pale blue, pale mint, pale lavender, pale yellow, plus deep gray strokes and text.
3. Text Rules:
All visible text must be simplified Chinese, except allowed technical abbreviations and symbols such as CPU, ONNX, 3D, EMA, and FPS if needed. Do not render extra English sentences. Do not render meta labels like “pipeline”, “module”, or “layer”.
4. Structural Fidelity:
The architecture must reflect the actual project pipeline: camera frame → face detection → eye region extraction → head pose estimation → gaze regression → coordinate transform → ray-plane intersection / screen geometry → smoothing → calibrated gaze point → interaction outputs.
5. Forbidden Style:
No software dashboard look, no glossy app windows, no random monitor screenshots, no gradients, no shiny icons, no childish color blocks.

---BEGIN PROMPT---
[Figure Goal]
Draw a complete end-to-end system architecture for real-time desktop gaze tracking, making the sensing-to-interaction chain explicit and technically clean.

[Global Layout]
Use a left-to-right architecture with five dashed macro-groups:
1. 感知层
2. 估计层
3. 几何层
4. 输出层
5. 交互层
Keep all groups horizontally aligned with strict spacing.

[Group 1: 感知层]
Inside a pale blue dashed box titled:
- “感知层”
Place three modules:
- “摄像头输入”
- “人脸检测”
- “眼区提取”
Show a small frame icon entering a face landmark panel, then two eye crops.
Add one tiny side tag:
- “视频帧”

[Group 2: 估计层]
Inside a pale mint dashed box titled:
- “估计层”
Place two vertically aligned modules:
- “头部姿态估计”
- “视线回归模型”
Head pose module outputs a compact pose vector:
- “yaw / pitch / roll”
Gaze model outputs:
- “3D视线向量”

[Group 3: 几何层]
Inside a pale lavender dashed box titled:
- “几何层”
Place three modules:
- “坐标变换”
- “射线-平面求交”
- “屏幕几何映射”
In the center of this group, include a tiny clean coordinate schematic with:
- eye origin point
- gaze ray
- screen plane
- intersection point
This should be schematic, not artistic.

[Group 4: 输出层]
Inside a pale yellow dashed box titled:
- “输出层”
Place two modules:
- “时序平滑”
- “校准后注视点输出”
Near smoothing, allow one compact label:
- “EMA”
Near final output, allow:
- “屏幕坐标”

[Group 5: 交互层]
Inside a pale blue-gray dashed box titled:
- “交互层”
Place three compact interaction outcomes:
- “注视光标”
- “停留点击”
- “交互面板”
Use extremely minimal icons.

[Deployment Strip]
Add a narrow auxiliary strip above or below the full pipeline with only two small deployment notes:
- “CPU实时运行”
- “ONNX部署”

[Connections]
Use straight solid arrows through the entire chain. Keep arrow labels minimal; only use these if needed:
- “关键点”
- “左右眼裁剪”
- “姿态”
- “3D视线向量”
- “屏幕坐标”

[Typography]
Typography must be exact, mature, and non-deformed. The whole figure must look as if carefully hand-authored for a serious systems paper: sparse, modular, logical, and elegant.
---END PROMPT---
```

## 图G-06 后续工作路线图（从“证明可行”到“证明可用”）

```text
Style Reference & Execution Instructions:
1. Art Style:
Generate a high-end academic roadmap figure in strict 2D flat design. No project-management dashboard style, no corporate slide style, no glossy milestones. The visual tone should be calm, precise, and publication-grade.
2. Visual Language:
Use dashed bounding boxes for workstreams, thin solid arrows for progression, restrained pastel palette, deep-gray typography, and very consistent alignment. Use compact milestone nodes, not oversized decorative icons.
3. Text Rules:
All visible text must be simplified Chinese, except abbreviations such as FPS if absolutely needed. No English explanatory text. No meta labels.
4. Content Rules:
The roadmap must show that the core route is already validated, and future work focuses on refinement, tail-case analysis, system verification, and paper consolidation. It must not look like a generic startup roadmap.
5. Forbidden Style:
No gantt chart template, no sticky-note style cards, no gradients, no shadows, no hand-drawn arrows, no office-suite smart-art look.

---BEGIN PROMPT---
[Figure Goal]
Draw a refined future-work roadmap showing how the project moves from “主路线已经走通” to “系统可用性与论文结论完整收束”.

[Global Layout]
Use three parallel workstreams in the middle, all converging to a final deliverable block on the right. Add a slim top title strip and a slim bottom conclusion strip.
Each workstream is enclosed in its own pastel dashed box.

[Top Title Strip]
Render only:
- “从证明路线可行到证明系统可用”

[Workstream 1]
Pale blue dashed box titled:
- “校准与映射优化”
Inside it place three ordered nodes:
- “校准点策略比较”
- “映射函数对比”
- “在线校准场景验证”

[Workstream 2]
Pale mint dashed box titled:
- “长尾用户误差分析”
Inside it place four ordered nodes:
- “设备差异分析”
- “光照与眼镜条件分析”
- “长尾样本检查”
- “针对性补充采样”

[Workstream 3]
Pale lavender dashed box titled:
- “系统级验证”
Inside it place four ordered nodes:
- “FPS测试”
- “模块耗时分析”
- “稳定性测试”
- “校准前后可用性比较”

[Convergence Block]
On the far right, place one more visually consolidated dashed block in pale warm yellow titled:
- “最终成果”
Inside it place three vertically stacked deliverables:
- “系统原型”
- “完整实验章节”
- “毕业论文”

[Flow Logic]
Each workstream should have a clear left-to-right progression arrow within itself.
Then all three workstreams converge with solid arrows into the final deliverable block.
Near the convergence area, allow one compact label:
- “结果固化”

[Bottom Conclusion Strip]
Render only:
- “后续重点不再是验证主路线是否成立，而是优化系统误差、补足系统级证据并完成论文收束”

[Typography]
Use high-quality Chinese sans-serif text, perfectly crisp, evenly spaced, dark gray. The full composition must feel like a carefully authored roadmap figure from a polished academic paper rather than any generic template.
---END PROMPT---
```
