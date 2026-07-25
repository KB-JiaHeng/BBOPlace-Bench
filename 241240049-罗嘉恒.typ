#set page(
  paper: "a4",
  margin: (top: 1.8cm, bottom: 1.8cm, left: 2.0cm, right: 2.0cm),
  numbering: "1",
)
#set text(font: ("Times New Roman", "Noto Serif CJK SC"), size: 10.5pt)
#set par(leading: 0.58em, justify: true)
#set heading(numbering: "1.1")
#set table(stroke: 0.42pt, inset: 3.2pt)
#set figure(gap: 0.55em)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK SC"), size: 8pt)
#show link: it => underline(text(fill: rgb("#1f5ca9"))[#it])
#show ref: it => text(fill: rgb("#1f5ca9"))[#it]

#let note(body) = block(
  fill: luma(245),
  stroke: (left: 2.2pt + rgb("#4c78a8")),
  inset: (left: 9pt, right: 8pt, top: 6pt, bottom: 6pt),
  radius: 2pt,
  width: 100%,
  body,
)

#let warning(body) = block(
  fill: rgb("#fff8e8"),
  stroke: (left: 2.2pt + rgb("#d28b00")),
  inset: (left: 9pt, right: 8pt, top: 6pt, bottom: 6pt),
  radius: 2pt,
  width: 100%,
  body,
)

#align(center)[
  #v(1.6cm)
  #text(size: 21pt, weight: "bold")[第五次作业实验报告]
  #v(0.45cm)
  #text(size: 14pt)[BBOPlace-Bench 算子实验与分析]
  #v(1.2cm)
  #text(size: 12pt)[PS: 我对框架代码提了两份PR]

  #text(size: 12pt)[具体请看#link(<Bug>)[这个部分] #emoji.ballot]
  #v(5.1cm)
  #line(length: 10.5cm)
  #v(0.45cm)
  #text(size: 13pt)[南京大学匡亚明学院]
  #v(0.3cm)
  #text(size: 13pt)[罗嘉恒]
  #v(0.3cm)
  #text(size: 13pt)[241240049]
  #v(0.45cm)
  #line(length: 10.5cm)
  #v(4.8cm)
  2026 年 7 月 25 日
]

#pagebreak()
#outline(title: [目录], depth: 3)

#pagebreak()
= Task 1：交叉与变异算子研究

== 实验配置

Task 1 的目标是在固定的 MGO 表示、单目标 MP-HPWL 评价和严格相同的评价预算下，比较不同交叉算子与变异算子的搜索行为。实验采用两个交叉算子（Uniform、SBX）和五个变异算子（Swap、Shift、Random Resetting、Shuffle、Polynomial Mutation），形成 $2 times 5$ 的完整析因设计。


#table(
  columns: (1.25fr, 2.75fr),
  table.header([*项目*], [*固定设置*]),
  [Benchmark], [ISPD 2005 adaptec1, 512 macros(非全维度)],
  [Placement representation], [MGO，$224 times 224$ grid，`rank_key = area_sum`],
  [Population], [20；每代最多产生 20 个 offspring],
  [Initialization], [每个 seed 随机产生 20 个 genotype，也就是`n_sampling_repeat = 1`],
  [Budget], [每个 run 严格 10,000 次真实 objective evaluation],
  [Selection / survival], [Binary tournament；$(mu + lambda)$ elitist fitness survival],
  [Crossover], [Uniform/SBX；总体概率均为 1.0],
  [Mutation], [Swap/Shift/Random Resetting/Shuffle/PM],
  [Randomness], [seeds 1-5；同一 seed 在所有配置中使用相同初始 population],
  [Duplicate handling], [删除重复 genotype],
  [Replicates（第一类实验）], [Crossover × Mutation $times$ 5 seeds = 50 runs],

  [Sensitivity（第二类实验）], [SBX $eta in {5,15,30}$ 与 PM $eta in {5,15,30}$，共 20 runs],
  [Total], [70 runs，700,000 次真实 objective evaluation],
)

其中：
- Uniform crossover 的变量交换概率为 0.5
- SBX 使用 `prob_var = 0.5`、`prob_exch = 1.0`、`prob_bin = 0.5`
- $eta = 30$。
- PM 的个体级概率和变量级概率均为 1.0

== 实验结果

=== 最终性能

#note(
  [*备注*: 主文中的标准差采用原仓库结果汇总代码 `np.std` 的口径，即 population standard deviation（`ddof = 0`）。`ddof = 1` 的 sample standard deviation 单独保留在审计 CSV 中。对 5 个 seeds，后者固定比前者大 $sqrt(5 / 4) approx 1.118$ 倍；统一口径后，各配置的 coefficient of variation 为 0.50%-1.75%，与论文中的Vanilla EA的标准差是相近的。
  ],
)

表 @tab-final 给出 evaluation 10,000 时的最终 best-so-far HPWL。数值均以 $10^5$ 为单位，标准差为 `ddof = 0`。


#figure(
  text(
    size: 7.6pt,
    table(
      columns: (0.45fr, 1.85fr, 1.05fr, 0.85fr, 0.85fr, 0.85fr),
      align: (center, left, center, center, center, center),
      [*Rank*], [*Configuration*], [*Mean $plus.minus$ Std*], [*Median*], [*Best*], [*Worst*],
      [1], [Uniform + Swap (Baldwinian)], [*5.7671 $plus.minus$ 0.0592*], [5.7819], [5.6759], [5.8275],
      [2], [Uniform + Swap (Lamarckian)], [(BLANCK)], [(BLANCK)], [(BLANCK)], [(BLANCK)],
      [3], [Uniform + Random Resetting], [5.7809 $plus.minus$ 0.0595], [5.7856], [5.6756], [5.8603],
      [4], [SBX + Shuffle], [5.7845 $plus.minus$ 0.0565], [5.7715], [5.6984], [5.8689],
      [5], [Uniform + Shuffle], [5.7864 $plus.minus$ 0.0457], [5.7826], [5.7376], [5.8700],
      [6], [SBX + PM], [5.8059 $plus.minus$ 0.0390], [5.7863], [5.7724], [5.8794],
      [7], [Uniform + PM], [5.8139 $plus.minus$ 0.0291], [5.8037], [5.7874], [5.8685],
      [8], [SBX + Swap], [5.8301 $plus.minus$ 0.0845], [5.8020], [5.7412], [5.9773],
      [9], [SBX + Random Resetting], [5.8689 $plus.minus$ 0.1027], [5.8190], [5.7651], [6.0520],
      [10], [Uniform + Shift], [5.9753 $plus.minus$ 0.0626], [5.9615], [5.9075], [6.0649],
      [11], [SBX + Shift], [6.0091 $plus.minus$ 0.0466], [6.0101], [5.9370], [6.0693],
    ),
  ),
  caption: [十种算子组合的最终 HPWL（$times 10^5$，5 seeds，population std）。],
) <tab-final>

#figure(
  image("assets/task1/final_hpwl_heatmap_2_by_5.png", width: 100%),
  caption: [两种 crossover 与五种 mutation 的最终 HPWL。\ 单元格为 mean $plus.minus$ population std，越低越好。],
) <fig-heatmap>


#figure(
  image("assets/task1/final_hpwl_distribution.png", width: 96%),
  caption: [五个 seeds 的最终 HPWL 分布。\ 菱形为均值，横线为中位数，散点为单个 seed。],
) <fig-distribution>

图 @fig-distribution 进一步说明均值排名对 seed 有一定敏感性。单次运行的全局最低值来自 Uniform + Random Resetting、seed 3，HPWL 为 5.6756 $times 10^5$；它只比同 seed 的 Uniform + Swap 低约 31.4，因此不能用单次最好结果替代跨 seed 平均。按每个 seed 内的 rank 再取平均，SBX + Shuffle 为 3.20，Uniform + Swap 为 3.60，Uniform + Shuffle 和 Uniform + Random Resetting 均为 3.80。

十种配置的 Friedman test 得到 $chi^2 = 23.967$、$p = 0.004353$，说明整体配置效应可检测；但仅有 5 个 paired seeds，前四名之间的精确配对检验均没有提供强证据。两个 Shift 组合相对 Uniform + Swap 在 5/5 seeds 上均更差，但双侧 exact sign-flip test 在 $n = 5$ 时最小只能达到 $p = 0.0625$，因此报告中同时给出方向一致性与样本量限制。

相对改进率定义为
$ ("best_at_20" - "best_at_10000") / "best_at_20" × 100% $

表 @tab-improvement 给出每个 seed 的配对结果。
#figure(
  text(
    size: 6.6pt,
    table(
      columns: (1.75fr, 0.62fr, 0.62fr, 0.62fr, 0.62fr, 0.62fr, 0.72fr),
      align: (left, center, center, center, center, center, center),
      [*Configuration*], [*S1*], [*S2*], [*S3*], [*S4*], [*S5*], [*Mean*],
      [Uniform + Swap], [13.640], [14.357], [13.285], [9.936], [11.652], [*12.574*],
      [Uniform + Random Resetting], [12.710], [13.874], [13.290], [10.370], [11.600], [12.369],
      [SBX + Shuffle], [12.923], [14.525], [12.941], [10.839], [10.322], [12.310],
      [Uniform + Shuffle], [12.677], [15.678], [12.097], [10.607], [10.305], [12.273],
      [SBX + PM], [12.793], [13.593], [11.598], [10.169], [11.796], [11.990],
      [Uniform + PM], [12.240], [13.753], [11.333], [10.533], [11.483], [11.868],
      [SBX + Swap], [11.518], [15.625], [11.915], [10.308], [8.666], [11.606],
      [SBX + Random Resetting], [12.207], [14.737], [7.540], [10.878], [9.745], [11.021],
      [Uniform + Shift], [9.032], [12.387], [9.661], [8.677], [7.327], [9.417],
      [SBX + Shift], [9.734], [10.804], [7.629], [8.220], [8.164], [8.910],
    ),
  ),
  caption: [从 initial population best 到 evaluation 10,000 best 的逐 seed 相对改进率（%）。],
) <tab-improvement>


#note([
  *与原论文结果的对照*: 原 #link("https://arxiv.org/abs/2510.23472")[BBOPlace-Bench] 的 MGO Vanilla-EA 配置对应 Uniform + Shuffle，并使用 population 50、`n_sampling_repeat = 5`。其 adaptec1 报告值约为 5.80 $plus.minus$ 0.03 $times 10^5$。本实验冻结协议下的对应组合为 5.7864 $plus.minus$ 0.0457 $times 10^5$：平均值接近，但标准差略高。该差异不能直接归因于实现错误，因为本实验主动将 population 降为 20，并取消了从 $50 times 5$ 个初始候选中筛选最优 50 个的重复采样过程；两项改变都会增加早期搜索路径对 seed 的敏感性。所以可以认为我们复现了论文对VanillaEA的结果。
])


=== 收敛过程

所有收敛图均以真实 objective evaluation 为横轴，每 20 次评价取一个已有 best-so-far 数据点。实线为 5-seed mean，阴影为 population std（`ddof = 0`）。

#figure(
  image("assets/task1/uniform_mutation_convergence.png", width: 100%),
  caption: [Uniform crossover 下五种 mutation 的收敛曲线。 ],
) <fig-uniform-convergence>

#figure(
  image("assets/task1/sbx_mutation_convergence.png", width: 100%),
  caption: [SBX crossover 下五种 mutation 的收敛曲线。],
) <fig-sbx-convergence>

Uniform + Swap 在 evaluation 1,000、5,000 和 10,000 的平均 HPWL 均为第一。其平均改进可分解为：evaluation 20-1,000 改进 70,718，1,000-5,000 改进 10,222，5,000-10,000 仅再改进 2,177；约在 evaluation 1,940 达到其最终总改进的 90%。这说明它的优势主要来自早期快速找到高质量结构，而非依赖很长的后期搜索。

Uniform + Random Resetting 的早期表现较弱，但 evaluation 5,000 后仍改进约 6,798，明显高于 Uniform + Swap 的 2,177。该现象与 Random Resetting 能持续引入全新 guide coordinates 的机制一致，但其跨 seed 结果也更不稳定。Shift 很早达到“自身最终改进”的 90%，却停在明显更差的平台；因此它的短 time-to-90% 不是高效收敛，而是过早停滞。

=== 分布指数敏感性

#figure(
  image("assets/task1/sbx_eta_sensitivity.png", width: 100%),
  caption: [SBX + Swap 的分布指数 $eta$ 敏感性。],
) <fig-sbx-eta>

SBX + Swap 的最终结果为：$eta = 5$ 时 5.8027 $plus.minus$ 0.0491，$eta = 15$ 时 5.8111 $plus.minus$ 0.0723，$eta = 30$ 时 5.8301 $plus.minus$ 0.0845。较小 $eta$ 的平均值更好，但是并不是非常的敏感。

#figure(
  image("assets/task1/pm_eta_sensitivity.png", width: 100%),
  caption: [Uniform + PM 的分布指数 $eta$ 敏感性。],
) <fig-pm-eta>

Uniform + PM 的结果为：$eta = 5$ 时 5.8769 $plus.minus$ 0.0294，$eta = 15$ 时 5.8140 $plus.minus$ 0.0277，$eta = 30$ 时 5.8139 $plus.minus$ 0.0291。$eta = 15$ 与 30 基本相同，而 $eta = 5$ 在 5/5 seeds 上都更差。由于本协议设置 `pm_prob_var = 1.0`，PM 会尝试扰动全部标量坐标；较小 $eta$ 进一步扩大扰动尺度，因而更容易破坏已有的优良结构。这一机制解释与数据高度一致。


== 实验分析

=== Crossover-mutation interaction

#figure(
  image("assets/task1/crossover_mutation_interaction.png", width: 96%),
  caption: [evaluation 10,000 时的 crossover-mutation interaction。误差条为 population std。],
) <fig-interaction>

跨两个 crossover 汇总后，mutation 的边际均值从好到差为 Shuffle、Swap、PM、Random Resetting、Shift；但该顺序不能完整表达交互关系。Uniform 的总体边际均值低于 SBX，然而 SBX 与 Shuffle、PM 搭配时略好，Uniform 与 Swap、Random Resetting、Shift 搭配时平均更好。图 @fig-interaction 中两条折线并不平行，这正是交互效应的直接表现。

=== 为什么 Uniform + Swap 会表现良好

我在实验前有个预期，认为Swap 不产生新坐标，因此探索不足，应该不会产生很好的结果。结果与我的判断相反。对此我的理解是，初始种群虽然只包含有限数量的 genotype，但可能已经提供了足够丰富的 guide-coordinate building blocks，并覆盖了多个有潜力的 MGO decoding basins。Uniform crossover 通过重新组合坐标分量，Swap mutation 通过重新分配宏与坐标对之间的对应关系，在不持续引入高破坏性新坐标的情况下形成了较大的组合搜索空间。因此，Uniform + Swap 能够更快地利用已有优质区域。相比之下，持续生成新坐标的算子虽然维持了更广的探索，但也会将更多评价预算用于进入新的或低质量的 decoding basins，从而减慢收敛并增加结果波动。

还有如下一些结论：

+ 在冻结的 adaptec1/MGO 协议下，Uniform + Swap 的平均最终 HPWL 与 mean convergence AUC 最好，可作为后续实验的 reference configuration。
+ Uniform + Random Resetting、SBX + Shuffle 和 Uniform + Shuffle 与其差距较小；前四名应视为统计上尚未解析的竞争组，而非确定的严格排序。
+ Shift 是最清晰的负面结果：无论与 Uniform 还是 SBX 搭配，都较早停在明显更差的平台。
+ Crossover 与 mutation 存在显著交互，不能脱离 mutation 单独宣称某一种 crossover 普遍更优。
+ “是否产生新坐标值”不足以预测 MGO 上的算子表现；guide-coordinate genotype 与 greedy legal phenotype 之间的非线性映射是解释结果的关键。
+ PM 在全变量变异设置下不适合过小的 $eta$；SBX 的 $eta$ 敏感性则需要更多 seeds 才能形成强结论。

#pagebreak()
= Task 2：多目标优化

Task 1 在统一的 MGO 表示与评价预算下比较了不同交叉、变异算子组合，并将 Uniform + Swap 选为后续实验的 reference configuration。它在最终 HPWL 和收敛速度上表现较好，但现有结果仍不足以解释这种优势从何而来。MGO 的 genotype 只表示宏的 guide coordinates，greedy decoder 才决定最终 placement；不同 genotype 可能落入同一个 decoding basin，也可能经过幅度相近的扰动后产生完全不同的布局。Shift 较早停在较差平台的现象，也提示 genotype 层面的变化强度无法直接代表 phenotype 层面的有效探索。

此外，Task 1 的单目标评价只按照 HPWL 选择个体，因此无法观察布局在连线长度与局部布线热点之间的权衡。若直接进入 Task 3 设计新的算子或 decoder-aware 机制，我们仍缺少几类关键证据：Uniform + Swap 实际移动了多少宏、多少 offspring 被 decoder 映射回相同 phenotype、不同选择机制是否会导致 population diversity 或 MOEA/D weight slots 的集中，以及多目标搜索是否覆盖了 Task 1 未访问的低拥塞区域。

Task 2 延续 Task 1 的 top-512 macro 子问题、MGO 表示、Baldwinian 继承、初始化、population size、Uniform + Swap 和 10,000 次真实评价预算，并在每个 seed 内为 NSGA-II 与 MOEA/D 使用同一初始 population。实验将正式目标扩展为 HPWL 与 `congestion_top10`，同时记录 phenotype hash、父子关系、实际移动宏数、placement 位移、目标变化、population diversity 和 MOEA/D slot occupancy。这样的控制设计使两种多目标搜索机制成为主要差异，也为 Task 3 选择干预对象提供直接证据。

#note([
  *本 Task 主要回答三个问题：*

  1. 在相同初始化、表示、decoder、variation 和真实评价预算下，NSGA-II 与 MOEA/D 的 Pareto-front approximation 是否存在稳定差异？
  2. `congestion_top10` 是否在高质量布局区域内提供区别于 HPWL 的有效权衡信号，多目标搜索是否扩展了现有 Task 1 参考解集？
  3. 两种算法在 phenotype diversity、搜索冗余和 collapse 上有何差异，Uniform + Swap 的 genotype 扰动又有多少能够转化为有效的 placement 变化？
])

== 实验配置

Task 2 使用与 Task 1 相同的 top-512 macro 子问题。512 个节点由上游 benchmark reader 按面积降序稳定排序后截取；由于截断边界存在面积并列，实验同时冻结了原始 `.nodes` 顺序、节点名称、几何尺寸和 net topology 的 hash。当前 PlaceDB 含 512 个 macro 与 693 条 macro-level nets，所有正式 run 在评价开始前验证实际构造出的 PlaceDB 与协议一致。

正式目标为

$ F(X) = ("HPWL"(X), "Congestion"_("top10")(X)). $

其中 HPWL 沿用 Task 1 的 `cpu_comp_res` evaluator。第二目标调用未修改的 DREAMPlace RUDY kernel，在 $512 times 512$ routing bins 上计算水平与垂直 demand，分别除以固定 unit capacity 1.5625 与 1.45，再取两个方向 utilization 的较大值；`congestion_top10` 是 utilization 最高的 10% bins 的算术均值。RUDY demand、total overflow、maximum overflow 与 overflow fraction 作为 auxiliary metrics 保存，不进入正式 objective vector。

#warning([
  `congestion_top10` 描述的是当前 macro-level netlist 上的 routing-utilization proxy。它没有执行 global routing，也没有放置 standard cells，因此本文不会将其解释为完整芯片的真实拥塞或 routability。
])

#table(
  columns: (1.25fr, 2.75fr),
  table.header([*项目*], [*固定设置*]),
  [Benchmark], [`adaptec1_upstream_native_top512_macro_subset`；512 macros，693 nets],
  [Representation / decoder], [MGO，$224 times 224$ grid，`rank_key = area_sum`；greedy decoder；Baldwinian],
  [Formal objectives], [MP-HPWL 与 `congestion_top10`],
  [Population], [20；初始 population 20；每个 seed 内两种算法共享相同生成顺序],
  [Budget], [每个 run 严格 10,000 次真实 objective evaluation],
  [Variation], [Uniform crossover + Swap mutation；概率与 Task 1 完全相同],
  [NSGA-II], [rank/crowding tournament；nondominated sorting + crowding survival],
  [MOEA/D], [20 个均匀双目标 weight vectors；neighborhood size 10；normalized weighted Tchebycheff；对所有改善的邻居执行 replacement],
  [Duplicate handling], [对完整已评价历史进行 genotype hash 排重；不删除 phenotype duplicate],
  [Invalid decode], [消耗评价预算；两个目标均返回 $10^16$；不 repair、不 resample],
  [Randomness], [seeds 1-5；paired initialization；每个 run 使用独立随机状态与临时目录],
  [Formal comparison], [2 algorithms $times$ 5 seeds = 10 runs，共 100,000 次真实评价],
  [Backend], [CPU；每 run 20 logical CPUs；RUDY 1 thread；最多 2 个并发 runs],
)

Hypervolume 以最终 internal population 为 primary set，以累计 nondominated archive 为 secondary set。两个目标先使用固定 metric-audit bank 的上下界归一化：

$ z_j = (f_j - l_j) / (u_j - l_j), $

其中 $l=(543427.75, 0.0087500)$、$u=(1356135.19, 0.0261809)$，两端均包含 5% calibration margin。主 reference point 为 $(1.1, 1.1)$，并在 $(1.05,1.05)$ 与 $(1.2,1.2)$ 上检查 ordering sensitivity。正式结果没有参与 normalization bounds 的估计。

== 实验结果

=== 最终 Pareto 前沿质量

#note([
  与 Task 1 相同，本节表格和曲线中的标准差采用 population standard deviation（`ddof = 0`）。算法差异另外报告五个 paired seeds 的逐对差值、bootstrap confidence interval 和 exact Wilcoxon test。
])

#figure(
  text(
    size: 7.4pt,
    table(
      columns: (1.05fr, 1.05fr, 1.05fr, 1.1fr, 1.1fr, 0.9fr, 0.72fr),
      align: (left, center, center, center, center, center, center),
      [*Method*], [*Final HV*], [*Archive HV*], [*Final ND phenotypes*], [*Archive ND phenotypes*], [*Valid eval/s*], [*Decode failures*],
      [NSGA-II], [1.1871 $plus.minus$ 0.0271], [1.1871 $plus.minus$ 0.0271], [19.4 $plus.minus$ 0.5], [39.2 $plus.minus$ 6.4], [8.34 $plus.minus$ 0.94], [32],
      [MOEA/D], [1.1983 $plus.minus$ 0.0261], [1.1998 $plus.minus$ 0.0263], [10.2 $plus.minus$ 2.9], [21.4 $plus.minus$ 11.1], [0.739 $plus.minus$ 0.067], [2],
    ),
  ),
  caption: [Task 2 最终结果汇总（5 seeds，population std）。ND phenotype 数在 phenotype deduplication 后计算。],
) <tab-task2-final>

MOEA/D 的平均 final-population HV 高出 0.01119，约为 NSGA-II 均值的 0.94%。图 @fig-task2-paired-hv 保留了 paired design：MOEA/D 在 seeds 1、2、4 上较高，NSGA-II 在 seeds 3、5 上较高。五个差值（NSGA-II $-$ MOEA/D）为 $-0.0454,-0.0577,0.0537,-0.0339,0.0274$，方向并不一致。

#figure(
  image("assets/task2/paired_final_hypervolume.png", width: 96%),
  caption: [五个 paired seeds 的最终 population hypervolume。每条灰线连接同一初始 population 下的 NSGA-II 与 MOEA/D。],
) <fig-task2-paired-hv>

平均 paired difference 的 bootstrap 95% confidence interval 为 $[-0.0480,0.0286]$，sample-standardized effect 为 $-0.229$，双侧 exact Wilcoxon test 得到 $p=0.625$。五个 pairs 的检验分辨率很低，$p>0.05$ 也不表示两种算法等价。当前证据支持“MOEA/D 平均值略高”，无法建立稳定的算法优势。三个 reference points 下的平均 ordering 都是 MOEA/D 略高，因此这一均值方向不依赖主 reference point 的单一选择。

=== Hypervolume 收敛过程

图 @fig-task2-hv-convergence 使用 evaluation 20、1,000、5,000 与 10,000 的真实 population snapshots。两种算法共享相同初始 population，因此 evaluation 20 的平均 HV 均为 0.9879。NSGA-II 在 evaluation 1,000 时为 1.1468，高于 MOEA/D 的 1.1382；到 evaluation 5,000 时两者分别为 1.1771 与 1.1762，差距已经很小；最后一个 checkpoint 中 MOEA/D 达到 1.1983，NSGA-II 为 1.1871。

#figure(
  image("assets/task2/hypervolume_convergence.png", width: 100%),
  caption: [两种算法的 population hypervolume 收敛曲线。实线为 5-seed mean，阴影为 population std；normalization 与 reference point 对所有 runs 固定。],
) <fig-task2-hv-convergence>

四个 checkpoints 显示 NSGA-II 的早期 front approximation 更快，MOEA/D 的平均优势出现在后半程。由于中间只保存了两个 checkpoints，图中不能精确确定 crossing evaluation。NSGA-II 的 archive HV 与 final-population HV 几乎相同；MOEA/D 的 archive 平均增加约 0.00153，说明其历史搜索中保存了少量未留在最终 weight slots 中的优质 trade-off points。

=== 最终种群与 phenotype diversity

表 @tab-task2-final 中两种算法最明显的差异来自最终解集组成。NSGA-II 的 20 个 population slots 平均保留 19.4 个 nondominated distinct phenotypes，MOEA/D 只有 10.2 个。累计 10,000 次评价中，NSGA-II 每个 run 平均访问 9,817.6 个 unique valid phenotypes，MOEA/D 为 9,181.2 个；对应的累计 phenotype redundancy 分别为 1.76% 与 8.18%。两种算法均没有重复评价 exact genotype，因此该差异来自 decoder 映射与搜索状态的集中。

#figure(
  image("assets/task2/phenotype_diversity_over_time.png", width: 100%),
  caption: [搜索过程中的 population phenotype diversity。上图为 unique phenotypes，下图为占比最高的单一 phenotype；实线为 5-seed mean，阴影为 population std。],
) <fig-task2-diversity>

初始 population 平均有 19.6 个 unique phenotypes。一个 generation/sweep 后，NSGA-II 仍为 19.8，MOEA/D 降至 3.8；evaluation 1,000 时分别为 19.6 与 5.8，evaluation 10,000 时分别为 19.4 与 10.6。MOEA/D 后期能够恢复部分多样性，但最终仍明显低于 NSGA-II。

== 实验分析

=== 第二目标是否提供有效的权衡信号

在解释 Pareto front 前，先用独立于正式结果的 layout bank 审计 `congestion_top10`。该 bank 包含可用的 Task 1 initial layouts、mixed saved-best placements 与额外随机初始布局。135 个 layouts 中 133 个成功解码并对应 133 个 distinct phenotypes。

#figure(
  text(
    size: 7.6pt,
    table(
      columns: (2.25fr, 1.25fr),
      align: (left, center),
      [*Audit item*], [*Result*],
      [Valid / distinct layouts], [133 / 133],
      [Phenotype-deduplicated nondominated layouts], [2],
      [HPWL-Congestion Spearman / Kendall], [0.6991 / 0.5157],
      [High-quality layouts（lowest-HPWL quartile）], [34],
      [High-quality congestion relative range], [29.07%],
      [$256 times 256$ vs. $512 times 512$ ranking Spearman], [0.99943],
      [`n7807` only-net maximum fraction], [4.88%],
    ),
  ),
  caption: [`congestion_top10` 的固定 layout-bank 有效性审计。],
) <tab-task2-metric-audit>

HPWL 与 congestion 在该 bank 上具有中等正相关，说明降低 HPWL 时经常也能降低该 proxy；相关系数远低于 1，高质量 HPWL 区域内仍保留 29.07% 的 congestion relative range。$256 times 256$ 与 $512 times 512$ 的排序相关达到 0.99943，指标排序对这一分辨率变化稳定。最高 degree net `n7807` 有 318 个 pins，但它单独产生的 congestion 最多只占完整指标的 4.88%，移除它后的 ranking Spearman 仍为 0.99933。

#figure(
  image("assets/task2/objective_space_and_task1_coverage.png", width: 100%),
  caption: [归一化目标空间。左：两种算法五个 seeds 的最终 phenotype-deduplicated fronts；右：Task 2 common front 与现有 Task 1 mixed saved-best reference。两个目标均越低越好。],
) <fig-task2-objective-space>

图 @fig-task2-objective-space 左侧显示正式搜索形成了连续的 HPWL-congestion trade-off。各 run 访问区域中的 phenotype-deduplicated nondominated 数量为 56-236；HPWL-Congestion Spearman 随 seed 与算法在 $-0.448$ 到 0.720 之间变化。第二目标在当前 audited/visited regions 中提供了可区分的局部选择信号。

=== Phenotype redundancy 与 MOEA/D collapse

MOEA/D 的多样性损失在第一个完整 sweep 就出现。每个 seed 的 20 个初始 weight slots 都对应 19 或 20 个 distinct phenotypes；evaluation 40 时平均只剩 3.8 个，单一 phenotype 恰好占 50% slots。五个 seeds 的 `first_half_population_collapse_sweep` 均为 1。

#figure(
  image("assets/task2/moead_slot_collapse.png", width: 100%),
  caption: [MOEA/D weight-slot occupancy。细线表示单个 seed，粗线表示 5-seed mean；横轴使用对数尺度，虚线对应第一个完整 sweep。],
) <fig-task2-moead-collapse>

最严重的 seed 曾由一个 phenotype 占据全部 20 个 slots；其他 seeds 的最大集中比例为 60%-75%。evaluation 10,000 时，五个 runs 分别恢复到 7、7、14、11、14 个 slot phenotypes，最大单一 phenotype 比例仍为 15%-40%。所有 runs 的 historical ideal point 均保持 component-wise non-regression，因此 collapse 与 ideal-point 更新错误无关。当前 MOEA/D 允许一个 offspring 同时替换所有改善的邻居；在 population size 为 20、neighborhood size 为 10 时，早期优质 offspring 可以迅速复制到多个相邻 weight directions。该机制与 slot-state trace 一致，但本文没有通过 replacement-cap ablation 验证独立因果效应。

MOEA/D 最终平均 HV 略高，说明 slot concentration 在本实验中没有阻止二维 HV 的继续改善；它显著减少了最终 phenotype 数，并提高了 seed 对历史搜索路径的依赖。若 Task 3 关注稳定、多样的 Pareto set，weight-slot diversity 是比继续增加评价预算更直接的干预对象。

=== Uniform + Swap 的实际扰动及其后果

Task 1 对 Uniform + Swap 的解释主要来自最终性能。Task 2 的 per-evaluation trace 可以直接比较 offspring 与两个父代解码后的 placement。图 @fig-task2-displacement 将每个 parent-child comparison 按实际移动宏数分组；rate 按每组 comparison 数量跨 seeds 加权。

#figure(
  image("assets/task2/displacement_conditioned_outcomes.png", width: 100%),
  caption: [按相对父代的实际移动宏数量分组后的 offspring survival 与 dominated-by-parent rate。数值按 parent-child comparisons 加权。],
) <fig-task2-displacement>

NSGA-II 中，移动 1-10、11-100、超过 100 个宏的 offspring survival rate 分别为 58.8%、30.7% 与 5.3%；被相应父代支配的比例从 7.7% 上升到 43.0% 与 59.0%。MOEA/D 的同三组 survival rate 为 7.5%、10.3% 与 5.6%，dominated rate 为 7.8%、35.3% 与 56.0%。超过 100 个宏的大范围 placement change 在两个算法中都更容易破坏已有 trade-off structure。

零位移 offspring 在 NSGA-II 中仍有 60.9% 被标记为 survivor，MOEA/D 中为 4.5%。这里的 survivor 表示 offspring object 是否保留在算法状态中；NSGA-II 的 rank/crowding survival 可以保留 objective-identical individuals，而 MOEA/D 只有改善 decomposition value 才执行 replacement，因此两者的绝对 survival rate 不能解释为 variation 本身的质量差异。

累计 trace 中，NSGA-II offspring 相对父代改善 HPWL 与 congestion 的比例平均高于 MOEA/D，但两种算法使用了不同 parent selection 和 survivor semantics。现有数据能够描述 combined Uniform-plus-Swap pipeline 的实际 phenotype effect，无法分离 Uniform crossover 与 Swap mutation 的独立贡献。

=== 与 Task 1 参考结果的比较

可用的 Task 1 历史 artifacts 没有保存五个 Uniform + Swap runs 的完整 final populations，也没有中间 population snapshots。当前 reference 由 15 个 mixed saved-best distinct phenotypes 构成，覆盖多个可用 Task 1 formal/smoke runs；它适合作为已有单目标结果的有限参照，不能代表某个算子组合的完整最终解集。

该 reference 在 Task 2 两个目标下只有 1 个 nondominated point，其 HV 为 1.1113。Task 2 十个 final populations 的 pooled common front 含 23 个 points，HV 为 1.2502；Task 1 的 nondominated reference point 被 Task 2 front 支配，Task 2 front 没有 point 被该 Task 1 point 支配。合并 front 的 23 个贡献全部来自 Task 2，如图 @fig-task2-objective-space 右侧所示。

多目标搜索扩展了现有 Task 1 saved-best reference，并找到了同时具有更低 utilization proxy 的高质量布局。由于 reference scope 不完整，本文不声称 Task 2 覆盖了 Task 1 Uniform + Swap 的完整 final populations。

=== 对 Task 3 的启示

Task 2 给出了三组可以直接指导 Task 3 的证据：

+ NSGA-II 的早期 HV 上升更快，最终维持接近满 population 的 distinct phenotypes；MOEA/D 的平均最终 HV 略高，但 paired differences 对 seed 敏感。
+ MOEA/D 在第一个 sweep 发生系统性的 slot collapse，后期只能部分恢复。限制单个 offspring 的邻域 replacement、维护 phenotype-aware occupancy 或增加适度的 diversity pressure 都有明确的数据动机。
+ Uniform + Swap 能产生大量实际 placement changes；超过 100 个宏的大范围变化通常具有较低 survival 和较高 dominated rate。Task 3 可以优先考虑 decoder-aware、位移受控的 variation，并通过独立 ablation 判断收益来自扰动尺度还是新的结构信息。

当前 Task 2 在一个 top-512 macro 实例上建立了受控 baseline 与机制诊断。完整 543-macro adaptec1、其他 benchmarks、真实 global-routing congestion、Shift 的早停机制、Lamarckian write-back、invalid repair 和完整 operator factorial 均需要新的实验设计，现有 trace 不足以支持这些外推。

#pagebreak()
= Task 3：改进算法


== 结果分析

尚无正式结果。

#pagebreak()
= 实验框架捉bug <Bug>

== Bug 1：MGO net traversal 非确定性

原实现通过无序集合构造 macro-to-net mapping，导致 net traversal 顺序可能随 Python hash 状态或输入插入顺序变化。由于 MGO 的 greedy placement 会逐步更新 wire mask，遍历顺序变化可能改变 phenotype 与 HPWL，并在并发环境中表现为难以复现的差异。修复后对 net 名称排序并存为不可变 tuple，同时增加输入顺序反转测试。详见 #link("https://github.com/lamda-bbo/BBOPlace-Bench/pull/12")[PR #12]。

== Bug 2：初始化 population 被重复评价

Sampling 已经评价全部初始候选并选出保留个体，但原接口只向 pymoo 返回 `X`，导致保留的初始 population 再被评价一次。这样既浪费真实预算，也使 pymoo 计数与 placer 真实计数不一致。修复后 Sampling 返回包含 `X`、`F`、`overlap_rate` 和 `macro_pos` 的 `Population`，并正确标记已知评价字段。详见 #link("https://github.com/lamda-bbo/BBOPlace-Bench/pull/13")[PR #13]。

== Bug 3：Swap mutation 的高级索引错误

原 Swap 使用二维高级索引在整个 batch 上采样和赋值，无法保证每个 individual 独立选择两个 macro，且赋值广播语义可能破坏预期交换。修复后逐 individual 无放回采样两个 macro index，并同时交换完整的 $(x, y)$ coordinate pair；输入数组保持不变。详见 #link("https://github.com/lamda-bbo/BBOPlace-Bench/pull/13")[PR #13]。

== Bug 4：DummyMutation 初始化与约束字段标记

`DummyMutation` 原先调用了另一个 mutation class 的 `super`，实例化即可触发错误。修复后使用本类正确的 `super().__init__`。此外，Sampling 不再无条件将 `G`、`H` 标记为已评价，而是仅在问题不存在对应约束时标记，避免约束问题被错误跳过。相关回归测试覆盖 no-op 行为、评价预算和约束字段。详见 #link("https://github.com/lamda-bbo/BBOPlace-Bench/pull/13")[PR #13]。


#pagebreak()
= 附录

== Task 1 运行与复现

运行：

```bash
python experiments/run_task1.py formal \
  --python=/home/sihengzhao/t1/env/bin/python \
  --workers=8 \
  --cpus-per-run=12
```

在我的实际配置上，每个 run 在独立进程与独立 Ray 临时目录中执行，设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`CUDA_VISIBLE_DEVICES=""` 和 `PYTHONHASHSEED=seed`。调度上限为 8 个并发 runs、每 run 12 CPU、总上限 96 logical CPUs，并以 `nice -n 10` 降低对共享服务器的影响。正式套件总耗时 32,546 s，即约 9 h 02 min 26 s。

结果目录包含：

```text
results/adaptec1/task1_formal__*/mgo/ea/seed_*/
  evaluation_trace.csv
  generation_metrics.csv
  metrics.csv
  initial_population.npz
  initial_population_hash.json
  resolved_config.yaml
  runtime_metadata.json
  run_complete.json
  placements/
  figures/
```

分析命令与报告图命令为：

```bash
python experiments/analyze_task1.py
python experiments/render_task1_report.py \
  --output-dir experiments/task1_analysis/report_figures
```

== Task 2 运行与复现

Task 2 的环境安装与激活脚本为：

```bash
bash script/setup_task2_remote_env.sh
source script/task2_remote_env.sh
```

正式运行前，`run_task2.py` 会读取冻结协议、自动生成 smoke decision，并验证 benchmark、代码、环境、DREAMPlace binary 与 experiment definition fingerprints。正式命令为：

```bash
python3 experiments/run_task2.py --mode formal
```

正式 scheduler 使用 CPU backend、2 个并发 runs、每 run 20 logical CPUs、`rudy_cpu_threads = 1`、`nice = 10`，并为其他用户保留至少 32 个 logical CPUs。10 个 runs 均完成 10,000 次真实评价，总计 100,000 次。最早 run 于 2026-07-24 15:29:20 UTC 启动，最后 run 于 2026-07-25 03:47:17 UTC 完成，formal suite wall time 为 44,276.8 s，即约 12 h 17 min 57 s。所有 runs 的累计进程时间为 74,275.5 s；两者不同是因为 scheduler 同时运行两个独立实验。

正式结果位于：

```text
experiments/task2_runs/formal/
  summary.json
  seed_*__nsga2.json
  seed_*__moead.json
  seed_*__*.log

results/adaptec1/task2_formal__*/mgo/task2_moea/seed_*/
  evaluation_trace.csv
  generation_metrics.csv
  final_population.npz
  offline_archive.npz
  population_snapshots/
  moead_state.csv              # MOEA/D only
  initial_population.npz
  initial_population_hash.json
  placedb_fingerprint.json
  resolved_config.yaml
  runtime_metadata.json
  run_complete.json
```

分析和报告绘图命令与 Task 1 对称：

```bash
python3 experiments/analyze_task2.py
python3 experiments/render_task2_report.py \
  --output-dir assets/task2
```

分析结果写入 `experiments/task2_analysis/`，包括 final/archive HV、paired differences、convergence、phenotype diversity、MOEA/D slot occupancy、displacement-conditioned outcomes 与 Task 1 coverage。绘图脚本同时生成 PNG 和 PDF，并写出 `assets/task2/figure_manifest.csv`。

正式 run 使用 commit `5bbbd42` 对应的 code fingerprint `bd58e94c...bac5` 与 environment fingerprint `af9633bb...9e5dd`。后续对 fingerprint postprocessing 与报告导出的修正没有重新评价任何 genotype，也没有改变原始 formal results。

== Task 3 运行结果、复现方法与运行时间

尚未产生正式结果。
