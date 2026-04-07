# 论文图表说明

本文件夹包含9张高质量的论文级别图表，所有图表均为300 DPI，适合直接插入论文。

---

## 基础对比图（图1-5）

### 图1: 命中率对比 (fig1_hit_rate_comparison.png)
**用途**: 展示三种方法在不同机动强度下的命中率
**关键信息**:
- PNG在高机动场景下性能急剧下降（10g仅15%）
- RL方法保持高命中率（Group 1: 91%, Group 2: 80%）
- 清晰展示RL的优势

**建议使用位置**: 论文正文第一张对比图，Results部分

---

### 图2: 脱靶量对比 (fig2_miss_distance_comparison.png)
**用途**: 展示三种方法的精度对比（对数坐标）
**关键信息**:
- 红色虚线标注命中标准（0.5m）
- PNG在高机动场景超出命中标准
- RL方法脱靶量稳定在0.3-0.5m范围
- 对数坐标清晰展示数量级差异

**建议使用位置**: Results部分，精度分析

---

### 图3: 能量消耗对比 (fig3_energy_consumption_comparison.png)
**用途**: 展示三种方法的能量消耗
**关键信息**:
- RL能量消耗是PNG的2-2.5倍
- 能量随机动强度增长
- Group 2比Group 1能量高11.5%

**建议使用位置**: Discussion部分，能量-性能权衡分析

---

### 图4: 飞行时间对比 (fig4_flight_time_comparison.png)
**用途**: 展示拦截时间对比
**关键信息**:
- RL拦截速度比PNG快24.8%
- 飞行时间随机动强度略有增加
- 三种方法的时间差异不大

**建议使用位置**: Results部分，次要指标

---

### 图5: 总体性能柱状图 (fig5_overall_performance_bar.png)
**用途**: 直观对比总体性能和高机动性能
**关键信息**:
- 蓝色：总体命中率（0-10g）
- 红色：高机动命中率（8-10g）
- 数值标签清晰
- 适合做摘要图或演示

**建议使用位置**: Abstract图形摘要，或Results开头

---

## 高级分析图（图6-9）

### 图6: 鲁棒性分析 (fig6_robustness_analysis.png)
**用途**: 展示性能下降趋势（相对于0g场景）
**关键信息**:
- PNG性能下降83%（0g→10g）
- Group 1性能下降仅9%
- Group 2性能下降20%
- 清晰展示鲁棒性差异

**建议使用位置**: Discussion部分，鲁棒性分析

**论文描述示例**:
"Figure 6 shows the performance degradation relative to the baseline (0g) scenario. PNG exhibits severe degradation (83%) under high maneuvers, while RL methods maintain robust performance with only 9-20% degradation."

---

### 图7: 能量效率分析 (fig7_energy_efficiency.png)
**用途**: 展示能量/命中率比值（越低越好）
**关键信息**:
- 对数坐标展示效率差异
- PNG在高机动场景效率极低（能量高但命中率低）
- RL方法效率更稳定

**建议使用位置**: Discussion部分，能量效率分析

**论文描述示例**:
"Figure 7 presents the energy efficiency metric (energy per hit rate). Despite higher absolute energy consumption, RL methods achieve better efficiency in high-g scenarios due to their superior hit rates."

---

### 图8: 雷达图 (fig8_radar_chart.png)
**用途**: 多维性能对比（5个维度）
**关键信息**:
- 5个维度：总体命中率、高机动命中率、脱靶量、能量、鲁棒性
- 所有指标归一化到0-100（越大越好）
- 面积越大表示综合性能越好
- 适合做总结图

**建议使用位置**: Conclusion部分，或作为图形摘要

**论文描述示例**:
"Figure 8 provides a multi-dimensional performance comparison. RL methods (blue and green) significantly outperform PNG (red) across most dimensions, with Group 1 achieving the best overall balance."

---

### 图9: 性能热力图 (fig9_performance_heatmap.png)
**用途**: 矩阵形式展示所有性能指标
**关键信息**:
- 3个子图：PNG、Group 1、Group 2
- 4个指标 × 6个场景 = 24个数据点
- 颜色深浅表示数值大小
- 数值标注清晰
- 适合做附录或补充材料

**建议使用位置**: Appendix或Supplementary Materials

**论文描述示例**:
"Figure 9 presents a comprehensive performance matrix across all test scenarios. The heatmap visualization reveals consistent high performance (blue) for Group 1 across all maneuver intensities."

---

## 图表使用建议

### 核心图表（必须包含）
1. **图1 (命中率)** - 最重要的性能指标
2. **图5 (总体性能柱状图)** - 直观的总结
3. **图8 (雷达图)** - 多维对比

### 补充图表（根据篇幅选择）
4. **图2 (脱靶量)** - 精度分析
5. **图6 (鲁棒性)** - 鲁棒性分析
6. **图3 (能量消耗)** - 能量分析

### 附录图表
7. **图4 (飞行时间)** - 次要指标
8. **图7 (能量效率)** - 深入分析
9. **图9 (热力图)** - 完整数据

---

## 图表质量说明

- **分辨率**: 300 DPI（出版级别）
- **格式**: PNG（支持透明背景）
- **字体**: Serif（学术风格）
- **颜色方案**:
  - PNG: 红色 (#E74C3C)
  - Group 1: 蓝色 (#3498DB)
  - Group 2: 绿色 (#2ECC71)
- **线宽**: 2.5pt（清晰可见）
- **标记大小**: 8pt（适中）

---

## LaTeX插入示例

### 单列图（推荐）
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\columnwidth]{fig1_hit_rate_comparison.png}
    \caption{Hit rate comparison across different target maneuver intensities.
    RL methods (Group 1 and Group 2) significantly outperform PNG, especially
    in high-g scenarios (8-10g).}
    \label{fig:hit_rate}
\end{figure}
```

### 双列图
```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{fig8_radar_chart.png}
    \caption{Multi-dimensional performance comparison using radar chart.
    Five key metrics are normalized to 0-100 scale (higher is better).}
    \label{fig:radar}
\end{figure*}
```

### 子图组合
```latex
\begin{figure*}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{fig1_hit_rate_comparison.png}
        \caption{Hit rate}
        \label{fig:sub1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{fig2_miss_distance_comparison.png}
        \caption{Miss distance}
        \label{fig:sub2}
    \end{subfigure}
    \caption{Performance comparison: (a) hit rate and (b) miss distance.}
    \label{fig:performance}
\end{figure*}
```

---

## 图表说明文字模板

### 图1说明
"Figure 1 compares the hit rates of PNG and RL-based guidance methods across six target maneuver intensities (0-10g). RL methods maintain high hit rates (>80%) even under extreme maneuvers (10g), while PNG performance degrades significantly (15% at 10g). Group 1 (8D full observation) achieves the best overall performance (98.0%), followed by Group 2 (6D partial observation, 94.8%)."

### 图5说明
"Figure 5 summarizes the overall performance comparison. The blue bars represent overall hit rates (averaged across all scenarios), while red bars show high-g performance (8-10g scenarios only). RL methods demonstrate superior performance in both metrics, with Group 1 achieving 98.0% overall and 94.2% in high-g scenarios."

### 图8说明
"Figure 8 presents a radar chart comparing five key performance dimensions: overall hit rate, high-g hit rate, miss distance (inverted), energy consumption (inverted), and robustness. All metrics are normalized to 0-100 scale where higher values indicate better performance. The larger coverage area of RL methods (blue and green) demonstrates their superior multi-dimensional performance compared to PNG (red)."

---

## 配色方案说明

选择这三种颜色的原因：
1. **红色 (PNG)**: 传统方法，警示色，暗示性能不足
2. **蓝色 (Group 1)**: 冷静、专业，代表最优方案
3. **绿色 (Group 2)**: 环保、可行，代表次优但可接受的方案

这三种颜色在色盲友好性测试中表现良好，适合黑白打印。

---

## 文件清单

```
final_results_baseline_comparison/
├── fig1_hit_rate_comparison.png          (8.0 × 6.0 inch, 300 DPI)
├── fig2_miss_distance_comparison.png     (8.0 × 6.0 inch, 300 DPI)
├── fig3_energy_consumption_comparison.png (8.0 × 6.0 inch, 300 DPI)
├── fig4_flight_time_comparison.png       (8.0 × 6.0 inch, 300 DPI)
├── fig5_overall_performance_bar.png      (10.0 × 6.0 inch, 300 DPI)
├── fig6_robustness_analysis.png          (8.0 × 6.0 inch, 300 DPI)
├── fig7_energy_efficiency.png            (8.0 × 6.0 inch, 300 DPI)
├── fig8_radar_chart.png                  (8.0 × 8.0 inch, 300 DPI)
└── fig9_performance_heatmap.png          (15.0 × 5.0 inch, 300 DPI)
```

---

**生成日期**: 2026-03-16
**生成脚本**: generate_publication_figures.py
**数据来源**: png_baseline.csv, group_1_full_obs_8d.csv, group_2_partial_obs_6d.csv
