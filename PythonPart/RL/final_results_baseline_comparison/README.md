# 基线对比实验最终结果
## Baseline Comparison: Full Observation vs Partial Observation

实验日期：2026-03-16

## 实验设计

### 观测空间对比

**Group 1 (完整观测, 8维)**:
- [0] lam_dot: 视线角速率 / 0.5
- [1] look_angle: 视线偏差角 / fov
- [2] lam_ddot: 视线角加速度 / 5.0
- [3] am: 当前加速度 / a_max
- [4] gamma_m: 弹体航向角 / π
- [5] t: 飞行时间 / t_max
- [6] r_dot: 接近速率 / V
- [7] r: 弹目距离 / r_init_max

**Group 2 (部分观测, 6维)**:
- [0] lam_dot: 视线角速率 / 0.5
- [1] look_angle: 视线偏差角 / fov
- [2] lam_ddot: 视线角加速度 / 5.0
- [3] am: 当前加速度 / a_max
- [4] gamma_m: 弹体航向角 / π
- [5] t: 飞行时间 / t_max

**关键**：Group 1 = Group 2 + {r, r_dot}（严格超集关系）

### 训练配置

- 算法：PPO (Proximal Policy Optimization)
- 训练步数：2,000,000
- 并行环境：4个
- 场景：4个固定场景（0g, 5g, 8g, 10g）
- 奖励函数：ZEM-based（原始奖励函数）
- 归一化：lam_dot/0.5（已验证80%基线）
- 随机种子：42

### 评估配置

- 测试场景：6个机动强度（0g, 2g, 4g, 6g, 8g, 10g）
- 每个场景：200次蒙特卡洛试验
- 总试验次数：1200次
- 命中标准：r ≤ 0.5m

## 文件说明

- `train_baseline_comparison.py`: 训练脚本
- `evaluate_baseline_comparison.py`: 评估脚本
- `resume_training.py`: 恢复训练脚本
- `models/`: 训练好的模型
  - `baseline_full_obs/best_model.zip`: Group 1 (8D)
  - `baseline_partial_obs/best_model.zip`: Group 2 (6D)
- `logs/`: 训练日志和TensorBoard数据
- `checkpoints/`: 训练过程中的检查点
- `outputs/`: 评估结果CSV文件

## 结果

见 `evaluation_results.txt`
