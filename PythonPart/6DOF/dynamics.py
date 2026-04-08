"""
6DOF刚体动力学核心
==================
状态向量12维: [x, y, z, u, v, w, phi, theta, psi, p, q, r]

坐标系约定（NED）:
  惯性系: X北 Y东 Z下
  体轴系: x前 y右 z下
  欧拉角: phi(滚转), theta(俯仰), psi(偏航), 旋转顺序 ZYX

积分方法: RK4
"""

import numpy as np
from config import MissileParams, PropulsionParams, G, RHO


# ============================================================
#  坐标变换
# ============================================================

def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    欧拉角 → 方向余弦矩阵 (体轴系→惯性系)
    旋转顺序 ZYX: 先偏航psi, 再俯仰theta, 最后滚转phi
    R_bi: 体轴系向量 → 惯性系向量
    v_inertial = R_bi @ v_body
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)

    R = np.array([
        [ct*cs,  sp*st*cs - cp*ss,  cp*st*cs + sp*ss],
        [ct*ss,  sp*st*ss + cp*cs,  cp*st*ss - sp*cs],
        [-st,    sp*ct,             cp*ct            ],
    ])
    return R


def body_to_inertial(v_body: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """体轴系向量 → 惯性系向量"""
    return euler_to_dcm(phi, theta, psi) @ v_body


def inertial_to_body(v_inertial: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """惯性系向量 → 体轴系向量"""
    return euler_to_dcm(phi, theta, psi).T @ v_inertial


# ============================================================
#  气动力计算
# ============================================================

def compute_aero(u: float, v: float, w: float,
                 p: float, q: float, r: float,
                 delta_e: float, delta_r: float, delta_a: float,
                 mp: MissileParams, mass: float) -> tuple:
    """
    计算气动力和力矩（体轴系）

    参数:
        u, v, w: 体轴系速度分量 m/s
        p, q, r: 体轴系角速率 rad/s
        delta_e: 俯仰舵偏角 rad（正→弹头上仰）
        delta_r: 偏航舵偏角 rad（正→弹头右偏）
        delta_a: 滚转舵偏角 rad
        mp: 导弹参数
        mass: 当前质量 kg

    返回:
        (Fx, Fy, Fz, Mx, My, Mz) 体轴系气动力(N)和力矩(N·m)
    """
    V = np.sqrt(u**2 + v**2 + w**2)
    if V < 0.1:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 攻角和侧滑角
    alpha = np.arctan2(w, max(u, 0.1))  # 攻角（u接近0时保护）
    beta = np.arctan2(v, max(u, 0.1))   # 侧滑角

    # 动压
    qbar = 0.5 * RHO * V**2
    S = mp.S_ref
    d = mp.d_ref

    # --- 力 ---
    # 轴向力（阻力，沿-x方向）
    Fx = -qbar * S * mp.CD0

    # 法向力（攻角产生，沿-z方向；侧滑角产生，沿-y方向）
    Fz = -qbar * S * mp.CNa * alpha
    Fy = -qbar * S * mp.CNa * beta

    # --- 力矩 ---
    # 静稳定力矩臂
    x_cp = mp.x_cp_ratio * mp.length  # 压心位置（从弹头）
    x_cg = mp.x_cg_ratio * mp.length  # 质心位置（从弹头）
    arm = x_cp - x_cg  # 正值 = 压心在质心之后 = 静稳定

    # 俯仰力矩 My: 攻角产生恢复力矩 + 阻尼 + 舵面控制
    # Cm_alpha 贡献: 法向力 * 力臂（攻角正→法向力向-z→力矩使弹头下压→恢复）
    My_alpha = -qbar * S * mp.CNa * alpha * arm
    # 俯仰阻尼力矩: Cmq * qbar * S * d * (q * d / (2V))
    My_damp = mp.Cmq_coeff * qbar * S * d * (q * d / (2.0 * V))
    # 舵面控制力矩
    My_ctrl = qbar * S * d * mp.Cmd * delta_e
    My = My_alpha + My_damp + My_ctrl

    # 偏航力矩 Mz: 侧滑角产生力矩 + 阻尼 + 舵面控制
    # 注意：对于轴对称弹体，Fy作用在CP处，力矩 = Fy * arm（绕CG）
    # Fy = -qbar*S*CNa*beta (beta>0时力向左)
    # CP在CG后方，左向力在后方 → 鼻向左转(Mz<0) → 恢复
    Mz_beta = qbar * S * mp.CNa * beta * arm
    Mz_damp = mp.Cmq_coeff * qbar * S * d * (r * d / (2.0 * V))
    Mz_ctrl = qbar * S * d * mp.Cmd * delta_r
    Mz = Mz_beta + Mz_damp + Mz_ctrl

    # 滚转力矩 Mx: 简化，仅舵面控制 + 阻尼
    Mx_damp = -2.0 * qbar * S * d * (p * d / (2.0 * V))  # 滚转阻尼
    Mx_ctrl = qbar * S * d * 0.5 * mp.Cmd * delta_a
    Mx = Mx_damp + Mx_ctrl

    return Fx, Fy, Fz, Mx, My, Mz


# ============================================================
#  6DOF状态导数
# ============================================================

def missile_derivatives(state: np.ndarray, t: float,
                        delta_e: float, delta_r: float, delta_a: float,
                        mp: MissileParams, prop: PropulsionParams,
                        mass: float) -> np.ndarray:
    """
    计算6DOF状态导数

    state: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
    返回: d(state)/dt (12维)
    """
    x, y, z, u, v, w, phi, theta, psi, p, q, r = state

    # --- 当前质量和惯量 ---
    # 惯量随质量线性缩放（简化）
    mass_ratio = mass / mp.m0
    Ix = mp.Ix0 * mass_ratio
    Iy = mp.Iy0 * mass_ratio
    Iz = mp.Iz0 * mass_ratio

    # --- 外力（体轴系）---
    # 1. 气动力
    Fx_a, Fy_a, Fz_a, Mx_a, My_a, Mz_a = compute_aero(
        u, v, w, p, q, r, delta_e, delta_r, delta_a, mp, mass)

    # 2. 推力（沿体轴x正方向）
    thrust = prop.get_thrust(t)
    Fx_t = thrust

    # 3. 重力（惯性系 [0, 0, g] → 体轴系）
    g_body = inertial_to_body(np.array([0.0, 0.0, G]), phi, theta, psi)
    Fx_g = mass * g_body[0]
    Fy_g = mass * g_body[1]
    Fz_g = mass * g_body[2]

    # 总力
    Fx = Fx_a + Fx_t + Fx_g
    Fy = Fy_a + Fy_g
    Fz = Fz_a + Fz_g

    # --- 平动方程（体轴系，含科氏项）---
    u_dot = Fx / mass + r * v - q * w
    v_dot = Fy / mass - r * u + p * w
    w_dot = Fz / mass + q * u - p * v

    # --- 转动方程（Euler方程）---
    p_dot = (Mx_a + (Iy - Iz) * q * r) / Ix
    q_dot = (My_a + (Iz - Ix) * p * r) / Iy
    r_dot = (Mz_a + (Ix - Iy) * p * q) / Iz

    # --- 欧拉角运动学 ---
    cp, sp = np.cos(phi), np.sin(phi)
    ct, tt = np.cos(theta), np.tan(theta)

    # 防止theta接近±90°时tan爆炸（本场景不会出现极端姿态）
    if abs(ct) < 1e-6:
        ct = 1e-6 * np.sign(ct) if ct != 0 else 1e-6
        tt = np.sin(theta) / ct

    phi_dot = p + (q * sp + r * cp) * tt
    theta_dot = q * cp - r * sp
    psi_dot = (q * sp + r * cp) / ct

    # --- 位置导数（体轴系速度 → 惯性系）---
    v_inertial = body_to_inertial(np.array([u, v, w]), phi, theta, psi)

    return np.array([
        v_inertial[0], v_inertial[1], v_inertial[2],  # dx, dy, dz
        u_dot, v_dot, w_dot,                            # du, dv, dw
        phi_dot, theta_dot, psi_dot,                    # dphi, dtheta, dpsi
        p_dot, q_dot, r_dot,                            # dp, dq, dr
    ])


# ============================================================
#  RK4积分器
# ============================================================

def rk4_step(state: np.ndarray, t: float, dt: float,
             delta_e: float, delta_r: float, delta_a: float,
             mp: MissileParams, prop: PropulsionParams,
             mass: float) -> np.ndarray:
    """RK4单步积分"""
    args = (delta_e, delta_r, delta_a, mp, prop, mass)

    k1 = missile_derivatives(state, t, *args)
    k2 = missile_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = missile_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = missile_derivatives(state + dt * k3, t + dt, *args)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
#  导弹6DOF仿真器
# ============================================================

class Missile6DOF:
    """
    6DOF导弹仿真器

    管理导弹状态、质量变化、舵偏角限幅，提供单步积分接口。
    """

    def __init__(self, mp: MissileParams = None, prop: PropulsionParams = None):
        self.mp = mp or MissileParams()
        self.prop = prop or PropulsionParams()

        # 12维状态
        self.state = np.zeros(12)
        self.mass = self.mp.m0
        self.t = 0.0

        # 舵偏角（当前值）
        self.delta_e = 0.0  # 俯仰
        self.delta_r = 0.0  # 偏航
        self.delta_a = 0.0  # 滚转

    def init_state(self, pos: np.ndarray, euler: np.ndarray,
                   speed: float = 0.0):
        """
        初始化导弹状态

        pos: [x, y, z] 惯性系位置 m
        euler: [phi, theta, psi] 欧拉角 rad
        speed: 初始速度 m/s（沿体轴x方向）
        """
        self.state = np.zeros(12)
        self.state[0:3] = pos
        self.state[3] = speed  # u = speed（体轴系前向）
        self.state[6:9] = euler
        self.mass = self.mp.m0
        self.t = 0.0
        self.delta_e = 0.0
        self.delta_r = 0.0
        self.delta_a = 0.0

    def set_fins(self, delta_e: float, delta_r: float, delta_a: float = 0.0):
        """
        设置舵偏角指令（含限幅和速率限制）
        """
        dm = self.mp.delta_max
        dr_max = self.mp.delta_rate_max

        # 限幅
        delta_e = np.clip(delta_e, -dm, dm)
        delta_r = np.clip(delta_r, -dm, dm)
        delta_a = np.clip(delta_a, -dm, dm)

        # 速率限制（在step中应用更精确，这里简化为直接设置）
        self.delta_e = delta_e
        self.delta_r = delta_r
        self.delta_a = delta_a

    def step(self, dt: float):
        """推进一个时间步长"""
        # RK4积分
        self.state = rk4_step(
            self.state, self.t, dt,
            self.delta_e, self.delta_r, self.delta_a,
            self.mp, self.prop, self.mass
        )

        # 质量更新
        mdot = self.prop.get_mass_flow(self.t)
        self.mass -= mdot * dt
        self.mass = max(self.mass, self.mp.m0 - self.mp.m_propellant)

        # 时间推进
        self.t += dt

    # --- 便捷属性 ---
    @property
    def pos(self) -> np.ndarray:
        return self.state[0:3].copy()

    @property
    def vel_body(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def euler(self) -> np.ndarray:
        return self.state[6:9].copy()

    @property
    def omega(self) -> np.ndarray:
        return self.state[9:12].copy()

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.state[3:6])

    @property
    def vel_inertial(self) -> np.ndarray:
        phi, theta, psi = self.state[6:9]
        return body_to_inertial(self.state[3:6], phi, theta, psi)

    @property
    def alpha(self) -> float:
        """攻角 rad"""
        u, v, w = self.state[3:6]
        return np.arctan2(w, max(u, 0.1))

    @property
    def beta(self) -> float:
        """侧滑角 rad"""
        u, v, w = self.state[3:6]
        return np.arctan2(v, max(u, 0.1))

    @property
    def normal_accel(self) -> float:
        """法向过载 (g)，近似为 q_bar * S * CNa * alpha / (mass * g)"""
        V = self.speed
        if V < 1.0:
            return 0.0
        qbar = 0.5 * RHO * V**2
        return qbar * self.mp.S_ref * self.mp.CNa * abs(self.alpha) / (self.mass * G)
