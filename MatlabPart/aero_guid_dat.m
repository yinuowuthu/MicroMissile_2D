% AERO_GUID_DAT_ANTIMAV  反无人机微型拦截弹参数
% 目标：弹簧刀300 / RQ-11渡鸦
% 基于 aero_guidance 示例模型框架

%==================================================================
% 常数
%==================================================================
d2r     = pi/180;
g       = 9.81;
m2ft    = 3.28084;
Kg2slug = 0.0685218;

%==================================================================
% 标准大气常数（不变）
%==================================================================
T0      = 288.16;
rho0    = 1.225;
L       = 0.0065;
R       = 287.26;
gam     = 1.403;
P0      = 101325.0;
h_trop  = 11000.0;

%==================================================================
% 拦截弹构型（微型反无人机导弹）
%==================================================================
d_ref   = 0.072;                  % 弹径 [m]
S_ref   = pi*(d_ref/2)^2;         % 参考面积 = 截面积 [m²] ≈ 0.00407
mass    = 3.5;                    % 发射质量 [kg]
Thrust  = 200.0;                  % 固推推力 [N]，燃烧约2.5s

% 惯量（轴对称细长体估算）
Iyy     = mass * (0.9)^2 / 12;    % 纵向惯量 ≈ 0.236 kg·m²
Ixx     = mass * (d_ref/2)^2 / 2; % 滚转惯量 ≈ 0.00045 kg·m²（远小于Iyy）
Izz     = Iyy;                    % 轴对称：Izz = Iyy
Ixz     = 0;

%==================================================================
% 气动数据（亚音速区间，Mach 0.3~0.9）
%==================================================================                         
Mach_vec  = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
alpha_vec = (-20:1:20)' * d2r;
[M, al]   = meshgrid(Mach_vec, alpha_vec/d2r);  % al单位：度

% 轴向力系数（阻力，亚音速波阻小）
Cx_alpha  = -0.4 * ones(size(al));

% 法向力系数（轴对称体+尾翼，亚音速线性区更宽）
% 结构：细长体贡献(线性) + 非线性修正项
an  =  0.000050;                  % 三次项（弱，亚音速非线性小）
bn  = -0.004000;                  % 二次项
cn  = -0.025000;                  % 线性项 [/deg]，典型亚音速尾翼弹
Cz_alpha = an*al.^3 + bn*al.*abs(al) + cn*al;  % Mach弱相关，去掉M项
Cz_el    = -0.018 / d2r;          % 舵面效率 [/rad]

% 俯仰力矩系数（静稳定，压心在质心后）
% 负号：正攻角→恢复力矩（稳定）
am  = -0.000100;
bm  = -0.008000;
cm  =  0.020000;                  % 线性项，静稳定余量约5%弦长
Cm_alpha = am*al.^3 + bm*al.*abs(al) - cm*al;
Cm_el    = -0.140 / d2r;          % 俯仰操纵效率
Cm_q     = -8.0;                  % 俯仰阻尼（亚音速较小）

%==================================================================
% 侧向气动（轴对称：直接镜像纵向）
%==================================================================
beta_vec  = alpha_vec;
[M_lat, be] = meshgrid(Mach_vec, beta_vec/d2r);

Cy_beta   = an*be.^3 + bn*be.*abs(be) + cn*be;  % = Cz_alpha(beta)
Cy_rudder = Cz_el;

Cn_beta   = -(am*be.^3 + bm*be.*abs(be) - cm*be);
Cn_rudder = -Cm_el;
Cn_r      = Cm_q;

Cl_beta   = zeros(size(be));      % 轴对称无滚转-侧滑耦合
Cl_p      = -0.3;                 % 滚转阻尼
Cl_aileron = -Cz_el / 4;

%==================================================================
% 初始条件
%==================================================================
x_ini     = 0;
h_ini     = 500;                  % 交战高度500m（低空反无人机典型场景）
v_ini     = 180;                  % 初速180 m/s（Mach≈0.53）
alpha_ini = 0 * d2r;
theta_ini = 0 * d2r;
q_ini     = 0 * d2r;
phi_ini   = 0 * d2r;
psi_ini   = 0 * d2r;
beta_ini  = 0 * d2r;
p_ini     = 0 * d2r;
r_ini     = 0 * d2r;
y_ini     = 0;

%==================================================================
% 目标参数（两种场景，注释切换）
%==================================================================

% --- 场景A：弹簧刀300（取冲刺速度，最恶劣情况）---
% pos_tgt   = [2000+x_ini, -h_ini-100];  % 2km前方，高100m
% v_tgt     = 45;        % 冲刺速度 45 m/s
% theta_tgt = 180*d2r;   % 迎头接近

% --- 场景B：RQ-11渡鸦（慢速目标）---
pos_tgt = [2000+x_ini, -h_ini+150];  % 目标比导弹低150m
v_tgt     = 20;
theta_tgt = 180*d2r;

%==================================================================
% 舵机参数（微型舵机，响应稍慢）
%==================================================================
wn_fin      = 80.0;               % 带宽降低（微型舵机）[rad/sec]
z_fin       = 0.7;
fin_act_0   = 0.0;
fin_max     =  25.0 * d2r;        % 偏转限制±25°
fin_min     = -25.0 * d2r;
fin_maxrate = 300 * d2r;          % 速率限制

%==================================================================
% 传感器（红外导引头）
%==================================================================
l_acc       = 0.2;                % 加速度计位置

%==================================================================
% 导引头参数
%==================================================================
wn_hom      = 5.0;                % 带宽降低（被动红外）
tors        = 0.08;               % 跟踪回路时间常数（略大）
max_gimbal  = 30 * d2r;
min_gimbal  = -30 * d2r;
wgyro       = 80 * 2 * pi;
Ks          = wgyro / 5;
K_r         = -0.01;
Beamwidth   = 15 * d2r;           % 红外导引头视场角（大于雷达）

%==================================================================
% 自动驾驶仪增益（亚音速重新整定）
% 变量名必须与查表块一致：M_sch, alpha_sch, Ka, K, KI, Kg
%==================================================================
% 只覆盖断点为亚音速
M_sch     = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1];
alpha_sch = [0, 4, 8, 12, 16, 20] * d2r;

% Ka/K/KI/Kg保持9×6矩阵，用原始值填满
Ka_val = 0.018;
Ka = Ka_val * ones(9, 6);
K  = Ka_val * ones(9, 6);
KI = 0.0053  * ones(9, 6);
Kg = 0.0858  * ones(9, 6);



KA   = 0.5;
KDC  = 0.0030 ;  
wI   = 0.0;
kg   = 0.017;
kac  = 0.00187;
c    = 0.0   ;   

max_acc = 25 * 9.81;

%==================================================================
% 几何参数（修正）
%==================================================================
d_ref   = 0.072;                      % 弹径/参考长度 [m]
S_ref   = 0.012;                      % 尾翼参考面积 [m²]

%==================================================================
% 气动数据
%==================================================================
Mach_vec = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3];
% 侧向气动断点（与纵向对齐）
Mach_vec_lat = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];  % 7点，对应41×7表
alpha_vec = (-20:1:20)' * d2r;
[M, al] = meshgrid(Mach_vec, alpha_vec/d2r);

Cx_alpha  = -0.4 * ones(size(al));    % 轴向力（阻力）

an  =  0.000050;
bn  = -0.004000;
cn  = -0.144;                         % 修正：保证足够法向力
Cz_alpha = an*al.^3 + bn*al.*abs(al) + cn*al;
Cz_el    = -0.045 / d2r;

am  = -0.000100;
bm  = -0.008000;
cm  =  0.060;                         % 修正：静稳定力矩
Cm_alpha = am*al.^3 + bm*al.*abs(al) - cm*al;
Cm_el    = -0.280 / d2r;
Cm_q     = -8.0;



% figure;
% plot(missile_pos(:,1), -missile_pos(:,2), 'b-', 'DisplayName','导弹');
% hold on;
% plot(target_pos(:,1), -target_pos(:,2), 'r-', 'DisplayName','目标');
% plot(missile_pos(1,1), -missile_pos(1,2), 'bs', 'MarkerSize',10);
% plot(target_pos(1,1), -target_pos(1,2), 'r^', 'MarkerSize',10);
% legend; grid on; axis equal;
% xlabel('水平距离 (m)'); ylabel('高度 (m)');
% title('弹目轨迹');

% 2D纵向PNG基线参数，非机动目标Miss Distance = 4.1m
% Ka = 0.0180 * ones(9, 6);
% K  = 0.0180 * ones(9, 6);
% KI = 0.0053 * ones(9, 6);
% Kg = 0.0858 * ones(9, 6);
% Anti-Windup = 500





% % AERO_GUID_DAT	Initialization file for missile guidance model
% %
% % See also: AERO_SANIM and Simulink model 'aero_guidance'
% 
% %   J.Hodgson
% %   Copyright 1990-2008 The MathWorks, Inc.
% 
% %==================================================================
% % Useful Constants
% %==================================================================
% 
% d2r     = pi/180;                 % Conversion Deg to Rad
% g       = 9.81;                   % Gravity [m/s/s]
% m2ft    = 3.28084;                % metre to feet
% Kg2slug = 0.0685218;              % Kg to slug
% 
% %==================================================================
% % Atmospheric Constants
% %==================================================================
% 
% T0      = 288.16;                 % Temp. at Sea Level [K]
% rho0    = 1.225;                  % Density [Kg/m^3]
% L       = 0.0065;                 % Lapse Rate [K/m]
% R       = 287.26;                 % Gas Constant J/Kg/K
% gam     = 1.403;                  % Ratio of Specific Heats
% P0      = 101325.0;               % Pressure at Sea Level [N/m^2]
% h_trop  = 11000.0;                % Height of Troposphere [m]
% 
% %==================================================================
% % Missile Configuration
% %==================================================================
% S_ref   = 0.44/m2ft^2;            % Reference area [m^2]
% d_ref   = 0.75/m2ft;              % Reference length [m]
% Iyy     = 182.5/(Kg2slug*m2ft^2); % Inertia
% mass    = 13.98/Kg2slug;          % Mass [Kg]
% Thrust  = 10e3;                   % Thrust [N]
% 
% %==================================================================
% % Missile Aerodynamics
% %==================================================================
% Mach_vec  = 2:0.5:4;              % Reference Mach Numbers
% alpha_vec = (-20:1:20)'*d2r;      % Reference Incidence Values [rad]
% [M,al]=meshgrid(Mach_vec,alpha_vec/d2r);
% 
% % Axial Force Coefficient
% Cx_alpha = -0.3*ones(length(alpha_vec),length(Mach_vec));
% 
% % Normal Force Coefficient
% an    =	 0.000103;                % [Deg^-3]
% bn    =	-0.009450; 		  % [Deg^-2]
% cn    = -0.169600;		  % [Deg^-1]
% Cz_alpha = an*al.^3 + bn*al.*abs(al) + cn*(2-M/3).*al;
% Cz_el = -0.034000/d2r;	
% 
% % Moment Coefficient
% am       = 0.000215;              % [Deg^-3]
% bm       =-0.019500;              % [Deg^-2]
% cm       = 0.051000;              % [Deg^-1]
% Cm_alpha = am*al.^3 + bm*al.*abs(al) - cm*(7-8*M/3).*al;
% Cm_el    = -0.206000/d2r;
% Cm_q     = -1.719;
% 
% %==================================================================
% % Define Initial Conditions
% %==================================================================
% x_ini      = 0;		        % Initial downrange position [m]
% h_ini      = 10000/m2ft;        % Initial altitude [m]
% v_ini      = 3*328;		% Initial velocity [m/s]
% alpha_ini  = 0*d2r;		% Initial incidence [rad]
% theta_ini  = 0*d2r;		% Initial Body Attitude [rad]
% q_ini      = 0*d2r;		% Initial pitch rate [rad/sec]
% 
% %==================================================================
% % Define Target 
% %==================================================================
% pos_tgt   = [4500+x_ini -h_ini-500]; % Initial Target position [m]
% v_tgt     = 328;		% Target Velocity [m/s]
% theta_tgt = 180*d2r;		% Target Direction [rad]
% 
% %==================================================================
% % Missile Actuators
% %==================================================================
% wn_fin      = 150.0;            % Actuator Bandwidth [rad/sec]
% z_fin 	    = 0.7;              % Actuator Damping
% fin_act_0   = 0.0;              % Initial Fin Angle [rad]
% fin_max     =  30.0*d2r;        % Max Fin Deflection [rad] 
% fin_min	    = -30.0*d2r;        % Min Fin Deflection [rad]
% fin_maxrate = 500*d2r;          % Max Fin Rate [rad/sec]
% 
% %==================================================================
% % Sensors
% %==================================================================
% l_acc       = 0.5;     % Position of accelerometer ahead of c.g [m]
% 
% %==================================================================
% % Define Homing Head Dynamics
% %==================================================================
% wn_hom	    = 7.0;		% Estimator Bandwidth [rad/sec]
% tors  	    = 0.05;		% Tracking Loop Time Constant [sec]
% max_gimbal  = 35*d2r;		% Maximum Gimbal Angle [rad]
% min_gimbal  = -35*d2r;		% Minimum Gimbal Angle [rad]
% wgyro       = 100*2*pi;		% Rate gyro bandwidth	[rad/sec]
% Ks          = wgyro/5;		% Rate Loop Bandwidth [rad/sec]
% K_r         = -0.02;		% Radome Aberration
% Beamwidth   = 10*d2r;		% Radar Beam Width [rad]
% 
% %==================================================================
% % Load Autopilot Gains
% %==================================================================
% load aero_guid_autop	        % Gain Scheduled autopilot gains
% max_acc = 40*9.81;              % Maximum demanded acceleration




