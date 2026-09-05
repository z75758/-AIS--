# -*- coding: utf-8 -*-
"""
伶仃护航 —— 受限水域 AIS-视觉融合碰撞风险预警 · 演示模拟器（最小可跑骨架）

运行方式：
    pip install PySide6 matplotlib numpy
    python simulator.py

说明：
    1. 坐标系：局部平面（米），x=东，y=北；航向从正北顺时针（度）。
    2. 本船（蓝色商货船）用滑杆控制航向/航速；渔船由状态机自动生成机动轨迹。
    3. 本船周围虚线椭圆 = 船舶领域边界；渔船为黑色，其 CRI 标签/轨迹颜色 = 碰撞危险度（绿/黄/红）。
    4. 渔船进入红区且 TCPA < t_c 时，判定"碰撞不可避免"，弹窗提示损失最小化建议。
"""
import math
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse, Polygon as MplPolygon, Rectangle, Patch
from matplotlib.lines import Line2D

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QSlider, QMessageBox,
                               QGroupBox)
from PySide6.QtCore import QTimer, Qt

# ======================= 全局参数 =======================
SIM_DT = 0.25            # 每帧仿真步长（秒）
TIMER_MS = 40            # 定时器间隔（毫秒），约 6x 快进
CHW = 250.0              # 航道半宽（米）
YMIN, YMAX = -1500.0, 1500.0

# CRI 参数（米 / 秒）
W_DCPA, W_TCPA, W_D = 0.4, 0.3, 0.3
DCPA_D1, DCPA_D2 = 500.0, 1200.0
TCPA_T1, TCPA_T2 = 60.0, 240.0
D_LOW, D_HIGH = 800.0, 4000.0

# 不可避免判定阈值
DCPA_SAFE = 450.0
T_C = 45.0

# 搁浅判定阈值（距航道边界时间，秒）
T_GROUND = 30.0

# 配色（简洁航海图风格）
WATER = '#dcebf7'      # 航道水域（浅蓝）
LAND = '#e9e1c6'       # 两侧浅滩/岸（米黄）
BANK = '#a08d6b'       # 岸线
OWN = '#1565c0'        # 本船（蓝）
FISH = '#111111'       # 渔船（黑）


def angle_diff(a, b):
    """返回 a-b，规范到 [-180, 180)。"""
    return (a - b + 180.0) % 360.0 - 180.0


_COMPASS = [(0, "北"), (45, "东北"), (90, "东"), (135, "东南"),
            (180, "南"), (225, "西南"), (270, "西"), (315, "西北")]


def compass_point(h):
    """把航向（0~360）转成 8 方位中文名。"""
    h = h % 360
    return min(_COMPASS, key=lambda p: min(abs(h - p[0]), 360 - abs(h - p[0])))[1]


# ======================= 模块 D：风险计算 =======================
def compute_dcpa_tcpa(own, target):
    """向量形式的 DCPA/TCPA（比 sin/cos 形式更稳健）。"""
    dx = target.x - own.x
    dy = target.y - own.y
    ovx, ovy = own.velocity_vec()
    tvx, tvy = target.velocity_vec()
    vrx = tvx - ovx
    vry = tvy - ovy
    vr2 = vrx * vrx + vry * vry
    if vr2 < 1e-6:                       # 无相对运动
        return math.hypot(dx, dy), float('inf')
    t_cpa = -(dx * vrx + dy * vry) / vr2
    cpa_x = dx + t_cpa * vrx
    cpa_y = dy + t_cpa * vry
    dcpa = math.hypot(cpa_x, cpa_y)
    return dcpa, t_cpa


def _dcpa_mu(dcpa):
    if dcpa <= DCPA_D1:
        return 1.0
    if dcpa >= DCPA_D2:
        return 0.0
    return 0.5 - 0.5 * math.sin(math.pi / (DCPA_D2 - DCPA_D1) * (dcpa - (DCPA_D1 + DCPA_D2) / 2.0))


def _tcpa_mu(tcpa):
    if tcpa < 0:        # 已过最近会遇点、正在远离，无风险
        return 0.0
    if tcpa <= TCPA_T1:
        return 1.0
    if tcpa >= TCPA_T2:
        return 0.0
    return ((TCPA_T2 - tcpa) / (TCPA_T2 - TCPA_T1)) ** 2


def _d_mu(D):
    if D <= D_LOW:
        return 1.0
    if D >= D_HIGH:
        return 0.0
    return (D_HIGH - D) / (D_HIGH - D_LOW)


def compute_cri(own, target):
    dcpa, tcpa = compute_dcpa_tcpa(own, target)
    D = math.hypot(target.x - own.x, target.y - own.y)
    return W_DCPA * _dcpa_mu(dcpa) + W_TCPA * _tcpa_mu(tcpa) + W_D * _d_mu(D)


def classify_zone(cri):
    if cri < 0.3:
        return 'green'
    if cri < 0.5:
        return 'yellow'
    return 'red'


def is_unavoidable(own, target):
    dcpa, tcpa = compute_dcpa_tcpa(own, target)
    return dcpa < DCPA_SAFE and 0.0 < tcpa < T_C


def compute_grounding_time(own):
    """航道识别：预测本船距冲出航道边界（触浅滩/触岸）的时间（秒）。
    无横向速度返回 inf；已在岸外返回 0。"""
    vx = own.speed * math.sin(math.radians(own.heading))
    if abs(vx) < 1e-3:
        return float('inf')
    dist = (CHW - own.x) if vx > 0 else (own.x + CHW)
    if dist <= 0:
        return 0.0
    return dist / abs(vx)


# ======================= 模块 B：船舶运动学 =======================
class Ship:
    def __init__(self, x, y, heading, speed, length, max_turn_rate=None):
        self.x = x
        self.y = y
        self.heading = heading          # 度，从正北顺时针
        self.speed = speed              # 米/秒
        self.length = length            # 米
        self.turn_rate = 0.0            # 度/秒
        self.max_turn_rate = max_turn_rate
        self.target_heading = heading   # 本船用（滑杆控制）
        self.history = [(x, y)]

    def velocity_vec(self):
        rad = math.radians(self.heading)
        return (self.speed * math.sin(rad), self.speed * math.cos(rad))

    def step(self, dt):
        self.heading += self.turn_rate * dt
        rad = math.radians(self.heading)
        self.x += self.speed * math.sin(rad) * dt
        self.y += self.speed * math.cos(rad) * dt
        self.history.append((self.x, self.y))
        if len(self.history) > 2000:    # 防止长时间运行内存膨胀
            del self.history[:1000]


# ======================= 模块 C：渔船状态机 =======================
class FishingBoat(Ship):
    STEADY, CROSS, STOP, TURN = 'steady', 'cross', 'stop', 'turn'

    def __init__(self, x, y, heading, speed, length, rng):
        super().__init__(x, y, heading, speed, length)
        self.rng = rng
        self.base_speed = speed
        self.target_speed = speed
        self.state = self.STEADY
        self.state_t = self._sample_duration()
        self.cri = 0.0
        self.zone = 'green'
        self.dcpa = None
        self.tcpa = None
        self.alerted = False

    def _sample_duration(self):
        return self.rng.uniform(10.0, 25.0)

    def _enter_state(self):
        if self.state == self.STEADY:
            self.turn_rate = 0.0
            self.target_speed = self.base_speed
        elif self.state == self.CROSS:
            # 横穿航道：朝对岸（东岸/西岸）走
            self.heading = 90.0 if self.x < 0 else 270.0
            self.turn_rate = 0.0
            self.target_speed = self.base_speed * 1.3
        elif self.state == self.STOP:
            self.turn_rate = 0.0
            self.target_speed = 0.0
        elif self.state == self.TURN:
            self.turn_rate = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(15.0, 30.0)
            self.target_speed = self.base_speed * 0.8
        self.state_t = self._sample_duration()

    def _transition(self):
        r = self.rng.random()
        if self.state == self.STEADY:
            if r < 0.30:
                self.state = self.TURN
            elif r < 0.60:
                self.state = self.STOP
            elif r < 0.80:
                self.state = self.CROSS
            else:
                self.state = self.STEADY
        elif self.state == self.CROSS:
            self.state = self.STEADY if r < 0.7 else self.STOP
        elif self.state == self.STOP:
            self.state = self.STEADY if r < 0.6 else self.CROSS
        elif self.state == self.TURN:
            self.state = self.STEADY if r < 0.6 else self.CROSS
        self._enter_state()

    def update(self, dt):
        self.state_t -= dt
        if self.state_t <= 0:
            self._transition()
        # 速度向目标速度平滑过渡（限加速度 2 m/s²）
        dv = self.target_speed - self.speed
        max_dv = 2.0 * dt
        self.speed += max(-max_dv, min(max_dv, dv))
        self.step(dt)


# ======================= 模块 F：绘制 =======================
def ship_vertices(x, y, heading, length, width):
    """船形五边形顶点（尖艏、收尾），更接近真实船体轮廓。"""
    rad = math.radians(heading)
    fx, fy = math.sin(rad), math.cos(rad)     # 船艏方向
    rx, ry = math.cos(rad), -math.sin(rad)    # 右舷方向
    hL, hW = length / 2.0, width / 2.0
    bow = (x + fx * hL, y + fy * hL)
    fwd_l = (x + fx * hL * 0.35 + rx * hW * 0.75, y + fy * hL * 0.35 + ry * hW * 0.75)
    fwd_r = (x + fx * hL * 0.35 - rx * hW * 0.75, y + fy * hL * 0.35 - ry * hW * 0.75)
    aft_l = (x - fx * hL + rx * hW * 0.55, y - fy * hL + ry * hW * 0.55)
    aft_r = (x - fx * hL - rx * hW * 0.55, y - fy * hL - ry * hW * 0.55)
    return [bow, fwd_l, aft_l, aft_r, fwd_r]


def draw_ship(ax, x, y, heading, length, color, edge='white', lw=0.7):
    L = max(length, 40.0)      # 小船放大保证可见
    W = L * 0.32
    verts = ship_vertices(x, y, heading, L, W)
    ax.add_patch(MplPolygon(verts, closed=True, fc=color, ec=edge, lw=lw, zorder=6))


def draw_domain(ax, ship):
    """船舶领域：半透明填充 + 柔和虚线边界，不抢视线。
    危险等级由渔船颜色（CRI）体现，避免"颜色=距离"与"颜色=危险度"两套口径打架。"""
    L = ship.length
    e = Ellipse((ship.x, ship.y), 6.0 * L, 1.6 * L, angle=90.0 - ship.heading,
                fc=OWN, alpha=0.08, ec=OWN, lw=1.0, ls=(0, (5, 5)), zorder=3)
    ax.add_patch(e)


# ======================= 模块 H：主窗口 =======================
class MainWindow(QMainWindow):
    ZONE_COLOR = {'green': '#2ca02c', 'yellow': '#f0a500', 'red': '#d62728'}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("伶仃护航 · 碰撞风险预警演示模拟器")
        self.resize(1100, 760)

        # 画布
        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 控制面板
        panel = QWidget()
        panel.setFixedWidth(240)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        ctrl_box = QGroupBox("本船控制")
        ctrl = QVBoxLayout(ctrl_box)
        ctrl.addWidget(QLabel("航速"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 15)
        self.speed_slider.setValue(8)
        self.speed_label = QLabel("8 m/s")
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} m/s"))
        ctrl.addWidget(self.speed_slider)
        ctrl.addWidget(self.speed_label)

        ctrl.addSpacing(6)
        ctrl.addWidget(QLabel("航向（0=正北，左=西 右=东）"))
        self.heading_slider = QSlider(Qt.Horizontal)
        self.heading_slider.setRange(-180, 180)
        self.heading_slider.setValue(0)
        self.heading_label = QLabel("0° (北)")
        self.heading_label.setAlignment(Qt.AlignCenter)
        self.heading_slider.valueChanged.connect(self._on_heading_changed)
        ctrl.addWidget(self.heading_slider)
        ctrl.addWidget(self.heading_label)
        panel_layout.addWidget(ctrl_box)

        status_box = QGroupBox("实时状态")
        status = QVBoxLayout(status_box)
        self.status_label = QLabel("—")
        status.addWidget(self.status_label)
        panel_layout.addWidget(status_box)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.reset_btn)
        panel_layout.addLayout(btn_row)

        panel_layout.addStretch(1)
        panel_layout.addWidget(QLabel("<i>系统只告警、只建议，\n不接管船舶自动舵/主机</i>"))

        # 布局
        central = QWidget()
        h = QHBoxLayout(central)
        h.addWidget(self.canvas, stretch=1)
        h.addWidget(panel)
        self.setCentralWidget(central)

        # 定时器
        self.timer = QTimer(self)
        self.timer.setInterval(TIMER_MS)
        self.timer.timeout.connect(self.on_timer)

        self.running = True
        self.ground_alerted = False
        self.time = 0.0
        self.t_ground = float('inf')
        self.ground_danger = False
        self._setup_scenario()
        self.timer.start()

    # ---------- 场景 ----------
    def _setup_scenario(self):
        self.own = Ship(0.0, -1000.0, 0.0, 8.0, 150.0, max_turn_rate=3.0)
        rngA = np.random.RandomState(1)
        rngB = np.random.RandomState(2)
        self.boats = [
            FishingBoat(0.0, 300.0, 180.0, 3.0, 12.0, rngA),   # 对遇（迎面）
            FishingBoat(120.0, -300.0, 270.0, 2.0, 10.0, rngB),  # 横穿
        ]
        self.ground_alerted = False
        self.time = 0.0
        self.t_ground = float('inf')
        self.ground_danger = False
        self.speed_slider.setValue(int(self.own.speed))
        self.heading_slider.setValue(int(self.own.heading))

    def _on_heading_changed(self, v):
        h = v % 360
        self.heading_label.setText(f"{h}° ({compass_point(h)})")

    def _update_status(self):
        hdg = self.own.heading % 360.0
        mx = max((b.cri for b in self.boats), default=0.0)
        if self.ground_danger:
            risk = "⚠ 搁浅风险"
        elif mx >= 0.5:
            risk = "● 碰撞高风险"
        elif mx >= 0.3:
            risk = "● 警戒"
        else:
            risk = "● 正常"
        self.status_label.setText(
            f"t = {self.time:.0f} s\n"
            f"航速 = {self.own.speed:.0f} m/s\n"
            f"航向 = {hdg:.0f}° {compass_point(hdg)}\n"
            f"状态 = {risk}")

    # ---------- 定时器主循环 ----------
    def on_timer(self):
        if not self.running:
            return
        self.time += SIM_DT
        # 本船：读滑杆，转向目标航向（限最大转弯率）
        self.own.speed = float(self.speed_slider.value())
        self.own.target_heading = float(self.heading_slider.value()) % 360.0
        dh = angle_diff(self.own.target_heading, self.own.heading)
        max_turn = self.own.max_turn_rate
        self.own.turn_rate = max(-max_turn, min(max_turn, dh / SIM_DT))
        self.own.step(SIM_DT)

        # 渔船：状态机更新
        for b in self.boats:
            b.update(SIM_DT)

        self._check_risk()
        self._update_status()
        self.render()

    # ---------- 风险判定 + 弹窗 ----------
    def _check_risk(self):
        for b in self.boats:
            b.dcpa, b.tcpa = compute_dcpa_tcpa(self.own, b)
            b.cri = compute_cri(self.own, b)
            b.zone = classify_zone(b.cri)

        # 航道识别：搁浅风险
        self.t_ground = compute_grounding_time(self.own)
        self.ground_danger = self.t_ground < T_GROUND

        # 搁浅告警（优先）。同一段危险只弹一次，危险解除后自动重新武装
        if self.ground_danger:
            if not self.ground_alerted:
                self.ground_alerted = True
                self._trigger_grounding()
            return
        self.ground_alerted = False

        # 碰撞告警：每艘渔船独立计数，危险解除后自动重新武装
        for b in self.boats:
            if is_unavoidable(self.own, b):
                if not b.alerted:
                    b.alerted = True
                    self._trigger_unavoidable(b)
                    return
            else:
                b.alerted = False

    def _trigger_unavoidable(self, b):
        self.running = False
        self.timer.stop()
        self.pause_btn.setText("继续")
        QMessageBox.warning(self, "碰撞不可避免", self._loss_min_suggestion(b))

    def _loss_min_suggestion(self, b):
        lines = [
            f"检测到目标渔船与本船即将碰撞（DCPA≈{b.dcpa:.0f} m，TCPA≈{b.tcpa:.0f} s）。",
            "",
            "已无法通过正常避让避免碰撞，建议采取损失最小化操纵：",
            "1. 立即减速至最小操纵航速（撞击动能 E=½mv²，减速最有效）；",
            "2. 尽量以船首受撞，避免船侧破舱漏油；",
        ]
        if self.own.x > CHW - 200:
            lines.append("3. 注意：右前方有浅滩，避免向右转向，防止搁浅。")
        else:
            lines.append("3. 控制转向，避开浅滩方向，防止搁浅。")
        lines.append("4. 拉响警报，通知渔船人员。")
        return "\n".join(lines)

    def _trigger_grounding(self):
        self.running = False
        self.timer.stop()
        self.pause_btn.setText("继续")
        msg = (f"警告：本船即将冲出航道 / 触浅滩（约 {self.t_ground:.0f} s）！\n\n"
               "建议立即转向回到航道中心，并减速。\n"
               "避免为避让渔船而把自己开上浅滩。")
        QMessageBox.warning(self, "搁浅风险", msg)

    # ---------- 渲染 ----------
    def render(self):
        ax = self.ax
        ax.clear()

        # —— 背景：两侧浅滩/岸（米黄）+ 航道水域（浅蓝） ——
        xl, xr = -450.0, CHW + 300
        yl, yu = YMIN - 100, YMAX + 100
        ax.set_facecolor(LAND)
        ax.add_patch(Rectangle((-CHW, yl), 2 * CHW, yu - yl, fc=WATER, ec='none', zorder=0))

        # 岸线 + 浅滩标注
        ax.plot([-CHW, -CHW], [YMIN, YMAX], color=BANK, lw=1.5, zorder=1)
        ax.plot([CHW, CHW], [YMIN, YMAX], color=BANK, lw=1.5, zorder=1)
        ax.text(-CHW, YMAX - 40, "浅滩", ha='right', fontsize=9, color=BANK)
        ax.text(CHW, YMAX - 40, "浅滩", ha='left', fontsize=9, color=BANK)

        # 搁浅威胁侧高亮（识别即将触到的岸线）
        if self.ground_danger:
            vx = self.own.speed * math.sin(math.radians(self.own.heading))
            bx = CHW if vx > 0 else -CHW
            ax.plot([bx, bx], [YMIN, YMAX], color='#d62728', lw=3, alpha=0.8, zorder=2)

        # 本船领域 + 本船
        draw_domain(ax, self.own)
        draw_ship(ax, self.own.x, self.own.y, self.own.heading, self.own.length, OWN)
        ax.text(self.own.x, self.own.y + 130, "本船", ha='center', fontsize=9, color=OWN, fontweight='bold')

        # 航道识别：搁浅风险提示
        if self.ground_danger:
            ax.text(self.own.x, self.own.y - 150, f"⚠ 距搁浅 {self.t_ground:.0f} s", ha='center',
                    fontsize=11, color='red', fontweight='bold')
        elif math.isfinite(self.t_ground) and self.t_ground < 120:
            ax.text(self.own.x, self.own.y - 150, f"距航道边界 {self.t_ground:.0f} s", ha='center',
                    fontsize=8, color='orange')

        # 渔船 + 轨迹 + 风险标签
        for b in self.boats:
            color = self.ZONE_COLOR[b.zone]
            trail = b.history[-60:]
            if len(trail) > 1:
                ax.plot([p[0] for p in trail], [p[1] for p in trail], color=color, lw=1, alpha=0.4, zorder=2)
            draw_ship(ax, b.x, b.y, b.heading, b.length, FISH, edge='none')
            ax.text(b.x, b.y + 40, f"渔船  CRI={b.cri:.2f}", ha='center', fontsize=8, color=color)
            # 碰撞倒计时（与搁浅倒计时同款）：真正会逼近（DCPA 过小）时显示距碰撞秒数
            if b.dcpa is not None and b.tcpa is not None and b.dcpa < DCPA_SAFE and 0 < b.tcpa:
                if b.tcpa < T_C:
                    ax.text(b.x, b.y - 40, f"⚠ 距碰撞 {b.tcpa:.0f} s", ha='center',
                            fontsize=11, color='red', fontweight='bold')
                elif b.tcpa < 120:
                    ax.text(b.x, b.y - 40, f"距碰撞 {b.tcpa:.0f} s", ha='center',
                            fontsize=8, color='orange')

        # 图例
        handles = [
            Patch(facecolor=OWN, label='本船（商货船）'),
            Patch(facecolor=FISH, label='渔船'),
            Line2D([0], [0], color=OWN, lw=1, ls=(0, (5, 5)), label='本船船舶领域'),
            Patch(facecolor=self.ZONE_COLOR['green'], label='CRI 低风险'),
            Patch(facecolor=self.ZONE_COLOR['yellow'], label='CRI 警戒'),
            Patch(facecolor=self.ZONE_COLOR['red'], label='CRI 高警戒'),
        ]
        ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.9,
                  edgecolor='#cccccc', borderpad=0.6, handlelength=1.5)

        # 坐标轴
        ax.set_xlim(xl, xr)
        ax.set_ylim(yl, yu)
        ax.set_aspect('equal')
        ax.set_title("伶仃航道碰撞风险预警演示", fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        self.canvas.draw_idle()

    # ---------- 按钮回调 ----------
    def _toggle_pause(self):
        self.running = not self.running
        if self.running:
            self.timer.start()
            self.pause_btn.setText("暂停")
        else:
            self.timer.stop()
            self.pause_btn.setText("继续")

    def _reset(self):
        self._setup_scenario()
        self.running = True
        self.timer.start()
        self.pause_btn.setText("暂停")
        self.render()


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
