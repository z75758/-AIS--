# 伶仃护航 —— AIS-视觉融合船舶碰撞风险预警系统

面向珠江口伶仃航道等受限水域，针对"大船有 AIS、小渔船无 AIS"的感知盲区，构建的碰撞风险预警系统。

系统**只告警、只建议，不接管船舶自动舵与主机**，由驾驶员决策执行。

## 目录结构

```
.
├── README.md                  # 本说明
├── 项目书/
│   ├── AIS视觉融合船舶碰撞风险预警-项目书.md       # 项目书源文件（Markdown）
│   ├── AIS视觉融合船舶碰撞风险预警-项目书-v2.docx   # 生成的 Word 版（含公式与参考文献）
│   └── convert_to_docx.py                           # Markdown → Word 转换脚本
└── 模拟器/
    ├── simulator.py           # 演示模拟器（PySide6 + matplotlib）
    └── requirements.txt       # 模拟器依赖
```

## 模拟器运行

```bash
pip install -r requirements.txt
python simulator.py
```

功能说明：

- 本船（蓝色商货船）用滑杆控制航向/航速；
- 渔船由**状态机**自动机动（直行 / 横穿 / 停船收网 / 急转）；
- 本船周围虚线椭圆 = **船舶领域边界**；渔船颜色 = **碰撞危险度（绿 / 黄 / 红，含轨迹预测）**；
- 渔船进入红区且 `TCPA < 45s` 时，弹窗提示"碰撞不可避免 + 损失最小化建议"。

## 项目书转 Word

`convert_to_docx.py` 使用 `python-docx` 将 Markdown 项目书转为 Word。

```bash
pip install python-docx
python convert_to_docx.py
```

> 注意：脚本顶部 `SRC` / `DST` 是绝对路径（原指向 `C:\Users\asus\...`）。项目移动到本目录后，如需重新生成 Word，请先修改这两个路径。
