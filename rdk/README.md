# 板端部署 · RDK X5

目标：把「一首歌 → 一段舞」在板子上跑到 50 Hz，并且能在现场换成任意一首歌即兴跳。

## 1. 物料清单（BOM）

| 物料 | 数量 | 用途 | 备注 |
|---|---|---|---|
| RDK X5 开发板（8 GB） | 1 | 主控 | 现场借用；X3 也能跑（算力更紧） |
| 16 路总线舵机 STS3215 | 16 | 关节 | 也可用 12 路最小版（去掉腰/头/踝） |
| 总线舵机驱动板 / USB-TTL | 1 | 与舵机通信 | CH340 亦可，1 Mbps |
| 7.4 V 5 A 锂电 + 5 V/5 A 降压 | 1 | 舵机与板子供电 | 舵机和板子**必须分开供电**，共地 |
| 3D 打印结构件 | 1 套 | 本体 | PLA/PETG，48 h 现场打印 |
| USB 麦克风 / I2S MEMS 麦 | 1 | 听音乐 | 板载无麦则必须外接 |
| USB 摄像头（可选） | 1 | 姿态跟随 | 第二阶段用 |

## 2. 接线与供电

```
 7.4V 电池 ──┬── 舵机驱动板 V+ ── 16× STS3215 串联总线
             └── 5V/5A 降压 ── RDK X5 (USB-C PD 亦可)
                              └── USB-TTL ── 舵机总线 (A/B)
 共地：电池负极 ── 舵机板 GND ── RDK GND ── TTL GND
```

- 舵机总电流峰值 ≈ 16 × 1.2 A ≈ 19 A，**不要**从板子的 5 V 取电。
- 第一次上电先跑 `python rdk/servo_bus.py`，它会 ping 1–16 号舵机。
- 舵机 ID 建议：1 腰、2 头、3–5 左臂、6–8 右臂、9–12 左腿、13–16 右腿。

## 3. 安装

```bash
# 板子上（Ubuntu 22.04 + hobot 源已配好）
sudo apt update && sudo apt install -y python3-pip python3-numpy alsa-utils
pip3 install pyserial
git clone <your-repo> && cd beatdancer
```

## 4. 跑起来

```bash
# A. 回放已经编好的舞（最稳，适合路演）
python3 -m rdk.dance_service --mode replay --table dist/servo_table.csv --port /dev/ttyUSB0

# B. 现场即兴：麦克风听歌，按小节换动作
python3 -m rdk.dance_service --mode live

# C. 没有硬件时验证逻辑（笔记本上也能跑）
python3 -m rdk.dance_service --mode replay --table dist/servo_table.csv --dry-run --speed 10
python3 -m rdk.beat_frontend --wav assets/demo_track.wav
```

## 5. 延迟预算（X5 CPU，单核）

| 环节 | 耗时 | 说明 |
|---|---|---|
| 采集缓冲 | 10 ms | `arecord` 10 ms 一块 |
| STFT + 梅尔 + onset | ≈ 1.2 ms | 512 点 FFT，40 带 |
| 速度估计（每 200 ms） | ≈ 0.8 ms | 8 s 窗口自相关 |
| 编排查询（查表/即兴） | < 0.1 ms | 预计算或按小节选 |
| 舵机 SYNC_WRITE | ≈ 1.0 ms | 100 字节 @ 1 Mbps |
| **合计** | **≈ 13 ms** | 20 ms 帧内留 35% 余量 |

> 表里是设计预算，**上板后需要用 `dance_service` 打印的真实时间戳复核**。
> 仓库里没有把未验证的数字说成实测值。

## 6. 安全

- 上电第一件事是 `torque(True)` 之前先读位置，超行程的关节不使能。
- `Ctrl-C` 会先把所有关节回到 rest 姿态再断力矩（见 `dance_service.py`）。
- 每帧角速度限幅 0.30 rad，对应约 15 rad/s，避免舵机堵转发热。
- 断拍 2 秒自动回安全姿态（`--mode live` 看门狗）。

## 7. 后续：把节拍模型搬上 BPU

现在的节拍分析跑在 CPU 上（已经够快）。如果要做「摄像头 + 姿态跟随」，
建议把 YOLO-Pose 走 BPU：

```bash
# 参考 D-Robotics/rdk_model_zoo 的 yolov8 pose 示例
hb_mapper checker --model-type onnx --march bayes-e --model yolov8n-pose.onnx
hb_mapper makertbin --config rdk/hb_mapper_beatnet.json
```

`hb_mapper_beatnet.json` 是一份给「节拍 CNN」用的量化模板（40×64 log-mel 输入），
留作第二阶段把 `beat_this` 蒸馏成小模型时使用。
