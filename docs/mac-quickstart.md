# Mac Quick Start

## Network Setup

Connect Mac to robot via USB Ethernet adapter, then set static IP:

```bash
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.100.2 255.255.255.0
```

Verify:
```bash
ping 192.168.100.1
```

Revert to DHCP later:
```bash
sudo networksetup -setdhcp "USB 10/100/1000 LAN"
```

## Start Robot Host (SSH)

```bash
ssh catplayer@192.168.100.1
# on the robot:
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/lerobot
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi --robot.cameras='{}' --host.connection_time_s=600
```

## Run Pipeline

```bash
conda activate lekiwi

# Terminal 1: perception + planning + visualization
python -m perception.viz_detection --localize --target cat --distance 0.3

# Terminal 2: robot bridge
python scripts/robot_bridge.py
```

Open http://localhost:8080 for the viser 3D viewer.

Use the **Robot ON/OFF** and **E-STOP** buttons in the GUI to control the robot.

## Without Robot (visualization only)

```bash
python -m perception.viz_detection --localize --target cat --no-zmq
```
