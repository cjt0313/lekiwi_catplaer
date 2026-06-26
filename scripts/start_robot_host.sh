#!/bin/bash
# Start the LeKiwi robot host via SSH.
# Requires: sshpass (apt install sshpass)

sshpass -p 'a' ssh -t catplayer@192.168.100.1 \
    'source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/lerobot && echo "" | python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi --robot.cameras={} --host.connection_time_s=600'
