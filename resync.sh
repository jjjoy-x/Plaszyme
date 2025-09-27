#!/bin/bash
# sync_to_westb.sh
# 用于同步本地项目到 SeetaCloud 服务器

# 本地目录（当前项目目录）
LOCAL_DIR="/Users/shulei/PycharmProjects/Plaszyme/"

# 远程目录（可以改成你要放置的路径，比如 /root/Plaszyme）
REMOTE_DIR="/gpfs/work/bio/shuleihe23/plaszyme_new/"

# 远程服务器
REMOTE_HOST="shuleihe23@login.hpc.xjtlu.edu.cn"
REMOTE_PORT=22

# 执行 rsync 同步
rsync -avz --progress \
    -e "ssh -p ${REMOTE_PORT}" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".git/" \
    --exclude ".idea/" \
    "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"