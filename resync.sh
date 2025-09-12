#!/bin/bash
# sync_to_westb.sh
# 用于同步本地项目到 SeetaCloud 服务器

# 本地目录（当前项目目录）
LOCAL_DIR="/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf_new"

# 远程目录（可以改成你要放置的路径，比如 /root/Plaszyme）
REMOTE_DIR="/root/autodl-tmp/sdf"

# 远程服务器
REMOTE_HOST="root@connect.westb.seetacloud.com"
REMOTE_PORT=11104

# 执行 rsync 同步
rsync -avz --progress \
    -e "ssh -p ${REMOTE_PORT}" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".git/" \
    --exclude ".idea/" \
    "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"