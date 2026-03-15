#!/bin/bash

# 配置路径（Git Bash 风格路径）
SRC_DIR="/d/Projects/Look2Act_Tracker_Project/docs/logs"
DEST_DIR="/d/UESTC/毕业设计/Conference/docs"

# 需要同步的文件名列表
FILES=("experiment_log.md" "paper_notes.md" "research_log.md")

echo "🚀 本地同步开启... 正在监听 $SRC_DIR"

# 获取初始文件的修改时间戳
declare -A LAST_MOD
for f in "${FILES[@]}"; do
    LAST_MOD[$f]=$(stat -c %Y "$SRC_DIR/$f" 2>/dev/null)
done

# 循环监听
while true; do
    for f in "${FILES[@]}"; do
        CURRENT_MOD=$(stat -c %Y "$SRC_DIR/$f" 2>/dev/null)
        
        # 如果当前时间戳与记录的不一致，说明文件被修改了
        if [ "$CURRENT_MOD" != "${LAST_MOD[$f]}" ]; then
            cp "$SRC_DIR/$f" "$DEST_DIR/$f"
            LAST_MOD[$f]=$CURRENT_MOD
            echo "[$(date +%H:%M:%S)] ✅ Synced: $f"
        fi
    done
    sleep 1 # 每秒检查一次，CPU 占用几乎为 0
done