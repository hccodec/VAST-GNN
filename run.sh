#!/bin/bash

# 命令列表
commands=$(cat <<EOF
python main.py --country France,Italy --ydays 1 --shift 2 --result-dir tests_1015 --device 3 --exp 6_missing_nodes_50_mask_tcn_mlp
python main.py --country France,Italy --ydays 1 --shift 6 --result-dir tests_1015 --device 3 --exp 6_missing_nodes_50_mask_tcn_mlp
python main.py --country France,Italy --ydays 1 --shift 13 --result-dir tests_1015 --device 3 --exp 6_missing_nodes_50_mask_tcn_mlp
python main.py --country France,Italy --ydays 1 --shift 2 --result-dir tests_1015 --device 3 --model mpnn_lstm --exp 7_mpnn_lstm_50_mask
python main.py --country France,Italy --ydays 1 --shift 6 --result-dir tests_1015 --device 3 --model mpnn_lstm --exp 7_mpnn_lstm_50_mask
python main.py --country France,Italy --ydays 1 --shift 13 --result-dir tests_1015 --device 3 --model mpnn_lstm --exp 7_mpnn_lstm_50_mask
python main.py --country France,Italy --ydays 1 --shift 2 --result-dir tests_1015 --device 3 --model lstm --exp 8_lstm_50_mask
python main.py --country France,Italy --ydays 1 --shift 6 --result-dir tests_1015 --device 3 --model lstm --exp 8_lstm_50_mask
python main.py --country France,Italy --ydays 1 --shift 13 --result-dir tests_1015 --device 3 --model lstm --exp 8_lstm_50_mask
EOF
)

# tmux会话名称
SESSION_NAME=hbj
WINDOW_NAME="1015"

# 读取所有命令到数组
readarray -t cmd_array <<< "$commands"

# 检查是否已有名为 WINDOW_NAME 的窗口
tmux list-windows -t $SESSION_NAME | grep -q "$WINDOW_NAME"

# 如果窗口不存在，则创建新窗口
if [ $? -ne 0 ]; then
    tmux new-window -t $SESSION_NAME -n "$WINDOW_NAME"
    # 窗口新建时，跳过分割
    first_command=true
else
    first_command=false
fi

# 在新窗口中分割窗格
for i in "${!cmd_array[@]}"; do
  # # 跳过空行
  # if [ -z "$cmd_array[$i]" ]; then
  #   # 暂停，等待用户确认后继续
  #   read -p "Press Enter to continue..."
  #   continue
  # fi

  # 如果窗口已经存在，且不是第一个命令，则进行分割
  if [ "$first_command" = false ]; then
    tmux split-window -v -t "$SESSION_NAME":"$WINDOW_NAME"
  fi
  first_command=false

  # 在每个窗格中执行别名exp来启用实验环境，然后执行列表中的命令
  tmux send-keys -t "$SESSION_NAME":"$WINDOW_NAME" "exp && ${cmd_array[$i]}" C-m
  
  # 调整窗格布局
  tmux select-layout tiled

done

