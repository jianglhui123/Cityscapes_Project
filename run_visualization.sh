#!/bin/bash
# 模型预测可视化脚本

echo "开始模型预测可视化..."
echo "当前时间: $(date)"
echo "工作目录: $(pwd)"

# 设置GPU
export CUDA_VISIBLE_DEVICES=2

# 创建基础目录
BASE_OUTPUT_DIR="./results/plots"
mkdir -p $BASE_OUTPUT_DIR

# 0. 清空或创建报告文件
REPORT_FILE="$BASE_OUTPUT_DIR/visualization_report.txt"
echo "模型预测可视化报告" > $REPORT_FILE
echo "生成时间: $(date)" >> $REPORT_FILE
echo "工作目录: $(pwd)" >> $REPORT_FILE
echo "使用的GPU: $CUDA_VISIBLE_DEVICES" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# 1. 使用最佳模型进行可视化
BEST_MODEL_DIR="$BASE_OUTPUT_DIR/best_model"
mkdir -p $BEST_MODEL_DIR

echo -e "\n1. 使用最佳模型进行可视化..."
echo "输出目录: $BEST_MODEL_DIR"
python visualize_predictions.py \
  --model ./results/weights/best_model.pth \
  --samples 4 \
  --split train \
  --output "$BEST_MODEL_DIR" \
  --evaluate
  

# 统计各个目录的图像文件
echo "最佳模型目录 ($BEST_MODEL_DIR):" >> $REPORT_FILE
if [ -d "$BEST_MODEL_DIR" ]; then
    find "$BEST_MODEL_DIR" -name "*.png" -type f | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  - $(basename "$file") (大小: $size)" >> $REPORT_FILE
    done
    count=$(find "$BEST_MODEL_DIR" -name "*.png" -type f | wc -l)
    echo "  总计: $count 个文件" >> $REPORT_FILE
else
    echo "  目录不存在" >> $REPORT_FILE
fi
echo "" >> $REPORT_FILE

# 2. 使用最终模型进行可视化
FINAL_MODEL_DIR="$BASE_OUTPUT_DIR/final_model"
mkdir -p $FINAL_MODEL_DIR

# echo -e "\n2. 使用最终模型进行可视化..."
# echo "输出目录: $FINAL_MODEL_DIR"
# python visualize_predictions.py \
#   --model ./results/weights/final_model.pth \
#   --samples 4 \
#   --output "$FINAL_MODEL_DIR" \
#   --evaluate

# echo "最终模型目录 ($FINAL_MODEL_DIR):" >> $REPORT_FILE
# if [ -d "$FINAL_MODEL_DIR" ]; then
#     find "$FINAL_MODEL_DIR" -name "*.png" -type f | while read file; do
#         size=$(du -h "$file" | cut -f1)
#         echo "  - $(basename "$file") (大小: $size)" >> $REPORT_FILE
#     done
#     count=$(find "$FINAL_MODEL_DIR" -name "*.png" -type f | wc -l)
#     echo "  总计: $count 个文件" >> $REPORT_FILE
# else
#     echo "  目录不存在" >> $REPORT_FILE
# fi
# echo "" >> $REPORT_FILE

# # 3. 在训练集上可视化
# TRAIN_SET_DIR="$BASE_OUTPUT_DIR/train_set"
# mkdir -p $TRAIN_SET_DIR

# echo -e "\n3. 在训练集上可视化..."
# echo "输出目录: $TRAIN_SET_DIR"
# python visualize_predictions.py \
#   --split train \
#   --samples 3 \
#   --output "$TRAIN_SET_DIR"

# echo "训练集目录 ($TRAIN_SET_DIR):" >> $REPORT_FILE
# if [ -d "$TRAIN_SET_DIR" ]; then
#     find "$TRAIN_SET_DIR" -name "*.png" -type f | while read file; do
#         size=$(du -h "$file" | cut -f1)
#         echo "  - $(basename "$file") (大小: $size)" >> $REPORT_FILE
#     done
#     count=$(find "$TRAIN_SET_DIR" -name "*.png" -type f | wc -l)
#     echo "  总计: $count 个文件" >> $REPORT_FILE
# else
#     echo "  目录不存在" >> $REPORT_FILE
# fi
# echo "" >> $REPORT_FILE


# 6. 打印完成信息
echo -e "\n所有可视化任务完成!"
echo "详细报告已保存到: $REPORT_FILE"