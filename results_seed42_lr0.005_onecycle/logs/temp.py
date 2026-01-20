import re
import sys

def clean_training_log(input_file, output_file=None):
    """
    清理训练日志，删除进度条内容，但保留Epoch信息
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则输出到控制台
    """
    if output_file is None:
        output_file = input_file.replace('.txt', '_cleaned.txt') if input_file.endswith('.txt') else input_file + '_cleaned'
    
    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    skip_next = 0  # 计数器，用于跳过指定行数
    
    for i, line in enumerate(lines):
        # 如果处于跳过模式，减少计数器并跳过当前行
        if skip_next > 0:
            skip_next -= 1
            continue
        
        # 检查是否是Epoch开始行
        if line.strip().startswith('Epoch') and '/' in line:
            # 保留Epoch行
            cleaned_lines.append(line)
            
            # 检查接下来的行是否需要跳过
            # 下一行可能是分隔符
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('==='):
                # 下下行可能是训练进度条
                if i + 2 < len(lines) and ('训练:' in lines[i + 2] or '训练:' in lines[i + 2]):
                    # 再下一行可能是验证进度条
                    if i + 3 < len(lines) and ('验证:' in lines[i + 3] or '验证:' in lines[i + 3]):
                        # 跳过接下来的3行（分隔符、训练进度条、验证进度条）
                        skip_next = 3
            continue
        
        # 检查是否是进度条行（训练或验证）
        if '训练:' in line or '验证:' in line:
            # 检查这一行是否包含进度条格式
            if '100%|' in line or 'it/s' in line:
                continue  # 跳过进度条行
        
        # 检查是否是分隔符行
        if line.strip().startswith('===') and len(line.strip()) > 10:
            # 跳过分隔符行，除非它紧跟在Epoch行后面（我们已经处理了这种情况）
            continue
        
        # 保留其他所有行
        cleaned_lines.append(line)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"日志清理完成！清理前: {len(lines)} 行，清理后: {len(cleaned_lines)} 行")
    print(f"清理后的文件已保存到: {output_file}")
    
    return cleaned_lines

def clean_log_interactive():
    """交互式清理日志"""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("请输入要清理的日志文件路径: ")
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    try:
        cleaned_lines = clean_training_log(input_file, output_file)
        
        # 显示前20行清理后的内容
        print("\n清理后的前20行内容:")
        print("=" * 50)
        for i, line in enumerate(cleaned_lines[:20]):
            print(f"{i+1:3}: {line}", end='')
        print("=" * 50)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    clean_log_interactive()