"""
日志记录工具
"""
import logging
import os
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_dir='./results/logs', name='Cityscapes_Train'):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            name: 日志记录器名称
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 设置最低日志级别
        
        # 清除已有的处理器（避免重复）
        self.logger.handlers.clear()
        
        # 创建文件处理器
        log_file = os.path.join(
            log_dir, 
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"日志记录器已初始化，日志文件: {log_file}")
    
    def get_logger(self):
        """获取日志记录器"""
        return self.logger
    
    def log_config(self, config):
        """记录配置信息"""
        self.logger.info("=" * 60)
        self.logger.info("训练配置信息")
        self.logger.info("=" * 60)
        
        for section, params in config.items():
            self.logger.info(f"[{section}]")
            if isinstance(params, dict):
                for key, value in params.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info(f"  {params}")
            self.logger.info("-" * 40)
    
    def log_epoch(self, epoch, num_epochs, train_loss, val_loss, miou, pixel_acc):
        """记录一个epoch的训练结果"""
        self.logger.info(f"Epoch {epoch+1}/{num_epochs} 结果:")
        self.logger.info(f"  训练损失: {train_loss:.6f}")
        self.logger.info(f"  验证损失: {val_loss:.6f}")
        self.logger.info(f"  验证mIoU: {miou:.6f}")
        self.logger.info(f"  像素准确率: {pixel_acc:.6f}")
        self.logger.info("-" * 40)
    
    def log_best_model(self, epoch, miou, save_path):
        """记录最佳模型保存信息"""
        self.logger.info("✨ 保存最佳模型 ✨")
        self.logger.info(f"  Epoch: {epoch+1}")
        self.logger.info(f"  最佳mIoU: {miou:.6f}")
        self.logger.info(f"  保存路径: {save_path}")
    
    def log_error(self, error_msg):
        """记录错误信息"""
        self.logger.error(f"错误: {error_msg}")
    
    def log_warning(self, warning_msg):
        """记录警告信息"""
        self.logger.warning(f"警告: {warning_msg}")

# 创建全局日志记录器实例
_logger = None

def setup_logger(log_dir=None, name='Cityscapes_Train'):
    """
    设置全局日志记录器
    
    Args:
        log_dir: 日志保存目录，如果为None则使用默认目录
        name: 日志记录器名称
    """
    global _logger
    
    # 如果没有指定log_dir，使用默认路径
    if log_dir is None:
        log_dir = './results/logs'
    
    _logger = Logger(log_dir, name)
    return _logger

def get_logger():
    """获取全局日志记录器"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger.logger