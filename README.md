# AI视频分析项目

这个项目包含了用于视频下载、预处理和AI分析的工具集合。

## 项目结构

- `testv1.py` - 使用Qwen2VL模型对视频进行AI分析和描述
- `video_clip_preprocess.py` - 视频预处理工具，用于裁剪和格式转换
- `main.py` - 主程序入口
- `temp/` - 存放处理后的视频片段
- `download_video/` - 存放下载的原始视频文件
- `temp_videos_analysis.json` - 视频分析结果（temp目录）
- `train_videos_analysis_v3.json` - 训练视频分析结果

## 功能特性

### 1. 视频分析 (testv1.py)
- 使用OpenGVLab/VideoChat-R1_7B_caption模型
- 批量处理视频文件
- 生成结构化的视频描述
- 支持多种视频格式

### 2. 视频预处理 (video_clip_preprocess.py)
- 基于时间码的视频切片
- 标准化视频格式（768x768, 24fps）
- 音频处理（16kHz采样率）
- FFmpeg支持

### 3. 数据处理
- JSON格式的分析结果存储
- 批量文件处理
- 错误处理和日志记录

## 使用方法

### 环境配置
```bash
# 创建conda环境
conda create -n aihuman python=3.10
conda activate aihuman

# 安装依赖
conda install ffmpeg -c conda-forge
pip install torch transformers qwen-vl-utils opencv-python pillow tqdm
```

### 视频分析
```bash
# 修改testv1.py中的video_folder路径
python testv1.py
```

### 视频预处理
```bash
# 修改video_clip_preprocess.py中的参数
python video_clip_preprocess.py
```

## 依赖项

- Python 3.10+
- PyTorch
- Transformers
- OpenCV
- Pillow
- FFmpeg
- Qwen-VL-Utils

## 输出格式

分析结果以JSON格式保存，包含以下字段：
- `video_file` - 视频文件名
- `video_path` - 视频完整路径
- `caption` - AI生成的视频描述
- `status` - 处理状态
- `timestamp` - 处理时间戳

## 注意事项

- 确保有足够的GPU内存运行Qwen2VL模型
- 大文件处理需要充足的磁盘空间
- 建议使用CUDA加速推理过程 