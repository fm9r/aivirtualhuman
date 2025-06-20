import os
import json
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import cv2
import numpy as np

def convert_time_to_seconds(time_str):
    """
    将时间字符串转换为秒数
    支持格式: "00:01:30" 或 "90" 或 "1:30"
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    time_str = str(time_str).strip()
    
    # 如果只是数字，直接返回
    if time_str.replace('.', '').isdigit():
        return float(time_str)
    
    # 处理时:分:秒格式
    parts = time_str.split(':')
    if len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:  # MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

def resize_and_pad(frame, target_size=768):
    """
    调整帧大小并添加填充以保持纵横比
    """
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整大小
    resized = cv2.resize(frame, (new_w, new_h))
    
    # 创建目标尺寸的黑色画布
    result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 计算居中位置
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    
    # 将调整后的图像放在画布中心
    result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return result

def clip_video_by_timecode(video_path, start_time, end_time, output_path, 
                          target_fps=24, target_size=768, target_sr=16000):
    """
    根据时间码截取视频片段
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return False
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换时间格式
        start_seconds = convert_time_to_seconds(start_time)
        end_seconds = convert_time_to_seconds(end_time)
        
        print(f"  截取时间: {start_seconds}s - {end_seconds}s")
        
        # 加载视频
        clip = VideoFileClip(video_path)
        
        # 检查时间范围是否有效
        if start_seconds >= clip.duration:
            print(f"  警告: 开始时间 {start_seconds}s 超出视频长度 {clip.duration}s")
            clip.close()
            return False
        
        if end_seconds > clip.duration:
            print(f"  警告: 结束时间调整为视频长度 {clip.duration}s")
            end_seconds = clip.duration
        
        # 截取视频片段
        clipped = clip.subclip(start_seconds, end_seconds)
        
        # 调整帧率
        if target_fps:
            clipped = clipped.set_fps(target_fps)
        
        # 调整音频采样率
        if clipped.audio and target_sr:
            audio = clipped.audio.set_fps(target_sr)
            clipped = clipped.set_audio(audio)
        
        # 如果需要调整视频尺寸
        if target_size:
            def resize_frame(get_frame, t):
                frame = get_frame(t)
                # 转换BGR到RGB (moviepy使用RGB)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # 调整大小并填充
                resized_frame = resize_and_pad(frame_bgr, target_size)
                # 转换回RGB
                return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            clipped = clipped.fl(resize_frame, apply_to=['mask'])
        
        # 保存处理后的视频
        clipped.write_videofile(output_path, 
                               codec='libx264', 
                               audio_codec='aac',
                               temp_audiofile='temp-audio.m4a',
                               remove_temp=True,
                               verbose=False,
                               logger=None)
        
        # 清理内存
        clipped.close()
        clip.close()
        
        print(f"  ✓ 成功保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ 处理失败: {str(e)}")
        return False

def process_videos_from_csv(csv_path, input_dir, output_dir, 
                           target_fps=24, target_size=768, target_sr=16000):
    """
    根据CSV文件批量处理视频
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"读取到 {len(df)} 条记录")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    successful_count = 0
    failed_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理视频"):
        try:
            # 提取信息
            video_url = row['URL']
            start_time = row.get('start_timecode', 0)
            end_time = row.get('end_timecode', None)
            
            # 从URL提取视频ID
            video_id = video_url.split('v=')[1]
            input_video_path = os.path.join(input_dir, f"{video_id}.mp4")
            
            # 构建输出路径
            output_filename = f"{video_id}_{start_time}_{end_time}.mp4"
            output_video_path = os.path.join(output_dir, output_filename)
            
            print(f"[{index+1}/{len(df)}] 处理: {video_id}")
            
            # 如果没有结束时间，跳过
            if pd.isna(end_time) or end_time is None:
                print(f"  跳过: 缺少结束时间")
                failed_count += 1
                continue
            
            # 如果输出文件已存在，跳过
            if os.path.exists(output_video_path):
                print(f"  跳过: 文件已存在")
                successful_count += 1
                continue
            
            # 处理视频
            success = clip_video_by_timecode(
                input_video_path, start_time, end_time, output_video_path,
                target_fps, target_size, target_sr
            )
            
            if success:
                successful_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"  ✗ 记录处理失败: {str(e)}")
            failed_count += 1
    
    print("-" * 50)
    print(f"处理完成!")
    print(f"成功: {successful_count}")
    print(f"失败: {failed_count}")
    print(f"总计: {len(df)}")

def main():
    # 配置参数
    csv_file = "echomimic_v2-main/EMTD_dataset/echomimicv2_benchmark_url+start_timecode+end_timecode.txt"
    input_directory = "download_video"
    output_directory = "processed_clips"
    
    # 处理参数
    target_fps = 24        # 目标帧率
    target_size = 768      # 目标尺寸 (768x768)
    target_sr = 16000      # 目标音频采样率
    
    # 检查输入文件
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        return
    
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录不存在: {input_directory}")
        return
    
    # 开始处理
    process_videos_from_csv(
        csv_file, 
        input_directory, 
        output_directory,
        target_fps, 
        target_size, 
        target_sr
    )

if __name__ == "__main__":
    main() 