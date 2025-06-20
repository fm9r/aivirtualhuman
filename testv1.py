import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import json
from datetime import datetime
import re

def extract_thinking(text):
    """提取 <think> 标签中的内容"""
    # 使用正则表达式匹配 <think> 标签内的内容
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # 返回think标签内的内容，去除首尾空白
        return match.group(1).strip()
    else:
        # 如果没有找到think标签，返回原文本
        return text.strip()

def extract_answer(text):
    """提取 <answer> 标签中的内容"""
    # 使用正则表达式匹配 <answer> 标签内的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # 返回answer标签内的内容，去除首尾空白
        return match.group(1).strip()
    else:
        # 如果没有找到answer标签，返回原文本
        return text.strip()

def load_model():
    print("开始下载模型和处理器（这可能需要几分钟时间）...")
    try:
        # 加载模型和处理器
        model_path = "OpenGVLab/VideoChat-R1_7B_caption"
        
        # 使用正确的模型类
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        print("模型加载完成！")
        return model, processor
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

def get_video_files(folder_path):
    """获取文件夹中的所有视频文件"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        # 递归搜索所有子文件夹
        pattern = os.path.join(folder_path, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return video_files

def process_video(video_path, model, processor):
    try:
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")
        
        print(f"开始处理视频: {video_path}")
        
        question = "请按照以下格式分析视频中主持人的手势变化：'在视频中，主持人手势的变化主要体现在以下几个关键动作转变上：1. 从[起始动作]到[结束动作]，表示[含义/目的]；2. 从[起始动作]到[结束动作]，[作用/效果]；3. 从[起始动作]到[结束动作]，[意图说明]。这些手势变化的整体作用是[总结]。'"
        ## 修改question


        # 使用正确的消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": f'"{question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags"'},
                ],
            }
        ]
        
        print("  准备推理...")
        # 应用聊天模板
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理视觉信息
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # 如果有GPU，将输入移到GPU
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        print("生成视频描述...")
        # 生成描述
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        # 提取thinking部分的内容
        full_response = output_text[0]
        thinking_only = extract_thinking(full_response)
        
        print(f"  原始回答长度: {len(full_response)} 字符")
        print(f"  提取thinking后长度: {len(thinking_only)} 字符")
        
        return thinking_only
    except Exception as e:
        print(f"  处理视频时出错: {str(e)}")
        return f"错误: {str(e)}"

def save_results(results, output_file):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")

def batch_process_videos(video_folder, output_file=None):
    """批量处理视频文件"""
    print(f"扫描文件夹: {video_folder}")
    
    # 获取所有视频文件
    video_files = get_video_files(video_folder)
    
    if not video_files:
        print("未找到任何视频文件！")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 加载模型（只需要加载一次）
    print("\n正在加载模型...")
    model, processor = load_model()
    
    # 处理结果
    results = []
    successful_count = 0
    failed_count = 0
    
    print(f"\n开始批量处理 {len(video_files)} 个视频文件...\n")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理文件: {os.path.basename(video_path)}")
        
        try:
            # 处理单个视频
            caption = process_video(video_path, model, processor)
            
            # 保存结果
            result = {
                "video_file": os.path.basename(video_path),
                "video_path": video_path,
                "caption": caption,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            successful_count += 1
            print(f"  ✓ 处理成功\n")
            
        except Exception as e:
            # 处理失败的情况
            result = {
                "video_file": os.path.basename(video_path),
                "video_path": video_path,
                "caption": "",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            failed_count += 1
            print(f"  ✗ 处理失败: {str(e)}\n")
    
    # 保存结果
    if output_file is None:
        output_file = f"video_descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    save_results(results, output_file)
    
    # 打印统计信息
    print(f"\n批量处理完成！")
    print(f"总文件数: {len(video_files)}")
    print(f"成功处理: {successful_count}")
    print(f"处理失败: {failed_count}")
    
    return results

def main():
    # 配置参数
    video_folder = "temp"  # 设置为temp目录中的切片视频
    output_file = "temp_videos_analysis.json"  # 输出文件名
    
    # 您也可以处理单个视频（保留原有功能）
    single_video_mode = False  # 设置为True使用单个视频模式
    
    if single_video_mode:
        # 单个视频处理模式
        video_path = "Abortion_Laws_-_Last_Week_Tonight_with_John_Oliver_HBO-DRauXXz6t0Y.webm/test/214438-00_07_16-00_07_26/214438-00_07_16-00_07_26.mp4"
        
        try:
            print("正在加载模型...")
            model, processor = load_model()
            
            print("正在处理视频...")
            caption = process_video(video_path, model, processor)
            
            print("\n视频描述:")
            print(caption)
        except Exception as e:
            print(f"发生错误: {str(e)}")
    else:
        # 批量处理模式
        try:
            batch_process_videos(video_folder, output_file)
        except Exception as e:
            print(f"批量处理时发生错误: {str(e)}")

if __name__ == "__main__":
    main()
