import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

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

def process_video(video_path, model, processor):
    try:
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")
        
        print(f"开始处理视频: {video_path}")
        
        # question = "请总结视频中**主持人**的**手势**如何**随时间变化**，并指出**关键动作转变**。"
        question = "请按照以下格式分析视频中主持人的手势变化：'在视频中，主持人手势的变化主要体现在以下几个关键动作转变上：1. 从[起始动作]到[结束动作]；2. 从[起始动作]到[结束动作]，[作用/效果]；3. 从[起始动作]到[结束动作]，[意图说明]。'"

        # question = "请识别视频中主持人动作变化的关键时刻，并按时间先后顺序描述这些关键动作节点，包括动作的完整变化过程。"
        # question = "请总结视频中**主持人**的**手势和面部表情**如何**随时间变化**，并指出**关键的表情或动作转变**。"
        # question = "请详细描述主持人动作的完整流程和转换过程，重点说明每个动作是如何从前一个动作自然过渡的，以及动作变化的节奏感。"
        # question = "请按照时间顺序详细描述主持人在视频中的手势和身体动作变化，包括：1) 视频开始时的动作；2) 中间过程的动作转换；3) 结束时的动作。请用'首先...然后...接着...最后...'的时序表达方式。"
        # question = "请总结主持人在视频中不同时刻的手势变化。"
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
        
        print("准备推理...")
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
        
        return output_text[0]
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        raise

def main():
    # 视频路径
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
        raise

if __name__ == "__main__":
    main()
