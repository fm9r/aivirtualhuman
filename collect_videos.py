import os
import shutil
import glob
from pathlib import Path

def collect_train_videos(source_folder, target_folder):
    """
    从源文件夹的各个子文件夹中收集train目录下的视频到目标文件夹
    
    Args:
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
    """
    
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv', '*.wmv']
    
    collected_count = 0
    failed_count = 0
    
    print(f"开始收集视频文件...")
    print(f"源文件夹: {source_folder}")
    print(f"目标文件夹: {target_folder}")
    print("-" * 50)
    
    # 遍历源文件夹下的所有子文件夹
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # 只处理文件夹
        if not os.path.isdir(item_path):
            continue
            
        # 检查是否存在train子文件夹
        train_folder = os.path.join(item_path, 'train')
        if not os.path.exists(train_folder):
            print(f"⚠️  {item}/train 文件夹不存在，跳过")
            continue
            
        print(f"📁 处理文件夹: {item}/train")
        
        # 在train文件夹中查找所有视频文件
        folder_video_count = 0
        for ext in video_extensions:
            pattern = os.path.join(train_folder, '**', ext)
            video_files = glob.glob(pattern, recursive=True)
            
            for video_file in video_files:
                try:
                    # 获取视频文件名
                    video_name = os.path.basename(video_file)
                    
                    # 为了避免文件名冲突，在文件名前加上父文件夹名
                    new_name = f"{item}_{video_name}"
                    target_path = os.path.join(target_folder, new_name)
                    
                    # 如果目标文件已存在，添加数字后缀
                    counter = 1
                    original_target_path = target_path
                    while os.path.exists(target_path):
                        name, ext = os.path.splitext(original_target_path)
                        target_path = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    # 复制文件
                    shutil.copy2(video_file, target_path)
                    print(f"   ✓ 复制: {os.path.basename(video_file)} -> {os.path.basename(target_path)}")
                    
                    collected_count += 1
                    folder_video_count += 1
                    
                except Exception as e:
                    print(f"   ✗ 复制失败: {video_file} - {str(e)}")
                    failed_count += 1
        
        if folder_video_count == 0:
            print(f"   📭 {item}/train 中未找到视频文件")
    
    print("-" * 50)
    print(f"✅ 收集完成!")
    print(f"成功收集: {collected_count} 个视频文件")
    print(f"失败: {failed_count} 个文件")
    print(f"文件已保存到: {target_folder}")

def main():
    # 配置路径
    source_folder = r"C:\Users\Sun\Desktop\fit\aivitrtualhuman\drive-download-20250610T113934Z-1-006"
    target_folder = r"C:\Users\Sun\Desktop\fit\aivitrtualhuman\collected_train_videos"
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"❌ 源文件夹不存在: {source_folder}")
        return
    
    # 开始收集
    collect_train_videos(source_folder, target_folder)

if __name__ == "__main__":
    main() 