import cv2
import os
import argparse
from tqdm import tqdm

def video_to_frames(video_path, output_dir, prefix='frame', extension='.jpg'):
    """
    将视频文件解析为图像序列（修正上下颠倒问题）
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出图像保存目录
        prefix: 图像文件名前缀
        extension: 图像文件扩展名
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"- 总帧数: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- 分辨率: {width}x{height}")
    
    # 读取并保存每一帧
    frame_count = 0
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 修正图像方向
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            frame = cv2.flip(frame, 0)  # 垂直翻转
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转回BGR
            
            # 生成输出文件名
            output_path = os.path.join(
                output_dir, 
                f"{prefix}_{frame_count:06d}{extension}"
            )
            
            # 保存图像
            cv2.imwrite(output_path, frame)
            
            frame_count += 1
            pbar.update(1)
    
    # 释放资源
    cap.release()
    
    print(f"\n完成! 共处理 {frame_count} 帧")
    print(f"图像保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将视频转换为图像序列")
    parser.add_argument("--video_path", default="/home/selflearning/dataset/BDD100K-MultiTask/video/00067cfb-5443fe39.mov", help="输入视频文件路径")
    parser.add_argument("--output_dir", default="/home/selflearning/dataset/BDD100K-MultiTask/video/00067cfb-5443fe39", 
                        help="输出图像保存目录")
    parser.add_argument("--prefix", default="frame", 
                        help="输出图像文件名前缀")
    parser.add_argument("--extension", default=".jpg", 
                        help="输出图像文件扩展名")
    
    args = parser.parse_args()
    
    video_to_frames(args.video_path, args.output_dir, 
                    args.prefix, args.extension)