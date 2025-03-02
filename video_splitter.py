import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import datetime, timedelta

class VideoSplitter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口
        self.input_files = []
        self.output_dir = ""
        self.split_duration = 0
        
    def select_input_files(self):
        """选择要处理的视频文件"""
        self.input_files = filedialog.askopenfilenames(
            title="选择要分割的视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("所有文件", "*.*")]
        )
        return len(self.input_files) > 0
        
    def select_output_directory(self):
        """选择输出目录"""
        self.output_dir = filedialog.askdirectory(title="选择输出目录")
        return bool(self.output_dir)
        
    def get_split_duration(self):
        """获取分割时间（分钟）"""
        duration = simpledialog.askinteger(
            "分割时间设置", 
            "请输入每个片段的时长（分钟）:",
            minvalue=1,
            maxvalue=60
        )
        if duration:
            self.split_duration = duration
            return True
        return False
        
    def process_videos(self):
        """处理所有选定的视频"""
        total_files = len(self.input_files)
        processed = 0
        
        for input_file in self.input_files:
            try:
                # 获取视频时长
                duration_cmd = [
                    'ffprobe', 
                    '-v', 'error', 
                    '-show_entries', 'format=duration', 
                    '-of', 'default=noprint_wrappers=1:nokey=1', 
                    input_file
                ]
                
                duration_output = subprocess.check_output(duration_cmd, stderr=subprocess.STDOUT)
                total_duration = float(duration_output.decode('utf-8').strip())
                
                # 计算需要分割的片段数
                split_seconds = self.split_duration * 60
                segments = int(total_duration / split_seconds) + (1 if total_duration % split_seconds > 0 else 0)
                
                # 获取文件名（不含扩展名）
                file_name = os.path.basename(input_file)
                name_without_ext = os.path.splitext(file_name)[0]
                
                # 为每个视频创建子目录
                video_output_dir = os.path.join(self.output_dir, name_without_ext)
                os.makedirs(video_output_dir, exist_ok=True)
                
                # 分割视频
                for i in range(segments):
                    start_time = i * split_seconds
                    output_file = os.path.join(video_output_dir, f"{name_without_ext}_part{i+1}.mp4")
                    
                    # 构建FFmpeg命令
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-i', input_file,
                        '-ss', str(start_time),
                        '-t', str(split_seconds),
                        '-c', 'copy',  # 复制编解码器，不重新编码
                        '-avoid_negative_ts', '1',
                        output_file
                    ]
                    
                    # 执行FFmpeg命令
                    subprocess.run(ffmpeg_cmd, check=True)
                
                processed += 1
                
            except Exception as e:
                messagebox.showerror("错误", f"处理文件 {input_file} 时出错:\n{str(e)}")
        
        return processed
        
    def run(self):
        """运行视频分割工具"""
        if not self.select_input_files():
            messagebox.showinfo("提示", "未选择任何视频文件，程序退出。")
            return
            
        if not self.get_split_duration():
            messagebox.showinfo("提示", "未设置分割时间，程序退出。")
            return
            
        if not self.select_output_directory():
            messagebox.showinfo("提示", "未选择输出目录，程序退出。")
            return
            
        processed = self.process_videos()
        
        if processed > 0:
            messagebox.showinfo("完成", f"成功处理 {processed}/{len(self.input_files)} 个视频文件。\n输出目录: {self.output_dir}")
        else:
            messagebox.showwarning("警告", "没有成功处理任何视频文件。")

if __name__ == "__main__":
    # 检查FFmpeg是否已安装
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        tk.messagebox.showerror("错误", "未找到FFmpeg。请确保FFmpeg已安装并添加到系统PATH中。")
        exit(1)
        
    splitter = VideoSplitter()
    splitter.run()