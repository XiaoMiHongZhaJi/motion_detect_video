import glob
import logging
import os
import sys
import cv2
import numpy as np
import time
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 运动检测灵敏度，值越大灵敏度越低
motion_sensitivity = 1000

# 运动持续时间超过这个时间才会被截取（秒）
motion_duration_threshold = 2

# 运动停止后延迟截取的时间（秒）
motion_post_delay = 1


def detect_motion(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {video_file}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_frame = None
    motion_start_time = 0
    motion_clips = []
    is_motion_detected = False

    frame_count = 0
    skipping_frames = False
    frames_to_skip = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if skipping_frames:
            frames_to_skip -= 1
            if frames_to_skip > 0:
                continue
            else:
                skipping_frames = False

        # 每隔5帧检测一次
        if frame_count % 5 != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_pixels = np.sum(thresh) // 255  # 计算运动像素的数量
        prev_frame = gray_frame

        if motion_pixels > motion_sensitivity:
            logging.debug(f"frame {frame_count} 触发运动 {motion_pixels} > {motion_sensitivity}")
            if not is_motion_detected:
                is_motion_detected = True
                motion_start_time = frame_count / fps
                logging.info(f"开始记录 motion_start_time {motion_start_time} {motion_pixels} > {motion_sensitivity}")
            # 只要记录到运动，就跳帧
            skipping_frames = True
            # 跳过 post_delay 时间长度的帧
            frames_to_skip = int(motion_post_delay * fps) - 1
            continue  # 跳过当前帧，等待跳帧结束后再比较
        else:
            if motion_pixels > 0:
                logging.debug(f"frame {frame_count} 未触发运动 {motion_pixels} < {motion_sensitivity}")
                logging.info(f"当前进度：{int(frame_count * 100 / frame_count_all)}%  {frame_count} / {frame_count_all}")
            if is_motion_detected:
                is_motion_detected = False
                motion_end_time = frame_count / fps
                during_time = motion_end_time - motion_start_time
                if during_time >= motion_duration_threshold:
                    logging.info(f"frame {frame_count} {motion_start_time} - {motion_end_time} 运动时间 {during_time} > {motion_duration_threshold} 保留")
                    logging.info(f"当前进度：{int(frame_count * 100 / frame_count_all)}%  {frame_count} / {frame_count_all}")
                    start_time_clip = max(0, motion_start_time)
                    end_time_clip = min(frame_count_all / fps, motion_end_time)
                    motion_clips.append((start_time_clip, end_time_clip))
                else:
                    logging.info(f"frame {frame_count} {motion_start_time} - {motion_end_time} 运动时间 {during_time} < {motion_duration_threshold} 丢弃")

                motion_start_time = None

    cap.release()
    return motion_clips


def clip_video(video_file, motion_clips):
    if not motion_clips:
        logging.warning("未检测到运动片段。")
        return None
    logging.info(f"motion_clips[{len(motion_clips)}]: {motion_clips}")
    final_clips = []
    for i, (start, end) in enumerate(motion_clips):
        clip = VideoFileClip(video_file).subclipped(start, end)
        final_clips.append(clip)
        # 除了最后一个片段，都在后面添加黑帧
        if i < len(motion_clips) - 1:
            # 获取当前片段的尺寸，用于创建黑帧
            width, height, fps = clip.w, clip.h, clip.fps
            black_frame = ColorClip(size=(width, height), color=(0, 0, 0), duration=1 / fps)
            final_clips.append(black_frame)

    if not final_clips:
        logging.warning("运动片段合成失败。")
        return None

    base_name = os.path.basename(video_file)
    name, ext = os.path.splitext(base_name)
    output_file = f"motion_detected_{name}.mp4"
    final_clip = concatenate_videoclips(final_clips)
    final_clip.write_videofile(output_file)
    return output_file


def process_video_file(video_files):
    for video_file in video_files:
        base_name = os.path.basename(video_file)
        if os.path.isfile(f"motion_detected_{base_name}"):
            logging.info(f"{video_file} 跳过处理，motion_detected_{base_name} 已存在")
            continue
        logging.info(f"{video_file} 开始处理")
        start_time = int(time.time())
        # 检测
        motion_clips = detect_motion(video_file)
        logging.info(f"{video_file} 检测完成，用时：{int(time.time()) - start_time}")
        # 拼接
        output_video = clip_video(video_file, motion_clips)
        logging.info(f"{video_file} 处理完成，总用时：{int(time.time()) - start_time}")
        if output_video:
            logging.info(f"{video_file} 运动检测完成，已保存到: {output_video}")
        else:
            logging.warning(f"{video_file} 未检测到运动")


if __name__ == "__main__":

    video_files_to_process = []

    if len(sys.argv) > 1:
        # 从命令行参数获取视频文件
        video_files_to_process = sys.argv[1:]
    else:
        logging.warning("未传参，将处理所有未处理过的文件")
        # 检测当前目录下所有非 motion_detected_ 开头的视频文件
        for file_path in glob.glob("*.mp4"):
            if not file_path.startswith("motion_detected_") and os.path.isfile(file_path):
                video_files_to_process.append(file_path)

    if not video_files_to_process:
        logging.info("没有指定要处理的视频文件，也没有在当前目录下找到合适的视频文件")
    else:
        process_video_file(video_files_to_process)
