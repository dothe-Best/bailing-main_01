# 导入必要的标准库和第三方库
import argparse   # 用于解析命令行参数
import json       # JSON数据处理
import logging    # 日志记录
import os
import yaml
import socket
import time
from bailing.utils import  read_config
import requests  # 添加 requests 库导入
from bailing import tts

# 配置日志记录系统
# 日志将同时输出到控制台和文件，便于调试和追踪
logging.basicConfig(
    level=logging.INFO,  # 改为 INFO 级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式：时间 - 模块名 - 日志级别 - 消息
    handlers=[
        logging.StreamHandler(),    # 控制台输出处理器
        logging.FileHandler('tmp/bailing.log')  # 文件输出处理器，日志将写入指定文件
    ]
)


# 导入项目自定义模块
from bailing import robot

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

open_text="您好！欢迎使用“百聆”智能访谈助手。我是一个基于人工智能技术的对话系统，旨在通过一次轻松、自然的交流，协助您完成心理健康方面的评估访谈。在接下来的时间里，我会像一位专业的访谈员一样，向您询问一些关于您近期感受、想法和经历的问题。请放心，您可以坦诚地分享，没有“对”或“错”的答案，我会耐心倾听。准备好后，我们就可以开始了。"

def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def upload_video_to_web(config, web_url="https://127.0.0.1:5000", max_retries=3):
    """
    将视频文件上传到Web服务器，带有重试机制和详细的错误处理

    Args:
        config (dict): 配置信息
        web_url (str, optional): Web服务器的基本URL
        max_retries (int, optional): 最大重试次数，默认3次

    Returns:
        str or None: 上传成功返回视频URL，失败返回None
    """
    logger.info("TTS 创建成功")
    tts_tts = tts.create_instance(
            config["selected_module"]["TTS"],
            config["TTS"][config["selected_module"]["TTS"]]
    )
    video_path = config['system']['initialization_video']['path']
    retry_count = 0
    retry_delay = 2  # 初始重试延迟（秒）

    # 检查视频文件
    try:
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return None
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.error(f"视频文件为空: {video_path}")
            return None
        
        logger.info(f"准备上传视频，文件大小: {file_size / (1024*1024):.2f}MB")
    except Exception as e:
        logger.error(f"检查视频文件时出错: {e}")
        return None

    # 确保使用HTTPS
    if web_url.startswith('http://'):
        web_url = web_url.replace('http://', 'https://')
        logger.info(f"已将URL更新为HTTPS: {web_url}")

    url = f"{web_url}/upload_video"
    cert_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server', 'certs', 'cert.pem')
    
    while retry_count < max_retries:
        try:
            logger.info(f"尝试上传视频 (第{retry_count + 1}次)")
            
            #TTS
            open_audio = tts_tts.to_tts(open_text)
            logger.info(open_audio)
            logger.info(f"开场白TTS完成")
            logger.info(f"开场白TTS文件:{open_audio}")
            
            # 设置超时和重试参数
            with open(video_path, "rb") as video_file:
                files = {"video": video_file}
                verify = cert_path if os.path.exists(cert_path) else False
                response = requests.post(
                    url,
                    files=files,
                    timeout=(30, 300),  # 连接超时30秒，读取超时300秒
                    verify=verify  # 使用自签名证书或在开发环境中禁用验证
                )

            if response.status_code == 200:
                video_url = response.json().get("url")
                logger.info(f"视频文件上传成功，URL: {video_url}")
                
                push2web({
                "role": "assistant", 
                "type": "text",
                "content": open_text,
                "need_confirm":False
                })
                logger.info(f"开场白推送完成")
                push2web({
                    "role": "assistant",
                    "type": "audio",
                    "content": open_audio,
                    "need_confirm":False
                })

                return video_url
            else:
                logger.warning(f"上传失败 (尝试 {retry_count + 1}/{max_retries}): HTTP {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"上传超时 (尝试 {retry_count + 1}/{max_retries})")
        except requests.exceptions.SSLError:
            logger.warning("SSL证书验证失败，已设置忽略证书验证")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"连接错误 (尝试 {retry_count + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"未预期的错误 (尝试 {retry_count + 1}/{max_retries}): {e}")
        
        retry_count += 1
        if retry_count < max_retries:
            wait_time = retry_delay * (2 ** (retry_count - 1))  # 指数退避
            logger.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

    logger.error(f"视频上传失败，已达到最大重试次数 ({max_retries})")
    return None


def push2web(payload, web_url="https://127.0.0.1:5000"):
    """
    将对话消息推送到本地Web服务器，支持文本和语音消息
    
    Args:
        payload (dict): 要推送的消息负载，包含消息类型和内容
        web_url (str): Web 服务器的基本 URL
    """
    
    try:
        message_type = payload.get("type", "text")  # 默认为文本消息
        message_content = payload.get("content", "")
        logger.info(f"推送消息：类型 {message_type}, 内容 {message_content}")

        # 设置证书验证
        cert_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server', 'certs', 'cert.pem')
        verify = cert_path if os.path.exists(cert_path) else False
        
        if message_type == "text":
            # 文本消息：直接发送到 Web 服务器
            url = f"{web_url}/add_message"
            headers = {'Content-Type': 'application/json; charset=utf-8'}
            payload = {
                "type": "text",
                "role": payload.get("role", "assistant"),
                "content": message_content,
                "need_confirm": payload.get("need_confirm")
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload).encode('utf-8'), verify=verify)
            logger.info(f"文本消息推送成功: {response.text}")
            
        elif message_type == "audio":
            # 语音消息：上传语音文件到Web服务器
            if os.path.exists(message_content):  # 检查语音文件是否存在
                url = f"{web_url}/audio2web"
                with open(message_content, "rb") as audio_file:
                    files = {"audio": audio_file}
                    response = requests.post(url, files=files, verify=verify)
                if response.status_code == 200:
                    audio_url = response.json().get("url")
                    logger.info(f"语音文件上传成功，URL: {audio_url}")
    
                    # 直接发送 URL，不使用递归
                    url = f"{web_url}/add_message"
                    headers = {'Content-Type': 'application/json; charset=utf-8'}
                    payload = {
                        "type": "audio",
                        "content": audio_url
                    }
                    requests.post(url, headers=headers, data=json.dumps(payload).encode('utf-8'), verify=verify)
                else:
                    logger.error(f"语音文件上传失败: {response.text}")
            else:
                logger.error(f"语音文件不存在: {message_content}")

        elif message_type in ['image', 'draw']:
            # 图片消息：上传图片文件到Web服务器
            if os.path.exists(message_content):  # 检查图片文件是否存在
                url = f"{web_url}/upload_image"
                with open(message_content, "rb") as image_file:
                    files = {"file": image_file}
                    response = requests.post(url, files=files, verify=verify)
                if response.status_code == 200:
                    image_url = response.json().get("url")
                    logger.info(f"图片文件上传成功，URL: {image_url}")
    
                    # 直接发送 URL，不使用递归
                    url = f"{web_url}/add_message"
                    headers = {'Content-Type': 'application/json; charset=utf-8'}
                    payload = {
                        "type": message_type,
                        "content": image_url
                    }
                    requests.post(url, headers=headers, data=json.dumps(payload).encode('utf-8'), verify=verify)
                else:
                    logger.error(f"图片文件上传失败: {response.text}")
            else:
                logger.error(f"图片文件不存在: {message_content}")

        elif message_type == "video":
            # 视频消息：上传视频文件到Web服务器
            if os.path.exists(message_content):  # 检查视频文件是否存在
                video_url = upload_video_to_web(message_content, web_url)
                if video_url:
                    logger.info(f"视频文件上传成功，URL: {video_url}")
    
                    # 直接发送 URL，不使用递归
                    url = f"{web_url}/add_message"
                    headers = {'Content-Type': 'application/json; charset=utf-8'}
                    payload = {
                        "type": "video",
                        "content": video_url
                    }
                    requests.post(url, headers=headers, data=json.dumps(payload).encode('utf-8'), verify=verify)
                else:
                    logger.error(f"视频文件上传失败")
            else:
                logger.error(f"视频文件不存在: {message_content}")

        else:
            logger.error(f"未知的消息类型: {message_type}")

    except Exception as e:
        logger.error(f"消息处理出错: {e}")


def main():
    """
    主程序入口函数
    1. 解析命令行配置文件路径
    2. 创建机器人实例
    3. 设置对话监听回调
    4. 运行机器人
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="百聆机器人启动脚本")

    # 添加配置文件路径参数，提供默认值
    parser.add_argument('--config_path', type=str, help="配置文件路径", default="config/config.yaml")

    # 解析命令行参数
    args = parser.parse_args()
    config_path = args.config_path

    # 加载配置
    config = read_config(config_path)

    # 返回视频长度视频
    video_time=upload_video_to_web(config)

    
        
    
    # 使用指定配置文件创建机器人实例
    bailing_robot = robot.Robot(config_path)
    
    # 设置对话监听回调，将对话消息推送到Web服务器
    bailing_robot.listen_dialogue(push2web)
    logger.info("Robot started")
    # 启动机器人
    bailing_robot.run()

# 脚本直接运行时的入口
if __name__ == "__main__":
    main()