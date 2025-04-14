import os
import sys
import time
import uuid
import wave
import queue
import asyncio
import logging
import datetime
import requests
import socketio
from gc import callbacks
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import argparse
from abc import ABC
from playsound import playsound
from bailing import (
    recorder,
    player,
    asr,
    tts
)
from bailing.dialogue import Message, Dialogue
from bailing.utils import is_interrupt, read_config, is_segment, extract_json_from_string
from plugins.registry import Action
from plugins.task_manager import TaskManager

from concurrent.futures import ThreadPoolExecutor, TimeoutError




from bailing.interviewer.interview_logger import InterviewLogger
from bailing.interviewer.question_generator import QuestionGenerator
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict
from bailing.interviewer.agent import InterviewerAgent
import asyncio
import socketio
import requests
load_dotenv()

logger = logging.getLogger(__name__)


app = FastAPI()

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")



def validate_questions(questions_str):
    try:
        # 先找到所有完整的问题对象
        valid_questions = []
        current_depth = 0
        start_pos = questions_str.find('{')
        
        # 跳过第一个大括号（最外层的开始）
        questions_str = questions_str[start_pos + 1:]
        current_question = ""
        
        for char in questions_str:
            if char == '{':
                current_depth += 1
                if current_depth == 1:  # 开始一个新的问题对象
                    current_question = '{'
                else:
                    current_question += char
            elif char == '}':
                current_depth -= 1
                current_question += char
                
                if current_depth == 0:  # 一个问题对象结束
                    try:
                        # 尝试解析这个问题对象
                        question = json.loads(current_question)
                        valid_questions.append(question)
                        current_question = ""
                    except json.JSONDecodeError:
                        # 如果解析失败，说明遇到了不完整的问题
                        break
                
            elif current_depth > 0:  # 在问题对象内部
                current_question += char
        
        return valid_questions
    except Exception as e:
        logger.error(f"处理错误: {str(e)}")
        return []


class InterviewSession:
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config

        # 尝试读取文件内容
        """try:
            with open(self.default_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"文件内容长度: {len(content)}")
        except Exception as e:
            logger.error(f"读取文件时发生错误: {e}")"""
        self.agent = None
        self.question_generator = QuestionGenerator(self.llm_config)
        
        self.logger = InterviewLogger()

    async def initialize_agent(self, patient_info):
        try:
            if patient_info:
                # Generate personalized questions based on patient info
                logging.info("Generating personalized questions...")
                logging.info(f"Patient info: {patient_info}")
                self.default_script_path = os.path.join(os.path.dirname(__file__), "interviewer\\temp_"+patient_info.get('scale')+"_script.json")
        
                logger.info(f"default_script_path: {self.default_script_path}")
                logger.info(f"文件是否存在: {os.path.exists(self.default_script_path)}")
                scale = patient_info.get('scale')
                patient_info_str=json.dumps(patient_info)
                questions = await self.question_generator.generate_questions(patient_info_str,scale)

                # 保存有效的问题到临时文件
                temp_script_path = os.path.join(os.path.dirname(__file__), "interviewer/temp_script.json")

                # Reconstruct the config structure expected by InterviewerAgent
                agent_config = {"llm": self.llm_config}
                self.agent = InterviewerAgent(temp_script_path, agent_config) # Pass reconstructed config
            else:
                logging.info("Using default questions")
                # Assume default script path needs similar reconstruction if used
                self.default_script_path = os.path.join(os.path.dirname(__file__), "interviewer/default_script.json") # Ensure default path exists or is handled
                agent_config = {"llm": self.llm_config}
                self.agent = InterviewerAgent(self.default_script_path, agent_config) # Pass reconstructed config
                
            self.logger.start_new_session()
            return True
            
        except Exception as e:
            logging.error(f"Error initializing agent: {str(e)}")
            return False
"""# Create a dictionary to store active interview sessions
active_sessions = {}
# Get the session
session_id = "default"
session = InterviewSession()
# 初始化agent（可以传入患者信息，这里使用默认脚本）
success = session.initialize_agent()
  """  

class Robot(ABC):
    def __init__(self, config_file):
        # Load the full configuration using the potentially new/unified function
        # Assuming load_config returns the same structure as read_config for now
        config = read_config(config_file)
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            sys.exit(1) # Or raise an exception

        # Keep the original read_config call if load_config is ONLY for LLM
        # config_legacy = read_config(config_file)

        self.audio_queue = queue.Queue()
        self.current_task = None
        self.task_lock = threading.Lock()
        self.async_lock = asyncio.Lock()

        # 尝试读取患者信息
        try:
            patient_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server', 'patient_info.json')
            logger.info(f"尝试读取患者信息文件: {patient_info_path}")
            
            if os.path.exists(patient_info_path):
                with open(patient_info_path, 'r', encoding='utf-8') as f:
                    self.patient_info = json.load(f)
                logger.info(f"成功加载患者信息: {self.patient_info}")
            else:
                self.patient_info = None
                logger.warning("未找到患者信息文件，将使用默认设置")
        except Exception as e:
            self.patient_info = None
            logger.error(f"读取患者信息时发生错误: {e}")

        # 初始化其他组件
        self.asr = asr.create_instance(
            config["selected_module"]["ASR"],
            config["ASR"][config["selected_module"]["ASR"]]
        )

        # Extract the specific LLM config needed by InterviewSession
        # Assuming the structure is config -> LLM -> provider_name -> details
        llm_configuration = config.get("llm") # Get the 'llm' sub-dictionary
        if llm_configuration:
            self.session = InterviewSession(llm_configuration) # Correct: Pass only LLM config
            logger.info("InterviewSession initialized with LLM configuration.")
        else:
            logger.error("LLM configuration ('llm' key) not found in the loaded config. Cannot initialize InterviewSession.")
            # Exit if LLM config is missing, as it's likely essential
            sys.exit(1)

        # Initialize the agent after session is created with correct config
        # (Assuming self.patient_info is defined earlier or passed to __init__)
        if hasattr(self, 'patient_info'): # Ensure patient_info exists before using it
             success = asyncio.run(self.session.initialize_agent(self.patient_info))
             if not success:
                 logger.error("Failed to initialize InterviewSession's agent.")
             else:
                 logger.info("InterviewSession agent initialized successfully.")
        else:
             logger.warning("Patient info not available at agent initialization point.")

        self.tts = tts.create_instance(
            config["selected_module"]["TTS"],
            config["TTS"][config["selected_module"]["TTS"]]
        )


        self.player = player.create_instance(
            config["selected_module"]["Player"],
            config["Player"][config["selected_module"]["Player"]]
        )


        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=10)



        # 打断相关配置
        self.INTERRUPT = config["interrupt"]
        self.silence_time_ms = int((1000 / 1000) * (16000 / 512))  # ms

        # 线程锁
        self.chat_lock = False

        # 事件用于控制程序退出
        self.stop_event = threading.Event()

        self.callback = None

        self.speech = []

        self.task_queue = queue.Queue()
        self.task_manager = TaskManager(config.get("TaskManager"), self.task_queue)
        self.start_task_mode = config.get("StartTaskMode")
        
        # 专门用于 TTS 的队列
        self.tts_queue = queue.Queue(maxsize=1)  # 限制队列大小为1
        
    def listen_dialogue(self, callback):
        self.callback = callback
        


    def _tts_priority(self):
        logger.info("TTS 线程启动")
        def priority_thread():
            while not self.stop_event.is_set():
                try:
                    # 尝试获取最新的 TTS 任务
                    future = self.tts_queue.get(timeout=5)
                    
                    try:
                        # 获取 TTS 文件
                        logger.debug("准备获取 TTS 文件")
                        tts_file = future.result()
                        logger.debug(f"TTS 文件生成结果: {tts_file}")
                        
                        if tts_file and os.path.exists(tts_file):
                            logger.debug(f"TTS 文件存在: {tts_file}")
                            
                            # 通过回调通知
                            if self.callback:
                                logger.debug("准备调用回调函数")
                                self.callback({
                                    "role": "assistant",
                                    "type": "audio",
                                    "content": tts_file
                                })
                                logger.debug("回调函数调用完成")
                        else:
                            logger.error(f"TTS 文件无效: {tts_file}")
                    
                    except concurrent.futures.CancelledError:
                        logger.info("TTS 任务已被取消")
                    except TimeoutError:
                        logger.error("TTS 任务超时")
                    except Exception as e:
                        logger.error(f"TTS 处理错误: {e}")
                
                except queue.Empty:
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"TTS 线程异常: {e}")
                    time.sleep(0.1)
        
        # 启动线程
        tts_priority = threading.Thread(target=priority_thread, daemon=True)
        tts_priority.start()

    def shutdown(self):
        """关闭所有资源，确保程序安全退出"""
        logger.info("Shutting down Robot...")
        self.stop_event.set()
        
        # Close InterviewAgent clients if agent exists
        logger.info("Attempting to close agent clients...")
        if hasattr(self.session, 'agent') and self.session.agent and hasattr(self.session.agent, 'close_clients'):
            try:
                # Running async function from sync context
                try:
                    loop = asyncio.get_running_loop()
                    loop.run_until_complete(self.session.agent.close_clients())
                except RuntimeError: # No running event loop
                    asyncio.run(self.session.agent.close_clients())
                logger.info("Agent clients closed successfully.")
            except Exception as e:
                logger.error(f"Error closing agent clients: {e}")
        else:
            logger.info("No agent or close_clients method found to close.")

        # Close QuestionGenerator client if it exists and has the method
        logger.info("Attempting to close question generator client...")
        if hasattr(self.session, 'question_generator') and self.session.question_generator and hasattr(self.session.question_generator, 'close_client'):
            try:
                # Running async function from sync context
                try:
                    loop = asyncio.get_running_loop()
                    loop.run_until_complete(self.session.question_generator.close_client())
                except RuntimeError: # No running event loop
                    asyncio.run(self.session.question_generator.close_client())
                logger.info("Question generator client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing question generator client: {e}")
        else:
            logger.info("No question generator or close_client method found to close.")

        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Shutdown complete.")

    def start_recording_and_vad(self):
        """启动后台音频监听和处理"""
        # 启动音频监听线程
        audio_thread = threading.Thread(target=self._listen_audio_stream, daemon=True)
        audio_thread.start()
        
 
        # tts优先级队列
        self._tts_priority()

    def cancel_current_task(self):
        """取消当前正在执行的任务"""
        with self.task_lock:
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                logger.info("已取消当前任务")

    def _listen_audio_stream(self):
        """后台监听音频流的线程函数"""
        sio = socketio.Client(ssl_verify=False)
        
        @sio.on('robot_audio_data')
        
        def on_audio_data(audio_data):
            logger.info("接收到音频数据:")
            logger.info(f"格式: {audio_data.get('format')}")
            logger.info(f"采样率: {audio_data.get('sampleRate')}")
            logger.info(f"时间戳: {audio_data.get('timestamp')}")
            logger.info(f"数据大小: {len(audio_data.get('audio', b''))} 字节")
            

            
            
            # 保存音频文件
            audio_dir = os.path.join(os.path.dirname(__file__), 'audio_files')
            os.makedirs(audio_dir, exist_ok=True)
            
            # 使用时间戳作为文件名
            timestamp = audio_data.get('timestamp', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            audio_path = os.path.join(audio_dir, f'robot_audio_{timestamp}.wav')
            
            # 保存音频数据
            with open(audio_path, 'wb') as f:
                f.write(audio_data.get('audio', b''))
            logger.info(f"robot_audio_data_音频文件已保存: {audio_path}")
            # 使用保存的WAV文件进行识别
            text, _ = self.asr.recognizer(audio_path)
            
            if text:
                logger.info(f"_listen_audio_streamASR识别结果: {text}")
                # 将完整的音频信息放入队列
                self.audio_queue.put(audio_data)
                if self.current_task:
                        logger.info("取消当前任务")
                        self.cancel_current_task()  # 直接调用取消方法

                
        sio.connect('https://localhost:5000')
        sio.wait()

    async def process_text(self, text):
        """处理文本的异步方法"""
        try:
            response_message = await self.chat_tool(text)
            if not self.current_task or not self.current_task.cancelled():
                await self.handle_response(response_message)
        except asyncio.CancelledError:
            logger.info("当前任务被取消")
        except Exception as e:
            logger.error(f"处理文本时出错: {e}")
        finally:
            self.chat_lock = False

    async def handle_response(self, response_message):
        """处理响应消息的异步方法"""
        try:
            # --- MODIFIED Error Handling ---
            if not response_message or not isinstance(response_message, dict):
                logger.error(f"Invalid response_message received by handle_response: {response_message}")
                return # Do nothing if the message structure is wrong

            # Check if chat_tool indicated an internal error
            if response_message.get("internal_error"):
                error_msg = response_message.get("error_message", "Unknown internal error")
                logger.error(f"Internal error detected in handle_response: {error_msg}")
                # DO NOT send this internal error to the user or TTS. Just log it.
                # Optionally, you could send a *very generic*, non-specific message like:
                # if self.callback:
                #     self.callback({"role": "assistant", "content": "抱歉，我暂时无法回应。", "need_confirm": False})
                return # Stop processing this message further
            # --- END MODIFIED Error Handling ---

            # --- Original logic for VALID responses ---
            if self.callback:
                # Send text content via callback
                text_content = response_message.get('response', '')
                self.callback({
                    "role": "assistant",
                    "content": text_content,
                    "need_confirm": response_message.get('need_confirm', False) # Pass confirm flag
                })

                # Submit for TTS only if there is text content
                if text_content:
                    future = self.executor.submit(self.speak_and_play, text_content)
                    # TTS Queue logic (same as before)
                    try:
                        old_future = self.tts_queue.get_nowait()
                        old_future.cancel()
                    except queue.Empty:
                        pass
                    try:
                        self.tts_queue.put_nowait(future)
                    except queue.Full:
                        logger.error("TTS队列已满")

            # Handle image types (same as before)
            response_type = response_message.get('type', 'text') # Default to text
            if response_type in ['image', 'draw']:
                image_path = response_message.get('content', response_message.get('image_path', '')) # Check both keys
                if image_path and self.callback:
                    self.callback({
                        "role": "assistant",
                        "type": response_type,
                        "content": image_path
                    })
        except Exception as e:
            logger.exception(f"处理响应时出错: {e}") # Use exception for traceback

    def _duplex(self):
        """后台处理语音流的线程函数"""
        logger.debug("_duplex")
        
        # 识别到vad开始
        # 处理识别结果
        audio_data = self.audio_queue.get()
        if audio_data is None:
            return
        
        
        # 从字典中获取音频数据和相关信息
        audio_bytes = audio_data.get('audio', b'')
        if isinstance(audio_bytes, list):
            # 如果是整数列表，转换为字节列表
            audio_bytes = [bytes([b]) for b in audio_bytes]
        
        timestamp = audio_data.get('timestamp', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # 保存音频文件
        audio_dir = os.path.join(os.path.dirname(__file__), 'duplex_audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 使用时间戳作为文件名
        audio_path = os.path.join(audio_dir, f'duplex_{timestamp}.wav')
        
        # 保存音频数据到WAV文件
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位采样
            wf.setframerate(16000)  # 16kHz采样率
            if isinstance(audio_bytes, list):
                wf.writeframes(b''.join(audio_bytes))
            else:
                wf.writeframes(audio_bytes)
        logger.info(f"_duplex方法中保存音频文件: {audio_path}")

        # 使用保存的WAV文件进行识别
        text, tmpfile = self.asr.recognizer(audio_path)

        # 检查识别结果
        if text is None:
            logger.debug("ASR识别结果为None，跳过处理")
            return
                
        if not text.strip():
            logger.debug("识别结果为空，跳过处理。")
            return

        logger.info(f"ASR识别结果: {text}")


        if self.callback:
            self.callback({"role": "user", "content": str(text)})

        # 创建新的异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self.task_lock:
            self.current_task = loop.create_task(self.process_text(text))
        loop.run_until_complete(self.current_task)
        loop.close()
        return True
            

    def run(self):
        
        
        script_path = os.path.join("bailing", "interviewer", "temp_script.json")
        with open(script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            initial_question=data["questions"][0]["question"]
        
        logger.info(f"初始问话：{initial_question}")
        # 发送开始录音信号
        try:
                response = requests.post('https://localhost:5000/start_main_recording', verify=False)
                if response.status_code == 200:
                    logger.info("已发送开始录音信号")
                else:
                    logger.error(f"发送开始录音信号失败: {response.text}")
        except Exception as e:
                logger.error(f"发送开始录音信号时发生错误: {str(e)}")

        # 更新对话
        tts_file = self.tts.to_tts(initial_question)
        #tts_file = None
        logger.debug(f"TTS 文件生成完毕{tts_file}")
        
        if self.callback:
            
            self.callback({
                "role": "assistant", 
                "content": initial_question,
                "need_confirm":False
            })
            self.callback({
                "role": "assistant",
                "type": "audio",
                "content": tts_file
            })
            
        
        try:
            self.start_recording_and_vad()  # 监听语音流
            logger.info("开始音频监听和处理...")
            while not self.stop_event.is_set():
                try:
                    self._duplex()  # 双工处理
                except queue.Empty:
                    # 队列为空是正常的，继续等待
                    continue
                except Exception as e:
                    logger.error(f"双工处理发生错误: {str(e)}", exc_info=True)
                    # 不要因为处理错误就退出循环
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Exiting...")
        except Exception as e:
            logger.error(f"音频处理主循环发生严重错误: {str(e)}", exc_info=True)
        finally:
            error_type = sys.exc_info()[0]
            if error_type is not None:
                logger.error(f"退出原因: {error_type.__name__}")
            logger.info("Shutting down Robot...")
            #self.shutdown()

    def speak_and_play(self, text):
        
        if text is None or len(text)<=0:
            logger.info(f"无需tts转换，query为空，{text}")
            return None
        tts_file = self.tts.to_tts(text)
        #tts_file = None
        if tts_file is None:
            logger.error(f"tts转换失败，{text}")
            return None
        logger.info(f"TTS 文件生成完毕{tts_file}")
        
        
        #if self.chat_lock is False:
        #    return None
        # 开始播放
        # self.player.play(tts_file)
        #return True
        # 播放语音
        #self.player.play(tts_file)
        return tts_file

    async def chat_tool(self, query):
        try:
            # ... (check agent) ...
            logger.info(f"开始处理查询: {query}")
            async with self.async_lock:
                self.current_task = asyncio.create_task(self.session.agent.generate_next_action(query))
                try:
                    llm_responses = await asyncio.wait_for(self.current_task, timeout=60.0) # Increased timeout slightly
                    logger.info(f"LLM 原始响应: {llm_responses}")

                    # --- Important: Check agent response format EARLY ---
                    if not isinstance(llm_responses, dict):
                         logger.error(f"Agent response format error, expected dict, got {type(llm_responses)}")
                         # Return internal error structure
                         return {"internal_error": True, "error_message": "Agent response format error"}

                    # Check if the agent itself reported an error
                    if llm_responses.get('action') == 'error' or 'error' in llm_responses:
                         error_msg = llm_responses.get('error', 'Agent reported an unknown error')
                         logger.error(f"Agent returned an error: {error_msg}")
                         # Return internal error structure
                         return {"internal_error": True, "error_message": f"Agent error: {error_msg}"}

                    # --- Process valid agent response ---
                    # (Original processing logic for normal/image responses)
                    content = llm_responses.get('response')
                    if not content and llm_responses.get('type') not in ['image', 'draw']: # Check content unless it's an image
                         logger.error(f"Agent response content is empty: {llm_responses}")
                         return {"internal_error": True, "error_message": "Agent response content empty"}

                    logger.info(f"最终响应内容: {content}")
                    # Return the valid agent response directly
                    return llm_responses # Pass the original structure through

                except asyncio.TimeoutError:
                    logger.error("LLM 响应超时 (wait_for)")
                    if self.current_task and not self.current_task.done(): self.current_task.cancel()
                    # Return internal error structure, NO user-facing "response"
                    return {"internal_error": True, "error_message": "LLM response timeout"}
                except asyncio.CancelledError:
                    logger.info("Agent task被取消")
                    # Return internal error structure
                    return {"internal_error": True, "error_message": "Task cancelled"}
                except Exception as e:
                    logger.error(f"LLM 调用或Agent处理失败: {str(e)}", exc_info=True)
                    if self.current_task and not self.current_task.done(): self.current_task.cancel()
                    # Return internal error structure
                    return {"internal_error": True, "error_message": f"LLM/Agent call failed: {str(e)}"}
                finally:
                    self.current_task = None

        except Exception as e:
            # Catch errors in chat_tool setup itself
            logger.exception(f"处理查询时发生意外错误: {query}")
            return {"internal_error": True, "error_message": f"Error in chat_tool: {str(e)}"}


    async def chat(self, query):
        logger.debug("开始处理聊天消息")
        
        # 直接使用 await 调用异步方法
        response_message = await self.chat_tool(query)
        
        logger.info(f"生成响应消息: {response_message}")

        self.chat_lock = True
        
        try:
            logger.debug("开始处理聊天消息")
            
            # 更新对话
            if self.callback:
                
                logger.info("调用回调函数发送文本消息")
                self.callback({
                    "role": "assistant", 
                    "content": response_message['response'],
                    "need_confirm": response_message['need_confirm']
                })
                # 提交新的语音合成任务
                logger.debug("提交 TTS 任务")
                future = self.executor.submit(self.speak_and_play, response_message['response'])
                
                # 清空并放入最新任务
                try:
                    # 尝试取出并取消之前的任务
                    old_future = self.tts_queue.get_nowait()
                    old_future.cancel()
                    logger.debug("取消了之前的 TTS 任务")
                except queue.Empty:
                    logger.debug("队列为空，无需取消")
                
                # 放入新任务
                try:
                    self.tts_queue.put_nowait(future)
                    logger.debug("TTS 任务入队完成")
                except queue.Full:
                    logger.error("TTS 队列已满，无法放入新任务")
            if response_message.get('type') in ['image', 'draw']:
                logger.debug(response_message)
                logger.debug("回复为图片")
                if self.callback:
                    self.callback({
                        "role": "assistant",
                        "type": response_message.get('type'),
                        "content": response_message['image_path']
                    })
        except Exception as e:
            logger.error(f"语音处理错误: {e}")
        
        finally:
            self.chat_lock = False
        
        return True
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="百聆机器人")

    # Add arguments
    parser.add_argument('config_path', type=str, help="配置文件", default=None)

    # Parse arguments
    args = parser.parse_args()
    config_path = args.config_path

    # 创建 Robot 实例并运行
    robot = Robot(config_path)
    logger.info("Robot started")
    asyncio.run(robot.run())
