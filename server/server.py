from flask import Flask, request, render_template, jsonify, send_file
from flask_socketio import SocketIO, emit, disconnect
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import uuid
import logging
import subprocess
import sys
import threading
import os
from OpenSSL import crypto
from datetime import datetime, timedelta
import asyncio
import wave
import io
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置 engineio 和 socketio 的日志级别为 WARNING，这样就不会显示 PING/PONG 消息
logging.getLogger('engineio.server').setLevel(logging.WARNING)
logging.getLogger('socketio.server').setLevel(logging.WARNING)

# 配置
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['STATIC_FOLDER'] = "static"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大上传大小为100MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 音频文件配置
UPLOAD_FOLDER_AUDIO = os.path.join(os.path.dirname(__file__), 'audio_files')
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}

# 确保音频上传目录存在
os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

# 初始化 Flask-SocketIO，允许所有源，增加重连和错误处理
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=100 * 1024 * 1024,  # 设置Socket.IO的最大传输大小为100MB
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1000,
    reconnection_delay_max=5000,
    async_mode='threading'  # 在这里设置异步模式
)

# 存储活动连接
active_connections = set()

# 对话记录
dialogue = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    active_connections.add(client_id)
    #logger.info(f'Client connected. ID: {client_id}')
    #logger.info(f'Active connections: {len(active_connections)}')
    emit('update_dialogue', dialogue, broadcast=False)

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    if client_id in active_connections:
        active_connections.remove(client_id)
    #logger.info(f'Client disconnected. ID: {client_id}')
    #logger.info(f'Active connections: {len(active_connections)}')

@socketio.on_error()
def error_handler(e):
    logger.error(f'SocketIO error: {str(e)}')
    emit('error', {'message': 'An error occurred'}, broadcast=False)

def start_main_script():
    """
    启动 main.py 脚本并实时捕获输出
    """
    try:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_script_path = os.path.join(project_root, 'main.py')
        
        # 使用当前 Python 解释器运行 main.py
        process = subprocess.Popen(
            [sys.executable, main_script_path], 
            cwd=project_root, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # 创建线程来读取输出
        def log_output(pipe, log_func):
            try:
                for line in iter(pipe.readline, ''):
                    log_func(f"main.py: {line.strip()}")
            except Exception as e:
                logger.error(f"读取输出时发生错误: {e}")
            finally:
                pipe.close()
        
        # 创建并启动线程来读取标准输出和标准错误
        stdout_thread = threading.Thread(
            target=log_output, 
            args=(process.stdout, logger.info),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=log_output, 
            args=(process.stderr, logger.error),
            daemon=True
        )
        
        # 启动线程
        stdout_thread.start()
        stderr_thread.start()
        
        logger.info(f"成功启动 main.py，进程ID: {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"启动 main.py 失败: {e}")
        return None

@socketio.on('patient_info')
def handle_patient_info(data):
    """
    处理患者信息并启动 main.py
    """
    try:
        logger.info(f"收到患者信息: {data}")
        
        # 将患者信息保存到临时文件
        patient_info_path = os.path.join(os.path.dirname(__file__), 'patient_info.json')
        with open(patient_info_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # 启动 main.py
        main_process = start_main_script()
        
        if main_process is not None:
            logger.info("main.py 启动成功")
            # 广播患者信息
            socketio.emit('patient_registered', {
                'patient_data': data,
                'main_process_started': True,
                'status': 'success'
            })
            logger.info("患者信息上传成功")
        else:
            logger.error("main.py 启动失败")
            socketio.emit('patient_registered', {
                'status': 'error',
                'message': 'Failed to start main process'
            }, broadcast=True)
            
    except Exception as e:
        logger.error(f"处理患者信息时发生错误: {e}")
        emit('patient_registered', {
            'status': 'error',
            'message': str(e)
        }, broadcast=True)

@app.route('/add_message', methods=['POST'])
def add_message():
    """
    添加消息到对话记录，支持文本和音频消息
    """
    data = request.json
    
    # 生成唯一消息 ID
    message_id = str(uuid.uuid4())
    
    message = {
        'id': message_id,  # 唯一标识符
        'role': data.get('role', 'assistant'),
        'content': data.get('content', ''),
        'type': data.get('type', 'text'),  # 默认为文本消息
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'audio_file': '',
        'audio_url': '',
        'need_confirm': data.get('need_confirm')
    }

    # 处理音频消息
    if message['type'] == 'audio':
        audio_url = data.get('content')
        if audio_url:
            message['audio_url'] = audio_url
            message['audio_file'] = os.path.basename(audio_url)

    # 添加到对话历史
    dialogue.append(message)
    
    # 限制对话历史长度（可选）
    if len(dialogue) > 100:
        dialogue.pop(0)

    # 广播消息
    socketio.emit('update_dialogue', dialogue)

    return jsonify({
        "status": "success", 
        "message_id": message_id
    }), 200

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """处理音频文件并直接转发给robot"""
    try:
        # 检查是否有文件
        if 'audio' not in request.files:
            logger.error('没有音频文件在请求中')
            return jsonify({'error': '没有音频文件'}), 400
        
        audio_file = request.files['audio']
        logger.info(f'接收到音频文件: {audio_file.filename}')
        
        # 检查文件名
        if audio_file.filename == '':
            logger.error('文件名为空')
            return jsonify({'error': '没有选择文件'}), 400
            
        # 检查文件类型
        if audio_file and allowed_audio_file(audio_file.filename):
            # 读取音频数据
            audio_data = audio_file.read()
            #logger.info(f"原始数据大小: {len(audio_data)} 字节")
            
            # 保存音频文件
            audio_dir = os.path.join(os.path.dirname(__file__), 'audio_files')
            os.makedirs(audio_dir, exist_ok=True)
            
            # 使用时间戳作为文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_path = os.path.join(audio_dir, f'audio_{timestamp}.wav')
            
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            #logger.info(f"音频文件已保存: {audio_path}")
            
            try:
                # 解析WAV文件头
                if len(audio_data) > 44:  # WAV头部至少44字节
                                       
                    header = audio_data[:44]
                    riff = header[:4].decode('ascii')
                    file_size = int.from_bytes(header[4:8], 'little')
                    wave = header[8:12].decode('ascii')
                    fmt_mark = header[12:16].decode('ascii')
                    fmt_size = int.from_bytes(header[16:20], 'little')
                    audio_format = int.from_bytes(header[20:22], 'little')
                    num_channels = int.from_bytes(header[22:24], 'little')
                    sample_rate = int.from_bytes(header[24:28], 'little')
                    byte_rate = int.from_bytes(header[28:32], 'little')
                    block_align = int.from_bytes(header[32:34], 'little')
                    bits_per_sample = int.from_bytes(header[34:36], 'little')
                    """ 
                    logger.info("WAV文件头解析结果：")
                    logger.info(f"标识符: {riff}")
                    logger.info(f"文件大小: {file_size + 8} 字节")
                    logger.info(f"格式: {wave}")
                    logger.info(f"格式块标识: {fmt_mark}")
                    logger.info(f"格式块大小: {fmt_size}")
                    logger.info(f"音频格式: {audio_format} (1表示PCM)")
                    logger.info(f"声道数: {num_channels}")
                    logger.info(f"采样率: {sample_rate} Hz")
                    logger.info(f"字节率: {byte_rate} bytes/sec")
                    logger.info(f"数据块对齐: {block_align} bytes")
                    logger.info(f"采样位数: {bits_per_sample} bits")
                    """
                    # 构建音频数据字典
                    audio_dict = {
                        'audio': audio_data,
                        'format': 'wav',
                        'sampleRate': sample_rate,
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                    }
                else:
                    raise ValueError("WAV文件头不完整")
                    
            except Exception as e:
                logger.error(f"WAV文件解析错误: {str(e)}")
                # 如果解析失败，使用默认值
                audio_dict = {
                    'audio': audio_data,
                    'format': 'raw',
                    'sampleRate': 16000,
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                }
                logger.warning("使用默认音频参数继续处理")
            
            # 构建音频数据字典
            audio_dict = {
                'audio': audio_data,
                'format': 'wav',  # 从前端接收的是wav格式
                'sampleRate': sample_rate,  # 采样率
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            # 通过Socket.IO发送给robot
            socketio.emit('robot_audio_data', audio_dict)
            #logger.info('已通过Socket.IO发送音频数据给robot')
            #logger.info('音频数据已转发给robot')
            
            return jsonify({
                'message': '音频数据处理成功',
                'status': 'forwarded',
                'dataSize': len(audio_data)
            })
        else:
            logger.error(f'不支持的文件类型: {audio_file.filename}')
            logger.error(f'允许的文件类型: {ALLOWED_AUDIO_EXTENSIONS}')
            return jsonify({'error': '不支持的文件类型'}), 400
            
    except Exception as e:
        logger.error(f'音频处理错误: {str(e)}')
        return jsonify({'error': str(e)}), 500


@app.route('/audio2web', methods=['POST'])
def audio2web():
    """处理音频文件并直接转发给robot"""
    try:
        # 检查是否有文件
        if 'audio' not in request.files:
            logger.error('没有音频文件在请求中')
            return jsonify({'error': '没有音频文件'}), 400
        
        audio_file = request.files['audio']
        logger.info(f'接收到音频文件: {audio_file.filename}')
        
        # 检查文件名
        if audio_file.filename == '':
            logger.error('文件名为空')
            return jsonify({'error': '没有选择文件'}), 400
            
        # 检查文件类型
        if audio_file and allowed_audio_file(audio_file.filename):
            # 读取音频数据
            audio_data = audio_file.read()
            
            # 保存音频文件
            os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)
            
            # 使用时间戳作为文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'audio_{timestamp}.wav'
            audio_path = os.path.join(UPLOAD_FOLDER_AUDIO, filename)
            
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            # 返回可访问的URL路径
            audio_url = f'/audio/{filename}'
            
            return jsonify({
                'message': '音频数据处理成功',
                'status': 'forwarded',
                'dataSize': len(audio_data),
                'url': audio_url
            }), 200
        else:
            logger.error(f'不支持的文件类型: {audio_file.filename}')
            logger.error(f'允许的文件类型: {ALLOWED_AUDIO_EXTENSIONS}')
            return jsonify({'error': '不支持的文件类型'}), 400
            
    except Exception as e:
        logger.error(f'音频处理错误: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """提供音频文件访问"""
    try:
        return send_file(
            os.path.join(UPLOAD_FOLDER_AUDIO, filename),
            mimetype='audio/wav'
        )
    except Exception as e:
        logger.error(f'音频文件访问错误: {str(e)}')
        return jsonify({'error': '音频文件不存在'}), 404

@app.route('/get_audio/<filename>', methods=['GET'])
def get_audio(filename):
    """
    提供语音文件的访问，增加安全性检查和调试日志
    """
    print(f"收到音频文件请求：{filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    print(f"音频文件完整路径：{file_path}")
    print(f"文件是否存在：{os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"错误：音频文件 {filename} 未找到")
        return jsonify({"error": "文件未找到"}), 404
    
    try:
        print(f"准备发送音频文件：{filename}")
        return send_file(
            file_path, 
            mimetype='audio/wav',
            as_attachment=False
        )
    except Exception as e:
        print(f"音频文件读取失败：{str(e)}")
        return jsonify({"error": f"文件读取失败: {str(e)}"}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    上传图片文件，并将图片消息添加到对话历史
    """
    if 'file' not in request.files:
        return jsonify({"error": "未提供文件"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "文件名无效"}), 400

    # 生成唯一文件名和消息ID
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
    message_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # 保存文件
        file.save(file_path)
        
        # 生成文件访问 URL
        image_url = f"/image/{filename}"
        
        # 创建图片消息
        image_message = {
            'id': message_id,
            'role': 'assistant',  # 假设上传的是助手的图片
            'content': '',
            'type': 'image',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_file': filename,
            'image_url': image_url
        }
        
        # 添加到对话历史
        dialogue.append(image_message)
        
        # 限制对话历史长度（可选）
        if len(dialogue) > 100:
            dialogue.pop(0)

        # 广播消息
        socketio.emit('update_dialogue', dialogue)
        
        return jsonify({
            "url": image_url,
            "filename": filename,
            "message_id": message_id
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"文件保存失败: {str(e)}"}), 500

@app.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    """
    提供图片文件的访问，增加安全性检查和调试日志
    """
    print(f"收到图片文件请求：{filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    print(f"图片文件完整路径：{file_path}")
    print(f"文件是否存在：{os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"错误：图片文件 {filename} 未找到")
        return jsonify({"error": "文件未找到"}), 404
    
    try:
        print(f"准备发送图片文件：{filename}")
        return send_file(
            file_path, 
            mimetype='image/png',
            as_attachment=False
        )
    except Exception as e:
        print(f"图片文件读取失败：{str(e)}")
        return jsonify({"error": f"文件读取失败: {str(e)}"}), 500

@app.route('/upload_drawing', methods=['POST'])
def upload_drawing():
    """
    处理用户绘制的图片上传
    """
    try:
        # 获取 base64 编码的图片数据
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"error": "未收到图片数据"}), 400
        
        # 解码 base64 图片
        import base64
        import io
        from PIL import Image
        
        # 移除 base64 前缀
        if image_data.startswith('data:image/png;base64,'):
            image_data = image_data.replace('data:image/png;base64,', '')
        
        # 解码图片
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 生成唯一文件名
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存图片
        image.save(file_path)
        
        # 生成图片访问 URL
        image_url = f"/image/{filename}"
        
        # 创建图片消息
        image_message = {
            'id': str(uuid.uuid4()),
            'role': 'user',
            'content': image_url,
            'type': 'image',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到对话历史
        dialogue.append(image_message)
        trim_dialogue_history()
        
        # 广播消息
        socketio.emit('update_dialogue', dialogue)
        
        return jsonify({
            "url": image_url,
            "filename": filename
        }), 200
    
    except Exception as e:
        logger.error(f"图片上传处理失败: {e}")
        return jsonify({"error": f"图片处理失败: {str(e)}"}), 500

@app.route('/dialogue_history', methods=['GET'])
def get_dialogue_history():
    """
    获取对话历史，支持分页
    """
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_dialogue = dialogue[start:end]
    
    return jsonify({
        "dialogue": paginated_dialogue,
        "total": len(dialogue),
        "page": page,
        "per_page": per_page
    }), 200

def trim_dialogue_history():
    if len(dialogue) > 100:
        dialogue.pop(0)

@socketio.on('video_uploaded')
def handle_video_uploaded(data):
    """
    处理视频上传事件
    """
    try:
        logger.info(f"收到视频上传事件: {data}")
        
        # 可以在这里进行额外的处理，如记录日志、触发其他操作等
        emit('video_ready', data, broadcast=True)
    except Exception as e:
        logger.error(f"处理视频上传事件时发生错误: {e}")

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    上传视频文件，并将视频消息添加到对话历史
    """
    if 'video' not in request.files:
        return jsonify({"error": "未提供视频文件"}), 400

    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({"error": "文件名无效"}), 400

    # 生成唯一文件名和消息ID
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.mp4"
    message_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # 保存文件
        video_file.save(file_path)
        
        # 生成文件访问 URL
        video_url = f"/video/{filename}"
        
        # 创建视频消息
        video_message = {
            'id': message_id,
            'role': 'user',  # 假设上传的是用户的视频
            'content': '',
            'type': 'video',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_file': filename,
            'video_url': video_url
        }
        
        # 添加到对话历史
        dialogue.append(video_message)
        
        # 限制对话历史长度（可选）
        if len(dialogue) > 100:
            dialogue.pop(0)

        # 广播消息
        socketio.emit('update_dialogue', dialogue)
        
        return jsonify({
            "url": video_url,
            "filename": filename,
            "message_id": message_id
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"文件保存失败: {str(e)}"}), 500

@app.route('/video/<filename>')
def get_video(filename):
    """
    提供视频文件访问
    """
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            mimetype='video/mp4'
        )
    except Exception as e:
        logger.error(f"获取视频文件失败: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/socket.io/video_uploaded', methods=['POST'])
def handle_video_uploaded_notification():
    """
    处理来自 main.py 的视频上传通知
    """
    try:
        video_data = request.json
        
        # 使用 SocketIO 触发视频上传事件
        socketio.emit('video_uploaded', video_data)
        
        #logger.info(f"收到视频上传通知: {video_data}")
        
        return jsonify({
            "status": "success", 
            "message": "视频上传通知已接收",
            "video_data": video_data
        }), 200
    except Exception as e:
        logger.error(f"处理视频上传通知时发生错误: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 400

@app.route('/start_main_recording', methods=['POST'])
def start_main_recording():
    """处理开始录音的HTTP请求"""
    try:
        # 广播开始录音信号给所有客户端
        socketio.emit('start_recording')
        #logger.info('已发送开始录音信号')
        return jsonify({"status": "success", "message": "已发送开始录音信号"}), 200
    except Exception as e:
        logger.error(f'发送开始录音信号失败: {str(e)}')
        return jsonify({"status": "error", "message": str(e)}), 500

def generate_self_signed_cert(common_name='localhost'):
    """
    生成自签名 SSL 证书
    :param common_name: 证书的通用名称
    :return: 证书和私钥文件路径
    """
    # 创建证书目录
    cert_dir = os.path.join(os.path.dirname(__file__), 'ssl_certs')
    os.makedirs(cert_dir, exist_ok=True)

    # 证书文件路径
    key_path = os.path.join(cert_dir, 'server.key')
    cert_path = os.path.join(cert_dir, 'server.crt')

    # 如果证书已存在，直接返回
    if os.path.exists(key_path) and os.path.exists(cert_path):
        return key_path, cert_path

    # 创建密钥对
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # 创建自签名证书
    cert = crypto.X509()
    cert.get_subject().C = "CN"
    cert.get_subject().ST = "Beijing"
    cert.get_subject().L = "Beijing"
    cert.get_subject().O = "Medical Interview"
    cert.get_subject().OU = "Development"
    cert.get_subject().CN = common_name

    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10年有效期
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # 写入私钥
    with open(key_path, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

    # 写入证书
    with open(cert_path, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

    return key_path, cert_path

# 在 app 初始化后添加 SSL 支持
# key_path, cert_path = generate_self_signed_cert()
# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

# 修改 run 方法
# 修改 run 方法
"""if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    socketio.run(app, host='localhost', port=5000, debug=True)
"""
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    
    # 生成证书文件路径
    CERT_DIR = os.path.join(os.path.dirname(__file__), 'certs')
    os.makedirs(CERT_DIR, exist_ok=True)

    # 生成证书和密钥
    cert_path = os.path.join(CERT_DIR, 'cert.pem')
    key_path = os.path.join(CERT_DIR, 'key.pem')
    
    # 如果证书不存在，生成新的证书
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        logger.info("正在生成SSL证书...")
        key_path, cert_path = generate_self_signed_cert()
        logger.info("SSL证书生成完成")
    
    ssl_context = (cert_path, key_path)
    
    logger.info("启动HTTPS服务器...")
    logger.info("服务器将在 https://0.0.0.0:5000 上运行")
    
    # 启动服务器
    socketio.run(app, 
                host='0.0.0.0', 
                port=5000, 
                ssl_context=ssl_context,
                debug=True,
                allow_unsafe_werkzeug=True)

@socketio.on('voice_detected')
def handle_voice_detected(data):
    """
    处理语音活动检测事件
    """
    logger.info(f"检测到语音活动: {data}")
    emit('voice_status', {'status': 'detected', 'volume': data.get('volume', 0)}, broadcast=True)

@socketio.on('voice_stopped')
def handle_voice_stopped():
    """
    处理语音停止事件
    """
    logger.info("语音活动已停止")
    emit('voice_status', {'status': 'stopped'}, broadcast=True)

@socketio.on('voice_detection_error')
def handle_voice_detection_error(data):
    """
    处理语音检测错误事件
    """
    logger.error(f"语音检测错误: {data}")
    emit('voice_error', data, broadcast=True)
