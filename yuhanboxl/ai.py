import dashscope
from http import HTTPStatus
from pathlib import Path
import requests
import json
from openai import OpenAI
from dashscope import ImageSynthesis
import google.generativeai as genai
from .global_functions import check_account
import re
import os
from PIL import Image
from io import BytesIO
import base64
import time

# 使用Gemini生成文本，参数分别为：提示文本，主要模型，备用模型。
# 默认主要模型：gemini-2.5-pro-preview-03-25；默认备用模型：gemini-2.0-flash
def gemini_generate_text(query, primary_model='gemini-2.5-pro-preview-03-25', fallback_model='gemini-2.0-flash'):
    # 在脚本中设置代理环境变量，看自己的网络是不能联外网，若使用clash的话，设置如下
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    GOOGLE_API_KEY = check_account("password", "GOOGLE_API_KEY")

    # 确保API密钥已正确设置
    if GOOGLE_API_KEY is None:
        raise ValueError("GOOGLE_API_KEY 未设置，请在数据库'D:\\data\\database\\mm.db'中增加")

    # 请求头设置
    headers = {
        'Content-Type': 'application/json'
    }

    # 请求体设置
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        ]
    }

    # 首先尝试使用主要模型
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{primary_model}:generateContent?key={GOOGLE_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        
        # 提取响应文本
        if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0]:
            text_parts = [part['text'] for part in result['candidates'][0]['content']['parts'] if 'text' in part]
            return ''.join(text_parts)
        return "无法解析响应内容"
        
    except Exception as e:
        try:
            # 如果第一个模型失败，尝试使用备用模型
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={GOOGLE_API_KEY}"
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            
            # 提取响应文本
            if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0]:
                text_parts = [part['text'] for part in result['candidates'][0]['content']['parts'] if 'text' in part]
                return ''.join(text_parts)
            return "无法解析响应内容"
            
        except Exception as e:
            # 如果两个模型都失败，返回错误信息
            return f"生成内容时发生错误: {e}"

# gemini生成图片并保存本地，参数分别为：提示文本，模型，保存目录，是否保存图片。
# 默认模型：gemini-2.0-flash-exp-image-generation；默认保存目录：D:\data\image；默认保存图片：True
def gemini_generate_image(prompt, model="gemini-2.0-flash-exp-image-generation", save_dir=r"D:\data\image", save_image=True):
    """
    生成AI图像并保存到指定目录
    
    参数:
        prompt (str): 用于生成图像的提示文本
        model (str): 使用的Gemini模型名称
        save_dir (str): 图像保存目录
        save_image (bool): 是否保存图像
        
    返回:
        dict: 包含生成结果的字典
    """
    # 创建保存图片的目录
    if save_image:
        os.makedirs(save_dir, exist_ok=True)
    
    # 获取API密钥
    api_key = check_account("password", "GOOGLE_API_KEY")
    
    # API 端点
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    # 构建请求数据
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generation_config": {
            "response_modalities": ["TEXT", "IMAGE"]
        }
    }
    
    # 发送请求
    response = requests.post(
        f"{url}?key={api_key}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    result = {
        "success": False,
        "message": "",
        "image_path": None
    }
    
    # 处理响应
    if response.status_code == 200:
        response_data = response.json()
        result["success"] = True
        
        for part in response_data["candidates"][0]["content"]["parts"]:
            if "text" in part:
                result["message"] = part["text"]
            elif "inlineData" in part:
                # 处理图片数据
                if save_image:
                    image_data = base64.b64decode(part["inlineData"]["data"])
                    image = Image.open(BytesIO(image_data))
                    image_filename = f"gemini_image_{int(time.time())}.png"
                    image_path = os.path.join(save_dir, image_filename)
                    image.save(image_path)
                    result["image_path"] = image_path
    else:
        result["success"] = False
        result["message"] = f"错误: {response.status_code} - {response.text}"
    
    return result


# 用gemini生成文章内容，参数分别为查询内容，主要模型名，备用模型名
def generate_text(query, primary_model='models/gemini-2.5-pro-preview-03-25', fallback_model='models/gemini-2.0-flash'):
    # 在脚本中设置代理环境变量，看自己的网络是不能联外网，若使用clash的话，设置如下
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    GOOGLE_API_KEY = check_account("password", "GOOGLE_API_KEY")

    # 确保API密钥已正确设置
    if GOOGLE_API_KEY is None:
        raise ValueError("GOOGLE_API_KEY 未设置，请在数据库'D:\\data\\database\\mm.db'中增加")

    # 使用API密钥配置SDK
    genai.configure(api_key=GOOGLE_API_KEY)

    # 首先尝试使用主要模型
    try:
        model = genai.GenerativeModel(primary_model)
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        
        try:
            # 如果第一个模型失败，尝试使用备用模型
            model = genai.GenerativeModel(fallback_model)
            response = model.generate_content(query)
            return response.text
        except Exception as e:
            # 如果两个模型都失败，返回错误信息
            return f"生成内容时发生错误: {e}"


# 通过Gemini API生成文本，可调参数，tp温度为0.5
def ask_gemini(question, model_name="models/gemini-2.0-pro-exp"):
    # 代理设置，按需配置
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    # 配置API密钥
    genai.configure(api_key=check_account("password", "GOOGLE_API_KEY"))

    # 设置模型参数
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }

    # 设置内容安全策略
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    # 创建模型实例
    model = genai.GenerativeModel(model_name=model_name,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    
    # 创建聊天会话
    convo = model.start_chat(history=[])
    
    # 发送传入的问题并获取响应
    convo.send_message(question)
    return convo.last.text


# openai-gpt4生成文本
def generate_gpt(content, model="gpt-4-0125-preview"):
    # 替换为您的 OpenAI API 密钥
    client = OpenAI(
        api_key=check_account("password", "OPENAI_API_KEY"),
    )
    
    # 使用 ChatCompletion API 生成文本
    response = client.chat.completions.create(
        model=model,  # 使用传入的模型参数
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    
    # 从响应中提取生成的文本
    text = response.choices[0].message.content  # 正确获取返回的消息内容
    return text


# 从文心api获得内容
def get_content_with_token(prompt):
    api_key = check_account("password", "BAIDU_API_KEY")
    secret_key = check_account("password", "BAIDU_SECRET_KEY")
    def get_access_token():
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key
        }
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()  # 如果请求失败，会抛出一个异常
            return response.json().get("access_token")
        except requests.exceptions.RequestException as e:
            print(f"获取access token时发生错误: {e}")
            return None

    access_token = get_access_token()
    if not access_token:
        return "获取 access token 失败"

    # completions_pro是ERNIE-Bot 4.0，eb-instant是ERNIE-Bot-turbo
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={access_token}"
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
        # 可以添加其他参数
    })
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # 如果请求失败，会抛出一个异常
        response_data = response.json()
        content = response_data.get("result")  # 获取结果文本
        return content if content else "未能获取有效内容"
    except requests.exceptions.RequestException as e:
        return f"请求过程中发生错误: {e}"


# 通过Moonshot AI API生成文本，即kimi生成文本，文档：https://platform.moonshot.cn/docs/api-reference#%E8%AF%B7%E6%B1%82%E5%86%85%E5%AE%B9
# 参数分别为用户消息，模型，温度
def kimi_chat(user_message, model="moonshot-v1-32k", temperature=0.3):
    client = OpenAI(
        api_key=check_account("password", "MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
          # {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
          {"role": "user", "content": user_message}
        ],
        temperature=temperature,
    )

    return completion.choices[0].message.content

# 通义千问api调用   https://help.aliyun.com/document_detail/611472.html
def call_with_messages(prompt, model=dashscope.Generation.Models.qwen_max):
    dashscope.api_key = check_account("password", "DASHSCOPE_API_KEY") 
    try:
        response = dashscope.Generation.call(
            model,     # 模型选择
            prompt=prompt,
            max_tokens=2000,    # 最大字数限制
            top_p=0.8,      # 多样性设置
            repetition_penalty=1.1,     # 重复惩罚设置
            temperature=1.0,      # 随机性设置
            result_format='message',  # 结果格式
        )
        content = response["output"]["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(f"生成故事时发生错误: {e}")
        return None


# 调用阿里云千问3模型的自定义函数，参数分别是：提示词，模型，温度，最大token
def call_qianwen3(prompt, model="qwen-max", temperature=0.7, max_tokens=10000):
    """
    调用阿里云千问3模型的自定义函数
    
    参数:
        prompt (str): 输入提示词
        model (str): 模型名称，可选值包括"qwen-max"、"qwen-plus"、"qwen-turbo"等
        temperature (float): 温度参数，控制输出的随机性，范围0-1
        max_tokens (int): 最大生成token数量
        
    返回:
        dict: 模型返回的完整响应
        str: 如果发生错误，返回错误信息
    """
    # 替换为您的API密钥
    api_key = check_account("password", "DASHSCOPE_API_KEY") 
    
    # API端点
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 请求体
    data = {
        "model": model,
        "input": {
            "prompt": prompt
        },
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }
    
    try:
        # 发送请求
        response = requests.post(url, headers=headers, json=data)
        
        # 检查响应状态
        if response.status_code == 200:
            return response.json()
        else:
            return f"错误: {response.status_code}, {response.text}"
    
    except Exception as e:
        return f"请求过程中发生错误: {str(e)}"
    
# 硅基流动API调用生成内容，参数：prompt（发送给模型的提示词），model（使用的模型名称，默认为"Qwen/QwQ-32B"）
def siliconflow_chat(prompt, model="Qwen/QwQ-32B"):
    """
    调用硅基流动API生成回复，只返回内容部分
    
    参数:
    prompt (str): 发送给模型的提示词
    model (str): 使用的模型名称，默认为"Qwen/QwQ-32B"
    
    返回:
    str: 模型生成的回复内容
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    api_key = check_account("password", "siliconflow_API_KEY")
    
    # 只保留必要的参数
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        # 确保请求成功
        response.raise_for_status()
        
        response_json = response.json()
        
        # 调试信息
        if "choices" not in response_json:
            print(f"API响应缺少choices字段，原始响应: {response_json}")
            return f"API调用失败: {response_json.get('error', {}).get('message', '未知错误')}"
        
        # 只返回content部分
        return response_json["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"请求错误: {str(e)}"
    except KeyError as e:
        return f"响应格式错误: {str(e)}, 响应内容: {response_json}"
    except Exception as e:
        return f"未知错误: {str(e)}"



# 用ollama生成内容-向本地运行的 Ollama 服务发送请求。参数分别是：用户提问内容、模型名称、系统提示词、是否使用流式响应、是否过滤掉思考部分。
def query_ollama(prompt, model="qwen3:4b", system="", stream=False, filter_think=True):
    """
    向本地运行的 Ollama 服务发送请求
    
    参数:
        prompt: 用户提问内容
        model: 要使用的模型名称，默认为"qwen3:4b"
        system: 系统提示词
        stream: 是否使用流式响应
        filter_think: 是否过滤掉思考部分，默认为True
    
    返回:
        模型响应的文本内容，根据filter_think参数决定是否过滤掉思考(think)部分
    """
    
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": stream
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            content = result["message"]["content"]
            
            if filter_think:
                # 过滤掉<think>...</think>或【思考】.*?【/思考】|\[think\].*?\[/think\]|<思考>.*?</思考>或其他类似标记的思考部分
                filtered_content = re.sub(r'<think>.*?</think>|【思考】.*?【/思考】|\[think\].*?\[/think\]|<思考>.*?</思考>', '', content, flags=re.DOTALL)
                # 去除可能留下的多余空行
                filtered_content = re.sub(r'\n\s*\n', '\n\n', filtered_content).strip()
                return filtered_content
            else:
                return content
        else:
            return f"错误: {response.status_code}, {response.text}"
    except Exception as e:
        return f"请求发生异常: {str(e)}"



# 调用OpenRouter API生成内容，参数分别：提示词、模型名称
def call_openrouter(prompt, model="deepseek/deepseek-r1:free"):
    """
    使用OpenRouter API调用不同的AI模型
    
    参数:
        prompt (str): 发送给模型的提示文本
        model (str): 要使用的模型名称，例如 "deepseek/deepseek-r1:free", "anthropic/claude-3-opus-20240229"
        
    返回:
        str: 模型生成的回复内容
    """
    api_key = check_account("password", "OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API密钥必须在环境变量OPENROUTER_API_KEY中设置")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-site.com",  # 替换为您的网站
        "X-Title": "My Application"  # 替换为您的应用名称
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )
    
    result = response.json()
    
    # 提取内容部分
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        return f"获取回复失败: {result}"
# 调用DeepSeek API生成内容，参数分别：提示词、模型名称、系统提示
def call_deepseek(prompt, model = "deepseek-chat", system_prompt = "You are a helpful assistant."):

    """
    调用DeepSeek API生成内容
    
    参数:
        prompt (str): 发送给模型的提示文本
        model (str): 模型名称，默认为'deepseek-chat'（DeepSeek-V3模型），另一模型为'deepseek-reasoner'（DeepSeek-R1推理模型）
        system_prompt (str): 系统提示，定义模型角色和行为
        
    返回:
        str: 模型生成的回复内容
    """
    api_key = check_account("password", "DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API密钥必须在环境变量DEEPSEEK_API_KEY中设置")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )
    
    if response.status_code == 200:
        result = response.json()
        # 提取内容部分
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"获取回复失败: {result}"
    else:
        raise Exception(f"API请求失败: {response.status_code}, {response.text}")


# 统一AI响应生成函数，支持多种AI模型。参数分别：提示词、模型类型、具体使用的模型名称
def generate_ai_response(prompt, model_type="gemini", model=None):
    """
    统一的AI响应生成函数，支持多种AI模型
    
    参数:
        prompt (str): 发送给模型的提示文本
        model_type (str): 模型类型，支持 "gemini", "openrouter", "deepseek"
        model (str): 具体使用的模型名称，不同类型有不同的默认值。主要包括：
            gemini: 'models/gemini-2.5-pro-preview-03-25'
            openrouter: 'deepseek/deepseek-r1:free'
            deepseek: 'deepseek-chat'
    
    返回:
        str: 模型生成的回复内容
    """
    if model_type.lower() == "gemini":
        # Gemini模型默认值
        primary_model = 'models/gemini-2.5-pro-preview-03-25'
        fallback_model = 'models/gemini-2.0-flash-exp'
        
        # 如果指定了特定模型，则使用指定的模型
        if model:
            primary_model = model
        
        GOOGLE_API_KEY = check_account("password", "GOOGLE_API_KEY")

        # 确保API密钥已正确设置
        if GOOGLE_API_KEY is None:
            raise ValueError("GOOGLE_API_KEY 未设置，请在数据库'D:\\data\\database\\mm.db'中增加")

        # 使用API密钥配置SDK
        genai.configure(api_key=GOOGLE_API_KEY)

        # 首先尝试使用主要模型
        try:
            model_instance = genai.GenerativeModel(primary_model)
            response = model_instance.generate_content(prompt)
            return response.text
        except Exception as e:
            if fallback_model != primary_model:
                try:
                    # 如果第一个模型失败，尝试使用备用模型
                    model_instance = genai.GenerativeModel(fallback_model)
                    response = model_instance.generate_content(prompt)
                    return response.text
                except Exception as e2:
                    # 如果两个模型都失败，返回错误信息
                    return f"生成内容时发生错误: {e2}"
            else:
                return f"生成内容时发生错误: {e}"
    
    elif model_type.lower() == "openrouter":
        # OpenRouter模型默认值
        default_model = "deepseek/deepseek-r1:free"
        
        # 如果指定了特定模型，则使用指定的模型
        if model:
            openrouter_model = model
        else:
            openrouter_model = default_model
        
        api_key = check_account("password", "OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API密钥未设置，请在数据库中增加OPENROUTER_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://your-site.com",  # 替换为您的网站
            "X-Title": "My Application"  # 替换为您的应用名称
        }
        
        data = {
            "model": openrouter_model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        result = response.json()
        
        # 提取内容部分
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"获取回复失败: {result}"
    
    elif model_type.lower() == "deepseek":
        # DeepSeek模型默认值
        default_model = "deepseek-chat"
        # 固定的系统提示
        system_prompt = "You are a helpful assistant."
        
        # 如果指定了特定模型，则使用指定的模型
        if model:
            deepseek_model = model
        else:
            deepseek_model = default_model
        
        api_key = check_account("password", "DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API密钥未设置，请在数据库中增加DEEPSEEK_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": deepseek_model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            # 提取内容部分
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"获取回复失败: {result}"
        else:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
    
    else:
        return f"不支持的模型类型: {model_type}，请使用 'gemini'、'openrouter' 或 'deepseek'"
# 阿里万相SD，500张，支持中文，用完需再申请https://help.aliyun.com/zh/dashscope/developer-reference/getting-started-with-stable-diffusion-models?spm=5176.28197632.0.0.97d87e06OPIVDX&disableWebsiteRedirect=true
def generate_and_save_images(prompt, n=1, size='1024*1024', save_path=r'D:\data\image', base_file_name='文章图片'):
    dashscope.api_key = check_account("password", "DASHSCOPE_API_KEY") 
    rsp = dashscope.ImageSynthesis.call(model=dashscope.ImageSynthesis.Models.wanx_v1,
                                        prompt=prompt,
                                        n=n,
                                        size=size)
    urls = []  # 初始化一个空列表来存储URLs
    if rsp.status_code == HTTPStatus.OK:
        for index, result in enumerate(rsp.output.results, start=1):
            if n > 1:
                # 如果生成多张图片，文件名后加上数字
                file_name = f'{base_file_name}{index}.png'
            else:
                # 如果只生成一张图片，直接使用基础文件名
                file_name = f'{base_file_name}.png'
            
            save_file_path = Path(save_path) / file_name
            with open(save_file_path, 'wb+') as f:
                f.write(requests.get(result.url).content)
            urls.append(result.url)  # 将每个图像的URL添加到列表中
        return urls  # 在处理所有图像后返回URL列表
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))

# 使用阿里万相模型生成图片url
def generate_ali_image_url(prompt):
    """
    使用阿里万相模型生成图片并返回URL
    
    参数:
        prompt: 图片生成提示词
    
    返回:
        成功时返回图片URL，失败时返回None
    """
    try:
        rsp = ImageSynthesis.call(api_key = check_account("password", "DASHSCOPE_API_KEY"),
                                model="wanx2.1-t2i-plus",
                                prompt=prompt,
                                n=1,
                                size='1024*1024')
        
        if rsp.status_code == HTTPStatus.OK:
            # 只返回URL，不下载图片
            return rsp.output.results[0].url
        else:
            print('生成图片失败, 状态码: %s, 错误码: %s, 错误信息: %s' %
                (rsp.status_code, rsp.code, rsp.message))
            return None
    except Exception as e:
        print(f"生成图片时发生错误: {e}")
        return None


# 使用minimax生成图片url
def generate_minimax_image_url(prompt, aspect_ratio="1:1", num_images=1, response_format="url", prompt_optimizer=True):
    """
    调用MiniMax API生成图片
    
    参数:
        prompt (str): 图片生成提示词
        aspect_ratio (str): 图片宽高比，默认"16:9"
        num_images (int): 生成图片数量，默认1
        response_format (str): 返回格式，默认"url"
        prompt_optimizer (bool): 是否使用提示词优化，默认True
        
    返回:
        str或list: 生成的图片URL (如果num_images=1)或URL列表(如果num_images>1)
    """
    url = "https://api.minimax.chat/v1/image_generation"
    api_key = check_account("password", "MINIMAX_API_KEY")
    
    payload = json.dumps({
        "model": "image-01", 
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "response_format": response_format,
        "n": num_images,
        "prompt_optimizer": prompt_optimizer
    })
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    result = response.json()
    
    # 从返回结果中提取图片URL
    if result.get('base_resp', {}).get('status_code') == 0:
        image_urls = result.get('data', {}).get('image_urls', [])
        # 当只有一张图片时，直接返回链接字符串而不是列表
        if num_images == 1 and image_urls:
            return image_urls[0]
        return image_urls
    return "" if num_images == 1 else []


# 使用minimax生成图片并保存到本地，参数分别是：图片生成的提示词、图片保存路径、自定义文件名
def generate_minimax_image_download(prompt, save_path=r"D:\data\image", custom_filename=None):
    """
    使用MiniMax API生成图片并保存
    
    参数:
    prompt (str): 图片生成的提示词
    save_path (str): 图片保存路径
    custom_filename (str, optional): 自定义文件名，如不指定则使用API返回的id命名
    
    返回:
    str: 保存的图片完整路径，失败则返回错误信息
    """
    url = "https://api.minimax.chat/v1/image_generation"
    api_key = check_account("password", "MINIMAX_API_KEY")

    payload = json.dumps({
        "model": "image-01", 
        "prompt": prompt,
        "aspect_ratio": "1:1",
        "response_format": "url",
        "n": 1,
        "prompt_optimizer": True
    })
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    
    # 创建下载目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)

    # 下载生成的图片
    if 'data' in response_data and 'image_urls' in response_data['data']:
        image_url = response_data['data']['image_urls'][0]
        img_response = requests.get(image_url)
        
        if img_response.status_code == 200:
            # 使用自定义文件名或API返回的id命名
            if custom_filename:
                filename = custom_filename
            else:
                filename = f"minimax_image_{response_data['id']}.png"
            
            filepath = os.path.join(save_path, filename)
            
            # 保存图片
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
            
            return filepath
        else:
            return f"下载图片失败，状态码: {img_response.status_code}"
    else:
        return "API响应中未找到图片URL"

if __name__ == "__main__":
    print(generate_text("你好"))
