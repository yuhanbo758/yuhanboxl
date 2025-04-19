
import dashscope
from http import HTTPStatus
from pathlib import Path
import requests
import json
from openai import OpenAI
import google.generativeai as genai
from global_functions import check_account


# 用gemini生成文章内容，参数分别为查询内容，主要模型名，备用模型名
def generate_text(query, primary_model='models/gemini-2.5-pro-preview-03-25', fallback_model='models/gemini-2.0-flash-exp'):
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

# 阿里万相SD，500张，支持中文，用完需再申请https://help.aliyun.com/zh/dashscope/developer-reference/getting-started-with-stable-diffusion-models?spm=5176.28197632.0.0.97d87e06OPIVDX&disableWebsiteRedirect=true
def generate_and_save_images(prompt, n=1, size='1024*1024', save_path=r'D:\wenjian\python\smart\data\image', base_file_name='文章图片'):
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


if __name__ == "__main__":
    print(generate_text("你好"))
