import akshare as ak
import pywencai  # 导入pywencai模块
import sqlite3  # 导入sqlite3模块
import pandas as pd  # 导入pandas模块
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import rcParams
import google.generativeai as genai
import os
import sys
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import logging
import dashscope
from http import HTTPStatus
from pathlib import Path
import requests
import numpy as np
from scipy import stats
import math
from datetime import datetime, timedelta
import pymysql
import json
from Bing_Image_Generator import Image_Gen
import time
from openai import OpenAI
from bing_brush import BingBrush
import random
import re
import html2text
import feedparser
from bs4 import BeautifulSoup as bs
import base64


# 读取mm.db，查询账号密码
def check_account(column_name, project_name):
    db_path = r"D:\wenjian\python\data\data\mm.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 使用 column_name 参数构建查询语句
        query = f"""
        SELECT {column_name}
        FROM connect_account_password
        WHERE project_name = ?
        """

        cursor.execute(query, (project_name,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return result[0]  # 返回查询结果而不是列表
        else:
            return None

    except Exception as e:
        print(f"数据库操作错误：{e}")
        return None

# 上传图片到GitHub作为图床，参数为GitHub仓库和本地图片路径
def upload_image_to_github(repo, image_path, commit_message="Upload image"):
    """
    将图片上传到 GitHub 仓库，如果图片已经存在，则重命名后上传。
    
    :param repo: GitHub 仓库名称，格式为 '用户名/仓库名'。
    :param image_path: 本地图像文件的路径。
    :param commit_message: 提交时的说明信息。
    :return: 上传图片的 URL，带有自定义域名。
    """
    try:
        # 获取 GitHub 令牌，确保你的令牌是正确的
        token = check_account("password", "github_token")

        # 读取图像并将其编码为 base64 格式
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 从 image_path 中提取图片文件名和扩展名
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        
        # 获取当前日期
        current_date = datetime.now().strftime("%Y%m%d")
        
        # 将文件名重命名为 "原来名字_日期" 的格式
        new_name = f"{name}_{current_date}{ext}"
        path = f"{new_name}"

        # GitHub API 创建或更新文件的 URL
        url = f"https://api.github.com/repos/{repo}/contents/{path}"

        # 准备请求头
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # 检查文件是否存在
        response = requests.get(url, headers=headers)
        counter = 1

        while response.status_code == 200:
            # 文件已存在，增加计数器并修改文件名
            new_name = f"{name}_{current_date}_{counter}{ext}"
            path = f"{new_name}"
            url = f"https://api.github.com/repos/{repo}/contents/{path}"
            response = requests.get(url, headers=headers)
            counter += 1

        # 准备请求数据
        data = {
            "message": commit_message,
            "content": image_base64,
        }

        # 发送请求以上传图片
        response = requests.put(url, headers=headers, json=data)

        if response.status_code in [201, 200]:
            # 返回图片的自定义域名 URL
            custom_domain = "image.sanrenjz.com"
            final_image_name = new_name if counter > 1 else new_name
            return f"https://{custom_domain}/{final_image_name}"
        else:
            # 处理可能的错误
            raise Exception(f"上传图片时出错: {response.json()}")
    
    except Exception as e:
        # 捕获并处理所有异常
        print(f"图片上传失败: {e}")
        return None


# 下载链接的图片放到内存，然后上传到github
def upload_bing_image_to_github(repo, image_url, commit_message="Upload image"):
    """
    将图片从URL上传到 GitHub 仓库，并将其命名为“日期+.png”。
    
    :param repo: GitHub 仓库名称，格式为 '用户名/仓库名'。
    :param image_url: 图片的URL。
    :param commit_message: 提交时的说明信息。
    :return: 上传图片的 URL，带有自定义域名。
    """
    try:
        # 获取 GitHub 令牌，确保你的令牌是正确的
        token = check_account("password", "github_token")

        # 从 URL 读取图片数据
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
        else:
            raise Exception(f"图片下载失败，状态码: {response.status_code}")

        # 将图片数据编码为 base64 格式
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 生成图片文件名，以“日期+.png”格式命名
        current_date = datetime.now().strftime("%Y%m%d")
        image_name = f"{current_date}.png"
        path = image_name

        # GitHub API 创建或更新文件的 URL
        url = f"https://api.github.com/repos/{repo}/contents/{path}"

        # 准备请求头
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # 检查文件是否存在
        response = requests.get(url, headers=headers)
        counter = 1

        while response.status_code == 200:
            # 文件已存在，增加计数器并修改文件名
            new_name = f"{current_date}_{counter}.png"
            path = f"{new_name}"
            url = f"https://api.github.com/repos/{repo}/contents/{path}"
            response = requests.get(url, headers=headers)
            counter += 1

        # 准备请求数据
        data = {
            "message": commit_message,
            "content": image_base64,
        }

        # 发送请求以上传图片
        response = requests.put(url, headers=headers, json=data)

        if response.status_code in [201, 200]:
            # 返回图片的自定义域名 URL
            custom_domain = "image.sanrenjz.com"
            # 根据是否存在文件返回正确的名称
            final_image_name = new_name if counter > 1 else image_name
            return f"https://{custom_domain}/{final_image_name}"
        else:
            # 处理可能的错误
            raise Exception(f"上传图片时出错: {response.json()}")
    
    except Exception as e:
        # 捕获并处理所有异常
        print(f"图片上传失败: {e}")
        return None






# 上传文件到 Lsky Pro 并返回 Markdown 链接，策略ID为2时，返回的链接为minlo图床的链接
def upload_to_lsky_pro(file_path):
    # 上传信息
    upload_url = "http://192.168.31.143:7791/api/v1/upload"
    token = check_account("password", upload_url)
    policy_id = 2  # minlo图床

    # 读取文件
    files = {'file': open(file_path, 'rb')}
    # 设置请求头
    headers = {
        'Authorization': f'Bearer {token}'
    }

    # 设置请求体
    data = {
        'strategy_id': policy_id
    }

    # 发起请求
    response = requests.post(upload_url, headers=headers, files=files, data=data)

    # 处理响应
    if response.status_code == 200:
        response_data = response.json()
        if response_data['status']:
            return response_data['data']['links']['markdown']
        else:
            raise Exception(f"上传失败: {response_data['message']}")
    else:
        raise Exception(f"上传失败: {response.status_code} {response.text}")


# 上传文件到腾讯云COS
def upload_to_cos(bucket, file_name, region='ap-guangzhou', cos_folder='temporary/'):
    """
    上传文件到腾讯云COS指定桶和文件夹中。

    :param bucket: 要上传到的桶名称。
    :param file_name: 本地文件的完整路径。
    :param region: 桶所在的区域，默认为'ap-guangzhou'。
    :param cos_folder: COS中的目标文件夹，默认为'temporary/'。
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # 从环境变量中获取secret_id和secret_key
    secret_id = check_account("username", "TENCENT_CLOUD_COS")  # 确保已经设置了环境变量COS_SECRET_ID
    secret_key = check_account("password", "TENCENT_CLOUD_COS")  # 确保已经设置了环境变量COS_SECRET_KEY
    token = None  # 使用临时密钥需要传入Token，默认为空,可不填
    scheme = 'https'  # 指定使用 http/https 协议来访问COS，默认为https, 可不填

    # 获取客户端对象
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
    client = CosS3Client(config)

    # 从本地文件路径中提取文件名
    base_file_name = os.path.basename(file_name)

    # 上传到COS的路径，包括文件夹和文件名
    cos_path = cos_folder + base_file_name

    # 上传文件
    response = client.upload_file(
        Bucket=bucket,
        LocalFilePath=file_name,
        Key=cos_path,
        PartSize=10,
        MAXThread=10
    )
    print(response['ETag'])

# 下载公众号文章
class Url2Md(object):
    """根据微信文章链接下载为本地Markdown文件"""

    def __init__(self, img_path="D:\\wenjian\\obsidian\\RSS图片"):
        self.img_path = img_path
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        self.data_src_re = re.compile(r'data-src="(.*?)"')
        self.data_croporisrc_re = re.compile(r'data-croporisrc="(.*?)"')
        self.src_re = re.compile(r'src="(.*?)"')

    @staticmethod
    def replace_name(title):
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        title = re.sub(rstr, "", title).replace("|", "").replace("\n", "")
        return title

    def download_img(self, url):
        # 根据链接提取图片名
        name = "{}.{}".format(url.split("/")[-2], url.split("/")[3].split("_")[-1])
        save_path = os.path.join(self.img_path, name)
        # 如果该图片已被下载，可以无需再下载，直接返回绝对路径即可
        if os.path.isfile(save_path):
            return os.path.abspath(save_path)
    
        response = requests.get(url)
        img = response.content
        with open(save_path, "wb") as f:
            f.write(img)
        return os.path.abspath(save_path)

    def html_to_md(self, html):
        soup = bs(html, "html.parser")

        # 处理图片链接，替换为绝对路径
        for img in soup.find_all('img'):
            img_url = img.get('data-src') or img.get('src')
            if img_url:
                local_img_path = self.download_img(img_url)
                img['src'] = local_img_path
                del img['data-src']  # 删除data-src属性，如果存在的话

        # 使用html2text转换器
        h = html2text.HTML2Text()
        h.ignore_links = False
        text = h.handle(str(soup))  # 需要将soup对象转换回字符串

        return text

    @staticmethod
    def get_title(html):
        soup = bs(html, "html.parser")
        title_tag = soup.find('h1') or soup.find('h2')
        if title_tag:
            return title_tag.get_text().strip()
        return "Untitled"
    
    def run(self, url):
        try:
            response = requests.get(url)
            html = response.content.decode('utf-8')
            soup = bs(html, 'html.parser')

            # 尝试从文章内容中提取标题
            title = self.get_title(html)
            title = self.replace_name(title)

            md_text = self.html_to_md(html)

            # Remove everything above two underscores (inclusive)
            md_text = re.sub(r'.*____\n', '', md_text, count=1, flags=re.DOTALL)

            # Remove everything after and including "预览时标签不可点"
            md_text = re.sub(r'预览时标签不可点.*', '', md_text, flags=re.DOTALL)

            # 保存Markdown文件
            md_filename = fr"D:\wenjian\obsidian\笔记\归纳检索\RSS订阅\{title}.md"
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(md_text)

            return title
        except Exception as e:
            return f"处理URL {url} 时发生错误: {e}"

# 获取RSS订阅中的新链接并更新数据库
def fetch_new_links_and_update_db(rss_url, db_path):
    new_links = []  # 用于存储数据库中不存在的新链接

    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute('''CREATE TABLE IF NOT EXISTS RSS订阅
                          (链接 TEXT PRIMARY KEY)''')
        
        # 解析RSS feed
        feed = feedparser.parse(rss_url)
        
        # 检查新链接，同时更新数据库
        if feed.entries:
            for entry in feed.entries:
                link = entry.get('link', None)
                if link:
                    # 检查链接是否已存在于数据库中
                    cursor.execute('SELECT * FROM RSS订阅 WHERE 链接=?', (link,))
                    if cursor.fetchone() is None:
                        # 如果链接不存在，则添加到列表和数据库中
                        new_links.append(link)
                        cursor.execute('INSERT INTO RSS订阅 (链接) VALUES (?)', (link,))
        
        # 提交事务并关闭连接
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"发生错误: {e}")
    
    return new_links


# 爬取微信公众号RSS订阅中的内容，并保存到本地，然后读取本地文件内容
def fetch_content_from_md_files(rss_url):
    # 固定的数据库路径和文件夹路径
    db_path = 'D:\\wenjian\\python\\smart\\data\\article.db'
    folder_path = 'D:\\wenjian\\obsidian\\笔记\\归纳检索\\RSS订阅'
    
    # 调用函数，传入RSS URL和数据库路径
    new_links = fetch_new_links_and_update_db(rss_url, db_path)
    
    um = Url2Md()
    all_content = ""  # 初始化一个空字符串，用于累积所有找到的文件内容
    for url in new_links:
        result = um.run(url)
        matched = False  # 标记是否找到匹配文件
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.md') and result in file:
                # 构建完整的文件路径
                file_path = os.path.join(folder_path, file)
                # 打开并读取文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    all_content += content + "\n\n"  # 将文件内容累积到字符串中，并添加分隔
                    matched = True
                    break  # 找到匹配的文件后停止查找
        if not matched:
            print(f'No matching .md file found for result: {result}')
    return all_content.strip()  # 返回累积的内容字符串，移除尾部的空白字符


# 用gemini生成文章内容
def generate_text(query):
    # 在脚本中设置代理环境变量，看自己的网络是不能联外网，若使用clash的话，设置如下
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    GOOGLE_API_KEY = check_account("password", "GOOGLE_API_KEY")

    # 确保API密钥已正确设置
    if GOOGLE_API_KEY is None:
        raise ValueError("请设置环境变量 GOOGLE_API_KEY 为您的API密钥。")

    # 使用API密钥配置SDK
    genai.configure(api_key=GOOGLE_API_KEY)

    # 选择一个模型并创建一个GenerativeModel实例
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # 使用模型生成内容
    response = model.generate_content(query)

    # 返回生成的文本
    return response.text


# 通过Gemini API生成文本，可调参数，tp温度为0.5
def ask_gemini(question):
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
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    
    # 创建聊天会话
    convo = model.start_chat(history=[])
    
    # 发送传入的问题并获取响应
    convo.send_message(question)
    return convo.last.text


# openai-gpt4生成文本
def generate_gpt(content):
    # 替换为您的 OpenAI API 密钥
    client = OpenAI(
        api_key=check_account("password", "OPENAI_API_KEY"),
    )
    
    # 使用 ChatCompletion API 生成文本
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",  # 根据实际可用模型来替换，这里假设模型是 gpt-4
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
def kimi_chat(user_message):
    client = OpenAI(
        api_key=check_account("password", "MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
    )
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=[
          # {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
          {"role": "user", "content": user_message}
        ],
        temperature=0.3,
    )

    return completion.choices[0].message.content

# 通义千问api调用   https://help.aliyun.com/document_detail/611472.html
def call_with_messages(prompt):
    dashscope.api_key = check_account("password", "DASHSCOPE_API_KEY") 
    try:
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_max,     # 模型选择
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


# bing Designer文本生成图片
def generate_image_links(prompt):
    try:
        brush = BingBrush(cookie='D:\\wenjian\\python\\data\\json\\bingbrush.json')  # cookie的路径
        image_urls = brush.process(prompt)
        # 使用 Markdown 格式和字符串格式化来生成图片链接
        markdown_images = []
        for i, url in enumerate(image_urls, start=1):
            # 跳过以.svg结尾的链接
            if url.endswith('.svg') or url.endswith('.js'):
                continue
            markdown_images.append(f'![图片{i}]({url})')
        
        # 返回结果
        return "\n".join(markdown_images)
    except Exception as e:
        # 如果出现异常，打印异常信息（可选）并返回空字符串或None
        print(f"生成图片链接时出现错误: {e}")
        return None

# bing Designer文本生成图片，并提取第一张图的链接
def generate_first_image_links(prompt):
    try:
        brush = BingBrush(cookie='D:\\wenjian\\python\\data\\json\\bingbrush.json')  # cookie的路径
        image_urls = brush.process(prompt)
        # 返回第一个有效的图片链接
        for url in image_urls:
            # 跳过以.svg结尾的链接
            if url.endswith('.svg') or url.endswith('.js'):
                continue
            return url
        # 如果没有找到有效的链接，返回空字符串
        return ""
    except Exception as e:
        # 如果出现异常，打印异常信息（可选）并返回空字符串或None
        print(f"生成图片链接时出现错误: {e}")
        return None









# 读取指定数据库中的指定表，并返回表的内容
def get_table(database_path, table_name):
    """
    读取指定数据库中的指定表，并返回表的内容
    """

    try:
        # 连接到数据库
        conn = sqlite3.connect(database_path)

        # 执行SELECT查询，获取表中的所有行
        query = f'SELECT * FROM "{table_name}"'
        df = pd.read_sql_query(query, conn)

        return df

    except sqlite3.Error as e:
        print("数据库操作错误:", e)

    finally:
        # 关闭连接
        conn.close()


# 将文本保存为.md文件
def save_as_md(generated_text, file_path):
    """
    将生成的文本保存为.md文件。

    :param generated_text: 生成的文本。
    :param file_path: 要保存的文件路径。
    """
    # 使用'w'模式打开文件，如果文件已存在则会被覆盖
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(generated_text)













