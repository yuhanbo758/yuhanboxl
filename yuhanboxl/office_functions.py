import requests
import sqlite3
import os
import re
from datetime import datetime, timedelta
import global_functions as gf
import sys
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import logging
import re
import html2text
import feedparser
from bs4 import BeautifulSoup as bs
import base64
from datetime import datetime, timedelta
import requests
from global_functions import check_account

# 更新 notion 页面，参数分别为获取notion id文件夹路径和存储id数据库路径
def process_notion_ids(folder_path, db_path):
    time_threshold = datetime.now() - timedelta(days=5)
    pattern = r"NotionID-博客:\s*([a-f0-9-]+)"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    new_ids = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_mtime > time_threshold:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    match = re.search(pattern, content)
                    if match:
                        notion_id = match.group(1)
                        cursor.execute("SELECT 1 FROM notion_id WHERE ID = ?", (notion_id,))
                        exists = cursor.fetchone()

                        if not exists:
                            new_ids.append(notion_id)
                            print(f"新增 NotionID: {notion_id}")

                            try:
                                update_notion_page(notion_id)
                            except Exception as e:
                                print(f"更新 Notion 页面失败: {e}")

                            try:
                                cursor.execute("INSERT INTO notion_id (ID) VALUES (?)", (notion_id,))
                                conn.commit()
                            except Exception as e:
                                print(f"写入数据库失败: {e}")

    conn.close()
    return new_ids


# 更新 notion 页面，封面和摘要，参数为notion id
def update_notion_page(page_id):
    NOTION_TOKEN = check_account("password", "notion_token")
    NOTION_API_URL = f"https://api.notion.com/v1/pages/{page_id}"

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    def get_page_content(page_id):
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        response = requests.get(url, headers=headers)
        return response.json()

    def update_page_properties(page_id, image_url=None, summary=None):
        data = {}

        if image_url:
            data["cover"] = {
                "external": {
                    "url": image_url
                }
            }
        if summary:
            data["properties"] = {
                "summary": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": summary
                            }
                        }
                    ]
                }
            }

        if data:
            response = requests.patch(NOTION_API_URL, json=data, headers=headers)
            return response.status_code, response.json()
        else:
            print(f"No valid data to update for page {page_id}. Skipping update.")
            return None, None

    page_content = get_page_content(page_id)

    if 'results' not in page_content:
        print(f"Error: 'results' key not found in page content for page_id {page_id}")
        return

    image_url = None
    summary = ""

    for block in page_content['results']:
        if block['type'] == 'image' and not image_url:
            image_url = block['image']['external']['url']
        if block['type'] == 'paragraph' and not summary:
            if block['paragraph']['rich_text']:
                text = block['paragraph']['rich_text'][0]['text']['content']
                summary = text[:100]
                print(f"Extracted text: {summary}")

    # 更新页面，只要有图片或摘要中的一个
    if image_url or summary:
        status, result = update_page_properties(page_id, image_url, summary)
    else:
        print(f"No valid image URL or summary found for page {page_id}. Skipping update.")


# 遍历文件夹，获取所有文件名，并从文件名中提取 notion_id，更新notion的封面和摘要，参数分别为获取notion id文件夹路径和存储id数据库路径
def update_notion_pages(folder_path, db_path):
    new_ids = process_notion_ids(folder_path, db_path)
    for notion_id in new_ids:
        update_notion_page(notion_id)



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


# 下载链接的图片放到内存，然后上传到github，参数分别为仓库名，图片url，提交信息
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






# 上传文件到 Lsky Pro 并返回 Markdown 链接，策略ID为2时，返回的链接为minlo图床的链接，参数为文件路径和上传url
def upload_to_lsky_pro(file_path, upload_url = "http://192.168.31.143:7791/api/v1/upload"):
    # 上传信息
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
        try:
            name = "{}.{}".format(url.split("/")[-2], url.split("/")[3].split("_")[-1])
            save_path = os.path.join(self.img_path, name)
            # 如果该图片已被下载，可以无需再下载，直接返回绝对路径即可
            if os.path.isfile(save_path):
                return os.path.abspath(save_path)
        
            response = requests.get(url, timeout=30)
            img = response.content
            with open(save_path, "wb") as f:
                f.write(img)
            return os.path.abspath(save_path)
        except Exception as e:
            print(f"下载图片失败 {url}: {e}")
            return url  # 如果下载失败，返回原始URL

    def html_to_md(self, html):
        soup = bs(html, "html.parser")

        # 处理图片链接，替换为绝对路径
        for img in soup.find_all('img'):
            img_url = img.get('data-src') or img.get('src')
            if img_url:
                try:
                    local_img_path = self.download_img(img_url)
                    img['src'] = local_img_path
                    if 'data-src' in img.attrs:
                        del img['data-src']  # 删除data-src属性，如果存在的话
                except Exception as e:
                    print(f"处理图片链接失败: {e}")

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
            # 设置请求头模拟浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Connection': 'keep-alive',
                'Referer': 'https://www.google.com/'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            html = response.content.decode('utf-8')
            
            # 检查是否是微信环境限制页面
            if "环境异常" in html or "访问过于频繁" in html or "请在微信客户端打开链接" in html:
                print(f"微信文章访问受限，请在微信客户端打开链接: {url}")
                # 使用特殊标题标记这是一个无法访问的链接
                return "微信文章无法访问"
                
            soup = bs(html, 'html.parser')

            # 尝试从文章内容中提取标题
            title = self.get_title(html)
            if title == "Untitled" or "环境异常" in title:
                print(f"无法获取标题或环境受限: {url}")
                return "微信文章无法访问"
                
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
            print(f"处理URL {url} 时发生错误: {e}")
            return f"处理URL {url} 时发生错误: {e}"

# 获取RSS订阅中的新链接并更新数据库，参数分别为RSS订阅链接，数据库路径
def fetch_new_links_and_update_db(rss_url, article_db_path=r"D:\data\database\article.db"):
    new_links = []  # 用于存储数据库中不存在的新链接

    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(article_db_path)
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


# 爬取微信公众号RSS订阅中的内容，并保存到本地，然后读取本地文件内容，参数分别为RSS订阅链接，数据库路径，本地文件夹路径
def fetch_content_from_md_files(rss_url, article_db_path=r"D:\data\database\article.db", folder_path=r"D:\wenjian\obsidian\笔记\归纳检索\RSS订阅"):
    
    # 调用函数，传入RSS URL和数据库路径
    new_links = fetch_new_links_and_update_db(rss_url, article_db_path)
    
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



# 上传文件到腾讯云COS，参数分别为桶名，文件路径，区域，目标文件夹
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

    
# 上传图片到腾讯云COS，参数分别为桶名，图片URL，区域，目标文件夹
def upload_bing_image_to_cos(bucket, image_url, region='ap-shanghai', cos_folder='image/'):
    """
    将图片从URL上传到腾讯云COS。
    
    :param bucket: 要上传到的桶名称。
    :param image_url: 图片的URL。
    :param region: 桶所在的区域，默认为'ap-shanghai'。
    :param cos_folder: COS中的目标文件夹，默认为'image/'。
    :return: 上传图片的URL。
    """
    try:
        # 从 URL 下载图片
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"图片下载失败，状态码: {response.status_code}")

        # 生成临时文件名（使用随机字符串避免冲突）
        temp_file = f"temp_{os.urandom(4).hex()}.png"
        
        # 将图片内容保存到临时文件
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        try:
            # 使用upload_to_cos函数上传文件
            result_url = upload_to_cos(bucket, temp_file, region, cos_folder)
            return result_url
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)

    except Exception as e:
        print(f"图片上传失败: {e}")
        return None


if __name__ == "__main__":
    # 更新notion的封面和摘要，参数为获取notion id文件夹路径和存储id数据库路径
    update_notion_pages(r"D:\wenjian\obsidian\笔记\自媒体\AI文章", r"D:\wenjian\python\smart\data\article.db")
