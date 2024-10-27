import requests
import sqlite3
import os
import re
from datetime import datetime, timedelta
import global_functions as gf

# 更新 notion 页面
def process_notion_ids(folder_path, db_path):
    time_threshold = datetime.now() - timedelta(days=3)
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


# 更新 notion 页面，封面和摘要
def update_notion_page(page_id):
    NOTION_TOKEN = gf.check_account("password", "notion_token")
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


# 遍历文件夹，获取所有文件名，并从文件名中提取 notion_id，更新notion的封面和摘要
def update_notion_pages(folder_path, db_path):
    new_ids = process_notion_ids(folder_path, db_path)
    for notion_id in new_ids:
        update_notion_page(notion_id)

if __name__ == "__main__":
    # 更新notion的封面和摘要，参数为获取notion id文件夹路径和存储id数据库路径
    update_notion_pages(r"D:\wenjian\obsidian\笔记\自媒体\AI文章", r"D:\wenjian\python\smart\data\article.db")
