
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
import global_functions as gf
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

if __name__ == '__main__':
    print(generate_text("写一首诗"))