import sqlite3  # 导入sqlite3模块
import pandas as pd  # 导入pandas模块
import os




# 创建账号密码数据库及表，参数为数据库路径
def create_account_database(db_data_path):
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(db_data_path), exist_ok=True)
        
        # 创建数据库连接
        conn = sqlite3.connect(db_data_path)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS connect_account_password (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            username TEXT,
            password TEXT
        )
        ''')
        
        # 提交更改并关闭连接
        conn.commit()
        conn.close()
        
        print(f"已成功创建数据库 {db_data_path} 和表 connect_account_password")
    except Exception as e:
        print(f"创建数据库时出错：{e}")

# 读取mm.db，查询账号密码，参数分别为列名（指定为username，password），项目名称（mm）。
def check_account(column_name, project_name):
    db_path = r"D:\data\database\mm.db"
    try:
        # 如果数据库不存在，先创建数据库
        if not os.path.exists(db_path):
            create_account_database(db_path)
            
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

# 向数据库添加账号密码，参数为项目名称，用户名，密码
def add_account(project_name, username, password):
    db_path = r"D:\data\database\mm.db"
    try:
        # 如果数据库不存在，先创建数据库
        if not os.path.exists(db_path):
            create_account_database(db_path)
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 插入数据
        cursor.execute('''
        INSERT INTO connect_account_password (project_name, username, password)
        VALUES (?, ?, ?)
        ''', (project_name, username, password))
        
        # 提交更改并关闭连接
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"已成功添加项目 {project_name} 的账号信息")
        return True
    except Exception as e:
        print(f"添加账号信息时出错：{e}")
        return False




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


if __name__ == "__main__":
    get_table(r"D:\data\database\article.db", "RSS订阅")



