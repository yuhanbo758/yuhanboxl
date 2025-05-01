
# 读取Markdown文件
def read_md_file(file_path):
    """
    读取Markdown文件内容
    
    参数:
        file_path (str): Markdown文件的路径
        
    返回:
        str: 文件内容，如果读取失败则返回空字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"读取Markdown文件时发生错误: {e}")
        return ""
    
if __name__ == "__main__":
    file_path = "example.md"  # 替换为实际的Markdown文件路径
    content = read_md_file(file_path)
    print(content)