

# 从 office_functions.py 导入函数
from .office_functions import (
    process_notion_ids,
    update_notion_page,
    update_notion_pages,
    upload_image_to_github,
    upload_bing_image_to_github,
    upload_to_lsky_pro,
    upload_to_cos,
    fetch_new_links_and_update_db,
    fetch_content_from_md_files,
    upload_bing_image_to_cos,
    Url2Md
)

# 从 global_functions.py 导入函数
from .global_functions import (
    check_account,
    create_account_database,
    add_account,
    get_table,
    save_as_md
)

# 从 ai.py 导入函数
from .ai import (
    generate_text,
    ask_gemini,
    generate_gpt,
    get_content_with_token,
    kimi_chat,
    call_with_messages,
    generate_and_save_images
)

# 定义 __all__ 变量来明确指定哪些函数应该被导出
__all__ = [
    # office_functions
    'process_notion_ids',
    'update_notion_page',
    'update_notion_pages',
    'upload_image_to_github',
    'upload_bing_image_to_github',
    'upload_to_lsky_pro',
    'upload_to_cos',
    'fetch_new_links_and_update_db',
    'fetch_content_from_md_files',
    'upload_bing_image_to_cos',
    'Url2Md',
    
    # global_functions
    'check_account',
    'create_account_database',
    'add_account',
    'get_table',
    'save_as_md',
    
    # ai.py
    'generate_text',
    'ask_gemini',
    'generate_gpt',
    'get_content_with_token',
    'kimi_chat',
    'call_with_messages',
    'generate_and_save_images'
]

