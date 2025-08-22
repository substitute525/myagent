import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def query_url(url: str) -> str:
    """
    仅可用于查询用户指定的url网页内容。
    :param url: 网页地址
    :return: 网页文本内容
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        return soup.get_text(separator='\n')[:2000]  # 限制返回长度
    except Exception as e:
        return f"[Error] {e}"

@tool
def search_web(keyword: str, topn: int = 3) -> list:
    """
    用关键字搜索网页，返回topn条结果，该工具禁止通过URL查询
    :param keyword: 搜索关键字
    :param topn: 返回结果数
    :return: [{'title':..., 'url':..., 'snippet':...}]
    """
    from ddgs import DDGS
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(keyword, max_results=topn):
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')
                })
        return results
    except Exception as e:
        return [f"[Error] {e}"] 