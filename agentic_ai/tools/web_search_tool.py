import requests
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import json
from urllib.parse import quote_plus
from .base_tool import BaseCustomTool, ToolParameter
from config import Config


class WebSearchTool(BaseCustomTool):
    """Web search tool for finding current information on the internet"""
    
    tool_name = "web_search"
    tool_description = "Search the internet for current information and news using web search"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query to find relevant web pages",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of search results to return",
            required=False,
            default=5
        ),
        ToolParameter(
            name="include_snippets",
            type="boolean",
            description="Whether to include content snippets from the web pages",
            required=False,
            default=True
        )
    ]
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo's instant answer API"""
        try:
            # DuckDuckGo instant answer API
            api_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Add instant answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Instant Answer'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': 'DuckDuckGo Instant Answer'
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:num_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else 'Related Topic',
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'DuckDuckGo Related'
                    })
            
            # If we don't have enough results, try the HTML search
            if len(results) < num_results:
                html_results = self._search_duckduckgo_html(query, num_results - len(results))
                results.extend(html_results)
            
            return results[:num_results]
            
        except Exception as e:
            print(f"DuckDuckGo API search error: {e}")
            # Fallback to HTML search
            return self._search_duckduckgo_html(query, num_results)
    
    def _search_duckduckgo_html(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Fallback HTML search for DuckDuckGo"""
        try:
            search_url = f"https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            response = self.session.post(search_url, data=params, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find search results
            result_divs = soup.find_all('div', class_='result')[:num_results]
            
            for div in result_divs:
                title_link = div.find('a', class_='result__a')
                snippet_div = div.find('div', class_='result__snippet')
                
                if title_link:
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ''
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'DuckDuckGo Search'
                    })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo HTML search error: {e}")
            return []
    
    def _get_page_content(self, url: str, max_chars: int = 1000) -> str:
        """Extract content from a web page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_chars] + "..." if len(text) > max_chars else text
            
        except Exception as e:
            return f"Error extracting content: {str(e)}"
    
    def _run(self, query: str, num_results: int = 5, include_snippets: bool = True) -> Dict[str, Any]:
        """Execute web search"""
        try:
            params = self.validate_parameters({
                "query": query,
                "num_results": num_results,
                "include_snippets": include_snippets
            })
            
            # Perform search
            search_results = self._search_duckduckgo(params["query"], params["num_results"])
            
            if not search_results:
                return {
                    "success": False,
                    "error": "No search results found",
                    "results": []
                }
            
            # Enhance results with content if requested
            enhanced_results = []
            for result in search_results:
                enhanced_result = result.copy()
                
                if params["include_snippets"] and result.get('url') and not result.get('snippet'):
                    # Get page content if snippet is not available
                    content = self._get_page_content(result['url'], 500)
                    enhanced_result['content'] = content
                
                enhanced_results.append(enhanced_result)
            
            return {
                "success": True,
                "query": params["query"],
                "results": enhanced_results,
                "total_results": len(enhanced_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _arun(self, query: str, num_results: int = 5, include_snippets: bool = True) -> Dict[str, Any]:
        """Async version of web search"""
        return self._run(query, num_results, include_snippets)
    
    def search_news(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for news articles"""
        news_query = f"{query} news"
        return self._run(news_query, num_results, True)
    
    def search_academic(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for academic/research content"""
        academic_query = f"{query} research paper academic"
        return self._run(academic_query, num_results, True)