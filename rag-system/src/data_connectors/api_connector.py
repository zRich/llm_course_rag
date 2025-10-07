from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
import logging
import requests
import time
import json
from urllib.parse import urljoin, urlparse

from data_connector import DataConnector

logger = logging.getLogger(__name__)

class APIConnector(DataConnector):
    """
    REST API连接器
    支持从REST API获取结构化数据
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化API连接器
        
        Args:
            config: API配置参数
                - base_url: API基础URL
                - endpoint: 数据端点
                - auth_type: 认证类型 (none, bearer, api_key, basic)
                - auth_token: 认证令牌
                - api_key: API密钥
                - username: 用户名 (basic认证)
                - password: 密码 (basic认证)
                - headers: 额外的请求头
                - timeout: 请求超时时间
                - rate_limit: 速率限制 (requests per second)
                - pagination_type: 分页类型 (offset, cursor, page)
                - page_size: 每页大小
                - timestamp_field: 时间戳字段名（用于增量同步）
        """
        super().__init__(config)
        self.session = None
        self.last_request_time = 0
        
    def connect(self) -> bool:
        """
        连接到API
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 初始化session
            self.session = requests.Session()
            
            # 设置认证
            api_key = self.config.get('api_key')
            if api_key:
                self.session.headers.update({
                    'Authorization': f'Bearer {api_key}'
                })
            
            # 设置自定义headers
            custom_headers = self.config.get('headers', {})
            self.session.headers.update(custom_headers)
            
            # 设置默认headers
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # 测试连接是否有效
            if self.test_connection():
                self.is_connected = True
                logger.info(f"Successfully connected to API: {self.config['base_url']}")
                return True
            else:
                self.is_connected = False
                logger.error(f"API connection test failed for: {self.config['base_url']}")
                return False
            
        except Exception as e:
            logger.error(f"API connector initialization failed: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        断开API连接（关闭session）
        
        Returns:
            bool: 断开是否成功
        """
        try:
            if self.session:
                self.session.close()
                self.session = None
            self.is_connected = False
            logger.info("API connector session closed")
            return True
        except Exception as e:
            logger.error(f"Error closing API connector session: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            bool: 连接是否正常
        """
        try:
            if not self.session:
                return False
            
            # 构建测试URL
            base_url = self.config['base_url']
            endpoint = self.config.get('endpoint', '')
            test_url = urljoin(base_url, endpoint)
            
            # 发送HEAD请求测试连接
            timeout = self.config.get('timeout', 30)
            response = self.session.head(test_url, timeout=timeout)
            
            # 如果HEAD不支持，尝试GET请求（限制1条记录）
            if response.status_code == 405:  # Method Not Allowed
                params = {'limit': 1} if self.config.get('pagination_type') == 'offset' else {}
                response = self.session.get(test_url, params=params, timeout=timeout)
            
            success = response.status_code < 400
            if success:
                logger.info(f"API connection test successful: {response.status_code}")
            else:
                logger.warning(f"API connection test failed: {response.status_code}")
            
            return success
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during API connection test: {str(e)}")
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """
        获取API数据结构信息
        
        Returns:
            Dict[str, Any]: API数据结构信息
        """
        try:
            if not self.is_connected:
                raise Exception("API not connected")
            
            # 获取一条样本数据来分析结构
            sample_data = list(self.fetch_data(limit=1))
            
            if not sample_data:
                return {"error": "No data available to analyze schema"}
            
            sample_record = sample_data[0]
            
            schema_info = {
                "endpoint": self.config.get('endpoint', ''),
                "base_url": self.config['base_url'],
                "fields": [
                    {
                        "name": field_name,
                        "type": type(field_value).__name__,
                        "sample_value": str(field_value)[:100] if field_value is not None else None
                    }
                    for field_name, field_value in sample_record.items()
                ],
                "total_fields": len(sample_record),
                "pagination_type": self.config.get('pagination_type', 'offset'),
                "page_size": self.config.get('page_size', 100)
            }
            
            logger.info(f"Retrieved API schema with {len(sample_record)} fields")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting API schema: {str(e)}")
            return {"error": str(e)}
    
    def fetch_data(self, 
                   query: Optional[str] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        获取API数据
        
        Args:
            query: 查询参数（JSON字符串或查询字符串）
            limit: 限制返回记录数
            offset: 偏移量
            
        Returns:
            Iterator[Dict[str, Any]]: 数据记录迭代器
        """
        try:
            if not self.is_connected:
                raise Exception("API not connected")
            
            base_url = self.config['base_url']
            endpoint = self.config.get('endpoint', '')
            url = urljoin(base_url, endpoint)
            
            # 构建查询参数
            params = {}
            
            # 处理自定义查询参数
            if query:
                try:
                    # 尝试解析为JSON
                    query_params = json.loads(query)
                    params.update(query_params)
                except json.JSONDecodeError:
                    # 如果不是JSON，作为查询字符串处理
                    from urllib.parse import parse_qs
                    query_params = parse_qs(query)
                    for k, v in query_params.items():
                        params[k] = v[0] if len(v) == 1 else v
            
            # 处理分页
            pagination_type = self.config.get('pagination_type', 'offset')
            page_size = self.config.get('page_size', 100)
            
            if limit:
                page_size = min(page_size, limit)
            
            total_fetched = 0
            current_offset = offset or 0
            
            while True:
                # 应用速率限制
                self._apply_rate_limit()
                
                # 设置分页参数
                if pagination_type == 'offset':
                    params.update({
                        'limit': page_size,
                        'offset': current_offset
                    })
                elif pagination_type == 'page':
                    page_num = (current_offset // page_size) + 1
                    params.update({
                        'page': page_num,
                        'per_page': page_size
                    })
                
                # 发送请求
                timeout = self.config.get('timeout', 30)
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # 处理不同的响应格式
                records = self._extract_records(data)
                
                if not records:
                    break
                
                for record in records:
                    if limit and total_fetched >= limit:
                        return
                    
                    yield record
                    total_fetched += 1
                
                # 检查是否还有更多数据
                if len(records) < page_size:
                    break
                
                current_offset += len(records)
                
                logger.debug(f"Fetched {total_fetched} records so far")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during API data fetch: {str(e)}")
            raise
    
    def fetch_incremental_data(self, 
                              last_sync_time: datetime,
                              query: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        获取增量数据
        
        Args:
            last_sync_time: 上次同步时间
            query: 额外的查询条件
            
        Returns:
            Iterator[Dict[str, Any]]: 增量数据记录迭代器
        """
        try:
            timestamp_field = self.config.get('timestamp_field', 'updated_at')
            
            # 构建增量查询参数
            incremental_params = {
                timestamp_field: last_sync_time.isoformat()
            }
            
            # 合并额外的查询条件
            if query:
                try:
                    query_params = json.loads(query)
                    incremental_params.update(query_params)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid query JSON, ignoring: {query}")
            
            incremental_query = json.dumps(incremental_params)
            
            logger.info(f"Fetching incremental data since: {last_sync_time}")
            
            yield from self.fetch_data(query=incremental_query)
            
        except Exception as e:
            logger.error(f"Incremental data fetch failed: {str(e)}")
            raise
    
    def get_total_count(self, query: Optional[str] = None) -> int:
        """
        获取数据总数
        
        Args:
            query: 查询条件
            
        Returns:
            int: 数据总数
        """
        try:
            if not self.is_connected:
                raise Exception("API not connected")
            
            # 尝试使用专门的计数端点
            count_endpoint = self.config.get('count_endpoint')
            if count_endpoint:
                base_url = self.config['base_url']
                count_url = urljoin(base_url, count_endpoint)
                
                params = {}
                if query:
                    try:
                        query_params = json.loads(query)
                        params.update(query_params)
                    except json.JSONDecodeError:
                        pass
                
                self._apply_rate_limit()
                timeout = self.config.get('timeout', 30)
                response = self.session.get(count_url, params=params, timeout=timeout)
                response.raise_for_status()
                
                count_data = response.json()
                return count_data.get('count', count_data.get('total', 0))
            
            # 如果没有专门的计数端点，通过获取第一页数据来估算
            first_page = list(self.fetch_data(query=query, limit=1))
            if not first_page:
                return 0
            
            # 这里只能返回一个估算值，实际实现中可能需要遍历所有数据
            logger.warning("No count endpoint available, returning estimated count")
            return 1000  # 返回一个估算值
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Count request failed: {str(e)}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during count: {str(e)}")
            return 0
    
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            List[str]: 必需的配置字段列表
        """
        return ['base_url']
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证API连接器配置
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 配置是否有效
        """
        # 检查必需字段
        if 'base_url' not in config:
            return False
        
        # 验证URL格式
        base_url = config.get('base_url', '')
        if not base_url or not base_url.startswith(('http://', 'https://')):
            return False
        
        # 验证认证配置（如果提供）
        auth_type = config.get('auth_type')
        if auth_type:
            if auth_type == 'bearer' and not config.get('auth_token'):
                return False
            elif auth_type == 'api_key' and not config.get('api_key'):
                return False
            elif auth_type == 'basic' and (not config.get('username') or not config.get('password')):
                return False
        
        # 验证超时设置
        timeout = config.get('timeout')
        if timeout is not None:
            try:
                timeout_val = float(timeout)
                if timeout_val <= 0:
                    return False
            except (ValueError, TypeError):
                return False
        
        # 验证速率限制
        rate_limit = config.get('rate_limit')
        if rate_limit is not None:
            try:
                rate_val = float(rate_limit)
                if rate_val <= 0:
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _apply_rate_limit(self):
        """
        应用速率限制
        """
        rate_limit = self.config.get('rate_limit')
        if rate_limit:
            time_since_last = time.time() - self.last_request_time
            min_interval = 1.0 / rate_limit
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _extract_records(self, data: Any) -> List[Dict[str, Any]]:
        """
        从API响应中提取记录
        
        Args:
            data: API响应数据
            
        Returns:
            List[Dict[str, Any]]: 提取的记录列表
        """
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 尝试常见的数据字段名
            for field in ['data', 'results', 'items', 'records', 'users']:
                if field in data and isinstance(data[field], list):
                    return data[field]
            
            # 如果没有找到列表字段，将整个字典作为单条记录
            return [data]
        else:
            logger.warning(f"Unexpected data format: {type(data)}")
            return []
    
    def make_request(self, 
                    method: str, 
                    endpoint: str,
                    params: Optional[Dict] = None,
                    data: Optional[Dict] = None,
                    max_retries: int = 3) -> Dict[str, Any]:
        """
        发送API请求（带重试机制）
        
        Args:
            method: HTTP方法
            endpoint: API端点
            params: 查询参数
            data: 请求体数据
            max_retries: 最大重试次数
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        try:
            if not self.is_connected:
                raise Exception("API not connected")
            
            base_url = self.config['base_url']
            url = urljoin(base_url, endpoint)
            
            for attempt in range(max_retries + 1):
                try:
                    self._apply_rate_limit()
                    timeout = self.config.get('timeout', 30)
                    
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=data,
                        timeout=timeout
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"{method} request to {endpoint} successful")
                        return response.json()
                    elif response.status_code >= 500 and attempt < max_retries:
                        # 服务器错误，重试
                        logger.warning(f"Server error {response.status_code}, retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        response.raise_for_status()
                        
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        logger.warning(f"Request failed, retrying... (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def make_custom_request(self, 
                           endpoint: str, 
                           method: str = 'GET',
                           params: Optional[Dict] = None,
                           data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        发送自定义API请求
        
        Args:
            endpoint: API端点
            method: HTTP方法
            params: 查询参数
            data: 请求体数据
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        try:
            if not self.is_connected:
                raise Exception("API not connected")
            
            base_url = self.config['base_url']
            url = urljoin(base_url, endpoint)
            
            self._apply_rate_limit()
            timeout = self.config.get('timeout', 30)
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            
            logger.info(f"Custom {method} request to {endpoint} successful")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Custom API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during custom request: {str(e)}")
            raise