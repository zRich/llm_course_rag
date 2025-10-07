import re
from typing import List, Optional, Tuple
import logging

from .chunker import DocumentChunker, DocumentChunk, ChunkingConfig

logger = logging.getLogger(__name__)

class SentenceChunker(DocumentChunker):
    """基于句子的文档分块器
    
    按照句子边界进行文档分块，保持句子的完整性
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        super().__init__(config)
        
        # 句子分割的正则表达式模式
        self.sentence_patterns = {
            'zh': r'[。！？；\n]+',  # 中文句子结束符
            'en': r'[.!?;\n]+',     # 英文句子结束符
            'auto': r'[。！？；.!?;\n]+'  # 自动检测模式
        }
        
        # 缩写词列表（避免误分割）
        self.abbreviations = {
            'en': ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Co.', 'Corp.',
                   'vs.', 'etc.', 'i.e.', 'e.g.', 'cf.', 'al.', 'Jr.', 'Sr.'],
            'zh': ['先生', '女士', '博士', '教授', '公司', '有限公司']
        }
    
    def get_chunker_type(self) -> str:
        """获取分块器类型"""
        return "sentence"
    
    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """将文本按句子分割成块
        
        Args:
            text: 要分割的文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
        """
        try:
            # 检测语言
            language = self._detect_text_language(text)
            
            # 分割句子
            sentences = self._split_sentences(text, language)
            
            # 组合句子成块
            chunks = self._combine_sentences_to_chunks(sentences, source_file)
            
            self.logger.info(f"句子分块完成: {len(sentences)} 个句子 -> {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"句子分块失败: {e}")
            raise
    
    def _detect_text_language(self, text: str) -> str:
        """检测文本主要语言
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码 ('zh', 'en', 'auto')
        """
        if self.config.language != "auto":
            return self.config.language
        
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return 'zh'
        elif english_chars > chinese_chars:
            return 'en'
        else:
            return 'auto'
    
    def _split_sentences(self, text: str, language: str) -> List[Tuple[str, int, int]]:
        """分割句子
        
        Args:
            text: 文本内容
            language: 语言代码
            
        Returns:
            List[Tuple[str, int, int]]: 句子列表，每个元素包含(句子内容, 开始位置, 结束位置)
        """
        sentences = []
        
        # 获取句子分割模式
        pattern = self.sentence_patterns.get(language, self.sentence_patterns['auto'])
        
        # 预处理：保护缩写词
        protected_text = self._protect_abbreviations(text, language)
        
        # 使用正则表达式分割
        parts = re.split(f'({pattern})', protected_text)
        
        current_pos = 0
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if not part:
                continue
            
            # 如果是分隔符
            if re.match(pattern, part):
                if current_sentence.strip():
                    # 恢复缩写词
                    sentence_content = self._restore_abbreviations(current_sentence + part)
                    sentence_content = sentence_content.strip()
                    
                    if sentence_content:
                        start_pos = current_pos - len(current_sentence)
                        end_pos = current_pos + len(part)
                        sentences.append((sentence_content, start_pos, end_pos))
                
                current_sentence = ""
            else:
                current_sentence += part
            
            current_pos += len(part)
        
        # 处理最后一个句子
        if current_sentence.strip():
            sentence_content = self._restore_abbreviations(current_sentence).strip()
            if sentence_content:
                start_pos = current_pos - len(current_sentence)
                sentences.append((sentence_content, start_pos, current_pos))
        
        return sentences
    
    def _protect_abbreviations(self, text: str, language: str) -> str:
        """保护缩写词不被分割
        
        Args:
            text: 原始文本
            language: 语言代码
            
        Returns:
            str: 保护后的文本
        """
        protected_text = text
        abbreviations = self.abbreviations.get(language, [])
        
        for abbr in abbreviations:
            # 将缩写词中的句号替换为特殊标记
            protected_text = protected_text.replace(abbr, abbr.replace('.', '<<DOT>>'))
        
        return protected_text
    
    def _restore_abbreviations(self, text: str) -> str:
        """恢复缩写词
        
        Args:
            text: 保护后的文本
            
        Returns:
            str: 恢复后的文本
        """
        return text.replace('<<DOT>>', '.')
    
    def _combine_sentences_to_chunks(self, sentences: List[Tuple[str, int, int]], 
                                   source_file: str) -> List[DocumentChunk]:
        """将句子组合成块
        
        Args:
            sentences: 句子列表
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_size = 0
        chunk_start_pos = sentences[0][1] if sentences else 0
        
        for i, (sentence, start_pos, end_pos) in enumerate(sentences):
            sentence_size = len(sentence)
            
            # 检查是否需要开始新块
            if (current_chunk_size + sentence_size > self.config.chunk_size and 
                current_chunk_sentences and 
                current_chunk_size >= self.config.min_chunk_size):
                
                # 创建当前块
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, chunk_start_pos, source_file, len(chunks)
                )
                chunks.append(chunk)
                
                # 处理重叠
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.config.chunk_overlap
                )
                
                # 开始新块
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_size = sum(len(s) for s, _, _ in current_chunk_sentences)
                chunk_start_pos = overlap_sentences[0][1] if overlap_sentences else start_pos
            else:
                # 添加到当前块
                current_chunk_sentences.append((sentence, start_pos, end_pos))
                current_chunk_size += sentence_size
        
        # 处理最后一个块
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, chunk_start_pos, source_file, len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_sentences(self, sentences: List[Tuple[str, int, int]], 
                                   start_pos: int, source_file: str, 
                                   chunk_index: int) -> DocumentChunk:
        """从句子列表创建文档块
        
        Args:
            sentences: 句子列表
            start_pos: 块开始位置
            source_file: 源文件路径
            chunk_index: 块索引
            
        Returns:
            DocumentChunk: 文档块
        """
        if not sentences:
            return self._create_chunk("", start_pos, start_pos, source_file, chunk_index)
        
        # 合并句子内容
        content_parts = []
        for sentence, _, _ in sentences:
            content_parts.append(sentence)
        
        content = ' '.join(content_parts)
        end_pos = sentences[-1][2]
        
        return self._create_chunk(content, start_pos, end_pos, source_file, chunk_index)
    
    def _get_overlap_sentences(self, sentences: List[Tuple[str, int, int]], 
                             overlap_size: int) -> List[Tuple[str, int, int]]:
        """获取重叠的句子
        
        Args:
            sentences: 句子列表
            overlap_size: 重叠大小（字符数）
            
        Returns:
            List[Tuple[str, int, int]]: 重叠的句子列表
        """
        if not sentences or overlap_size <= 0:
            return []
        
        overlap_sentences = []
        current_overlap_size = 0
        
        # 从后往前选择句子
        for sentence, start_pos, end_pos in reversed(sentences):
            sentence_size = len(sentence)
            
            if current_overlap_size + sentence_size <= overlap_size:
                overlap_sentences.insert(0, (sentence, start_pos, end_pos))
                current_overlap_size += sentence_size
            else:
                break
        
        return overlap_sentences
    
    def split_by_nltk(self, text: str, language: str = 'english') -> List[str]:
        """使用NLTK进行句子分割（可选方法）
        
        Args:
            text: 文本内容
            language: 语言（'english', 'chinese'等）
            
        Returns:
            List[str]: 句子列表
        """
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # 下载必要的数据（如果需要）
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # 分割句子
            sentences = sent_tokenize(text, language=language)
            return [s.strip() for s in sentences if s.strip()]
            
        except ImportError:
            self.logger.warning("NLTK未安装，使用正则表达式分割")
            return self._regex_sentence_split(text)
        except Exception as e:
            self.logger.error(f"NLTK句子分割失败: {e}")
            return self._regex_sentence_split(text)
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """使用正则表达式分割句子（备用方法）
        
        Args:
            text: 文本内容
            
        Returns:
            List[str]: 句子列表
        """
        # 简单的正则分割
        sentences = re.split(r'[.!?;。！？；]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_sentence_statistics(self, text: str) -> dict:
        """获取句子统计信息
        
        Args:
            text: 文本内容
            
        Returns:
            dict: 统计信息
        """
        try:
            language = self._detect_text_language(text)
            sentences = self._split_sentences(text, language)
            
            if not sentences:
                return {
                    'total_sentences': 0,
                    'avg_sentence_length': 0,
                    'min_sentence_length': 0,
                    'max_sentence_length': 0,
                    'language': language
                }
            
            sentence_lengths = [len(sentence) for sentence, _, _ in sentences]
            
            return {
                'total_sentences': len(sentences),
                'avg_sentence_length': sum(sentence_lengths) / len(sentence_lengths),
                'min_sentence_length': min(sentence_lengths),
                'max_sentence_length': max(sentence_lengths),
                'language': language,
                'sentences_preview': [sentence[:50] + '...' if len(sentence) > 50 else sentence 
                                    for sentence, _, _ in sentences[:5]]
            }
            
        except Exception as e:
            self.logger.error(f"获取句子统计失败: {e}")
            return {'error': str(e)}