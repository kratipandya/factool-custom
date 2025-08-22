import asyncio
import json
import os
import numpy as np
import jsonlines
import pdb
import re
from factool.utils.openai_wrapper import OpenAIEmbed

class local_search():
    def __init__(self, snippet_cnt, data_link, embedding_link=None):
        self.snippet_cnt = snippet_cnt
        self.data_link = data_link
        self.embedding_link = embedding_link
        self.openai_embed = OpenAIEmbed()
        self.data = None
        self.embedding = None
        self.full_text = None
        self.sentences = None
        
    async def initialize(self):
        """Initialize the search tool asynchronously when needed"""
        print("Initializing local search with PDF...")
        self.load_data_from_pdf()
        print("Loaded data for keyword search")
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract clean text from PDF file using pdfplumber"""
        text = ""
        try:
            # Try using pdfplumber first (better text extraction)
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                print(f"PDF has {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Clean up the text - remove headers, footers, page numbers
                        cleaned_text = self.clean_page_text(page_text, page_num + 1)
                        if cleaned_text:
                            text += cleaned_text + "\n\n"
                        print(f"Page {page_num + 1}: {len(page_text)} characters")
            
            print(f"Successfully extracted {len(text)} total characters from PDF using pdfplumber")
            
        except ImportError:
            print("pdfplumber not available, falling back to PyPDF2")
            # Fallback to PyPDF2
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        cleaned_text = self.clean_page_text(page_text, page_num + 1)
                        if cleaned_text:
                            text += cleaned_text + "\n\n"
            
            print(f"Extracted {len(text)} characters using PyPDF2")
            
        except Exception as e:
            print(f"Error reading PDF: {e}")
        
        return text

    def clean_page_text(self, text, page_num):
        """Clean up page text by removing headers, footers, page numbers, etc."""
        # Remove common PDF artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove lines that look like headers/footers/page numbers
            if self.is_header_footer(line, page_num):
                continue
            
            # Remove lines with file paths or compilation info
            if any(artifact in line for artifact in ['Comp. by:', 'Stage :', 'ChapterID:', 'Date:', 'Time:', 'Filepath:', 'd:/womat-filecopy']):
                continue
            
            # Remove lines that are just punctuation or very short
            if len(line.strip()) > 3 and not line.strip().isdigit():
                cleaned_lines.append(line.strip())
        
        return ' '.join(cleaned_lines)

    def is_header_footer(self, line, page_num):
        """Check if a line looks like a header or footer"""
        line_lower = line.lower()
        
        # Common header/footer patterns
        patterns = [
            r'page\s+\d+',  # Page X
            r'\d+\s+of\s+\d+',  # X of Y
            r'chapter\s+\d+',  # Chapter X
            r'section\s+\d+',  # Section X
            r'oxford handbook',  # Book title
            r'history of linguistics',  # Book subtitle
            r'\.{20,}',  # Dotted lines
            r'\-{20,}',  # Dashed lines
            r'_{20,}',   # Underscore lines
        ]
        
        for pattern in patterns:
            if re.search(pattern, line_lower):
                return True
        
        # Very short lines or mostly punctuation
        if len(line.strip()) < 5 or sum(1 for c in line if c.isalnum()) < 3:
            return True
            
        return False

    def load_data_from_pdf(self):
        """Load and process PDF data"""
        print(f"Loading PDF from: {self.data_link}")
        self.full_text = self.extract_text_from_pdf(self.data_link)
        
        if self.full_text and len(self.full_text.strip()) > 1000:  # Reasonable amount of text
            # Split into sentences for better search
            self.sentences = re.split(r'[.!?]+', self.full_text)
            # Clean up sentences
            self.sentences = [sentence.strip() for sentence in self.sentences if len(sentence.strip()) > 20]
            print(f"Extracted {len(self.sentences)} clean sentences from PDF")
            
            # Save cleaned text for debugging
            with open('cleaned_pdf_text.txt', 'w', encoding='utf-8') as f:
                f.write(self.full_text)
            print("Saved cleaned text to cleaned_pdf_text.txt")
        else:
            print("Warning: Very little text extracted from PDF")
            print(f"Extracted text length: {len(self.full_text) if self.full_text else 0}")
            self.sentences = []

    async def search(self, query):
        """Improved keyword-based search with better matching"""
        if not self.sentences:
            await self.initialize()
            
        if not self.sentences:
            return [{"content": "No data available for search", "source": self.data_link}]
        
        # Preprocess query for better matching
        query_words = query.lower().split()
        query_words = [word for word in query_words if len(word) > 2]  # Keep shorter words
        
        results = []
        
        # First pass: exact phrase match (case insensitive)
        query_lower = query.lower()
        for sentence in self.sentences:
            if query_lower in sentence.lower():
                results.append({"content": sentence, "source": self.data_link})
                if len(results) >= self.snippet_cnt:
                    return results
        
        # Second pass: all important words match
        if len(results) < self.snippet_cnt and query_words:
            for sentence in self.sentences:
                sentence_lower = sentence.lower()
                if all(word in sentence_lower for word in query_words):
                    results.append({"content": sentence, "source": self.data_link})
                    if len(results) >= self.snippet_cnt:
                        break
        
        # Third pass: any word match with context
        if len(results) < self.snippet_cnt and query_words:
            for sentence in self.sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words):
                    # Add some context around the match
                    results.append({"content": sentence, "source": self.data_link})
                    if len(results) >= self.snippet_cnt:
                        break
        
        return results if results else [{"content": "No matching content found in the PDF", "source": self.data_link}]

    async def run(self, queries):
        """Process multiple queries"""
        if not self.sentences:
            await self.initialize()
            
        flattened_queries = []
        for sublist in queries:
            if sublist is None:
                sublist = ['None', 'None']
            for item in sublist:
                flattened_queries.append(item)
        
        print(f"Searching for {len(flattened_queries)} queries...")
        
        # Search for each query using keyword search
        snippets = await asyncio.gather(*[self.search(query) for query in flattened_queries])
        
        # Debug: show what was found for each query
        for i, (query, result) in enumerate(zip(flattened_queries, snippets)):
            print(f"Query {i+1}: '{query}' -> Found {len(result)} results")
            if result and result[0]['content'] != "No matching content found in the PDF":
                print(f"  First result: {result[0]['content'][:100]}...")
        
        snippets_split = [snippets[i] + snippets[i+1] for i in range(0, len(snippets), 2)]
        return snippets_split