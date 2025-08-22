import asyncio
import PyPDF2
import json
import os
import numpy as np
import jsonlines
import pdb
from factool.utils.openai_wrapper import OpenAIEmbed

class local_search():
    def __init__(self, snippet_cnt, data_link, embedding_link=None):
        self.snippet_cnt = snippet_cnt
        self.data_link = data_link
        self.embedding_link = embedding_link
        self.openai_embed = OpenAIEmbed()
        self.data = None
        self.embedding = None
        asyncio.run(self.init_async())
        
    
    async def init_async(self):
        print("Initializing local search with PDF...")
        self.load_data_from_pdf()
        if self.embedding_link is None:
            await self.calculate_embedding()
        else:
            self.load_embedding_by_link()
        print("Loaded data and embedding from PDF")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks for better search"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def load_data_from_pdf(self):
        """Load and process PDF data"""
        print(f"Loading PDF from: {self.data_link}")
        raw_text = self.extract_text_from_pdf(self.data_link)
        self.data = self.chunk_text(raw_text)
        print(f"Extracted {len(self.data)} chunks from PDF")

    def add_suffix_to_json_filename(self, filename):
        base_name, extension = os.path.splitext(filename)
        return base_name + '_embed.jsonl'

    def load_embedding_by_link(self):
        self.embedding = []
        try:
            with jsonlines.open(self.embedding_link) as reader:
                for obj in reader:
                    self.embedding.append(obj)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embedding = None
    
    def save_embeddings(self):
        try:
            with jsonlines.open(self.add_suffix_to_json_filename(self.data_link), mode='w') as writer:
                for emb in self.embedding:
                    writer.write(emb)
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    async def calculate_embedding(self):
        try:
            result = await self.openai_embed.process_batch(self.data, retry=3)
            self.embedding = [emb["data"][0]["embedding"] if emb else None for emb in result]
            # Filter out None values
            self.embedding = [emb for emb in self.embedding if emb is not None]
            self.save_embeddings()
        except Exception as e:
            print(f"Error calculating embeddings: {e}")
            self.embedding = None

    async def search(self, query):
        if not self.embedding or not self.data:
            return [{"content": "No data available for search", "source": "local"}]
        
        try:
            result = await self.openai_embed.create_embedding(query)
            if not result:
                return [{"content": "Embedding failed", "source": "local"}]
                
            query_embed = result["data"][0]["embedding"]
            dot_product = np.dot(self.embedding, query_embed)
            sorted_indices = np.argsort(dot_product)[::-1]
            top_k_indices = sorted_indices[:self.snippet_cnt]
            return [{"content": self.data[i], "source": self.data_link} for i in top_k_indices]
        except Exception as e:
            print(f"Search error: {e}")
            return [{"content": f"Search error: {str(e)}", "source": "local"}]

    
    async def run(self, queries):
        flattened_queries = []
        for sublist in queries:
            if sublist is None:
                sublist = ['None', 'None']
            for item in sublist:
                flattened_queries.append(item)
        
        snippets = await asyncio.gather(*[self.search(query) for query in flattened_queries])
        snippets_split = [snippets[i] + snippets[i+1] for i in range(0, len(snippets), 2)]
        return snippets_split