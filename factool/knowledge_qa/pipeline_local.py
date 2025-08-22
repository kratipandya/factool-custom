import json
import yaml
import os
import time
import math
import pdb
from typing import List, Dict

from factool.knowledge_qa.tool import local_search
from factool.utils.base.pipeline import pipeline

class knowledge_qa_pipeline(pipeline):
    def __init__(self, foundation_model, snippet_cnt=5, data_link=None, embedding_link=None):
        """Constructor for local search only - no search_type parameter needed"""
        super().__init__('knowledge_qa', foundation_model)
        
        # Always use local search
        self.tool = local_search(snippet_cnt=snippet_cnt, data_link=data_link, embedding_link=embedding_link)
        
        # Load prompts with proper encoding or use fallbacks
        try:
            with open(os.path.join(self.prompts_path, "claim_extraction.yaml"), 'r', encoding='utf-8') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
            self.claim_prompt = data['knowledge_qa']
        except:
            self.claim_prompt = {
                'system': 'You are a helpful assistant that extracts factual claims from text.',
                'user': 'Extract factual claims from the following text:\n\n{input}'
            }

        try:
            with open(os.path.join(self.prompts_path, 'query_generation.yaml'), 'r', encoding='utf-8') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
            self.query_prompt = data['knowledge_qa']
        except:
            self.query_prompt = {
                'system': 'You are a helpful assistant that generates search queries to verify factual claims.',
                'user': 'Generate a search query to verify the following claim: {input}'
            }

        try:
            with open(os.path.join(self.prompts_path, 'agreement_verification.yaml'), 'r', encoding='utf-8') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
            self.verification_prompt = data['knowledge_qa']
        except:
            self.verification_prompt = {
                'system': 'You are a fact-checker. Determine if the claim is supported by the evidence.',
                'user': 'Claim: {claim}\n\nEvidence: {evidence}\n\nIs the claim supported by the evidence? Respond with JSON: {{"factuality": true/false, "reasoning": "explanation"}}'
            }
    
    async def _claim_extraction(self, responses):
        messages_list = [
            [
                {"role": "system", "content": self.claim_prompt['system']},
                {"role": "user", "content": self.claim_prompt['user'].format(input=response)},
            ]
            for response in responses
        ]
        return await self.chat.async_run(messages_list, List)
    
    async def _query_generation(self, claims):
        if claims == None:
            return ['None']
        messages_list = [
            [
                {"role": "system", "content": self.query_prompt['system']},
                {"role": "user", "content": self.query_prompt['user'].format(input=claim['claim'] if 'claim' in claim else '')},
            ]
            for claim in claims
        ]
        return await self.chat.async_run(messages_list, List)

    async def _verification(self, claims, evidences):
        messages_list = [
            [
                {"role": "system", "content": self.verification_prompt['system']},
                {"role": "user", "content": self.verification_prompt['user'].format(claim=claim['claim'], evidence=str(evidence))},
            ]
            for claim, evidence in zip(claims, evidences)
        ]
        return await self.chat.async_run(messages_list, dict)
    
    async def run_with_tool_live_without_claim_extraction(self, claims):
        queries = await self._query_generation(claims)
        search_outputs = await self.tool.run(queries)
        
        # Flatten the search outputs structure
        evidences = []
        for output in search_outputs:
            if isinstance(output, list):
                evidences.append([item['content'] for item in output])
            else:
                evidences.append([output['content']])
        
        final_response = await self._verification(claims, evidences)
        for i in range(len(final_response)):
            if final_response[i] != None:
                final_response[i]['queries'] = queries[i]
                final_response[i]['evidences'] = evidences[i]

        return final_response
    
    async def run_with_pre_extracted_claims(self, claims, source_file_path):
        """Run with pre-extracted claims and a specific source file"""
        # Reinitialize local search with the provided source file
        self.tool = local_search(snippet_cnt=5, data_link=source_file_path)
        
        # Run the pipeline without claim extraction
        return await self.run_with_tool_live_without_claim_extraction(claims)