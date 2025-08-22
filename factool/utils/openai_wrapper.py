from __future__ import annotations

import os
import yaml
import openai
import ast
import pdb
import asyncio
from typing import Any, List
import os
import pathlib
import openai
import re


# OpenRouter configuration - these will be set from the main script
OPENROUTER_API_BASE = None
OPENROUTER_API_KEY = None
DEEPSEEK_MODEL = "deepseek/deepseek-chat"

def set_openrouter_config(api_base, api_key):
    global OPENROUTER_API_BASE, OPENROUTER_API_KEY
    OPENROUTER_API_BASE = api_base
    OPENROUTER_API_KEY = api_key

class OpenAIChat():
    def __init__(
            self,
            model_name=DEEPSEEK_MODEL,
            max_tokens=2500,
            temperature=0,
            top_p=1,
            request_timeout=120,
    ):
        # Use global configuration or environment variables
        self.api_base = OPENROUTER_API_BASE or os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is not set. Please set OPENROUTER_API_KEY environment variable.")
        
        # Configure for OpenRouter
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        
        # Set default headers for OpenRouter
        openai.default_headers = {
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Factool",
        }

        self.config = {
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'request_timeout': request_timeout,
        }

    

    def extract_list_from_string(self, input_string):
        start_index = input_string.find('[')  
        end_index = input_string.rfind(']') 

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index:end_index + 1]
        else:
            return None
        
    def extract_dict_from_string(self, input_string):
        start_index = input_string.find('{')
        end_index = input_string.rfind('}')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index:end_index + 1]
        else:
            return None
    
    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        except:
            return None

    async def dispatch_openai_requests(
        self,
        messages_list,
    ) -> list[str]:
        """Dispatches requests to OpenRouter API asynchronously."""
        async def _request_with_retry(messages, retry=3):
            for _ in range(retry):
                try:
                    response = await openai.ChatCompletion.acreate(
                        model=self.config['model_name'],
                        messages=messages,
                        max_tokens=self.config['max_tokens'],
                        temperature=self.config['temperature'],
                        top_p=self.config['top_p'],
                        request_timeout=self.config['request_timeout'],
                    )
                    return response
                except openai.error.RateLimitError:
                    print('Rate limit error, waiting for 40 second...')
                    await asyncio.sleep(40)
                except openai.error.APIError:
                    print('API error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.Timeout:
                    print('Timeout error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.ServiceUnavailableError:
                    print('Service unavailable error, waiting for 3 second...')
                    await asyncio.sleep(3)
                except openai.error.APIConnectionError:
                    print('API Connection error, waiting for 3 second...')
                    await asyncio.sleep(3)
                except Exception as e:
                    print(f'Unexpected error: {e}, waiting for 3 second...')
                    await asyncio.sleep(3)

            return None

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)
    
    async def async_run(self, messages_list, expected_type):
        retry = 1
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            print(f'{retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]
            
            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
            )

            preds = [self._type_check(self._boolean_fix(prediction['choices'][0]['message']['content']), expected_type) if prediction is not None else None for prediction in predictions]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])
            
            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]
            
            retry -= 1
        
        return responses

def set_openrouter_config(api_base, api_key):
    global OPENROUTER_API_BASE, OPENROUTER_API_KEY
    OPENROUTER_API_BASE = api_base
    OPENROUTER_API_KEY = api_key

class OpenAIEmbed():
    def __init__(self):
        # For embeddings, we'll use OpenAI since OpenRouter doesn't provide embeddings
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OpenAI API key not found. Embeddings may not work.")
    
    async def create_embedding(self, text, retry=3):
        for _ in range(retry):
            try:
                response = await openai.Embedding.acreate(input=text, model="text-embedding-ada-002")
                return response
            except openai.error.RateLimitError:
                print('Rate limit error, waiting for 1 second...')
                await asyncio.sleep(1)
            except openai.error.APIError:
                print('API error, waiting for 1 second...')
                await asyncio.sleep(1)
            except openai.error.Timeout:
                print('Timeout error, waiting for 1 second...')
                await asyncio.sleep(1)
            except Exception as e:
                print(f'Embedding error: {e}, waiting for 1 second...')
                await asyncio.sleep(1)
        return None

    async def process_batch(self, batch, retry=3):
        tasks = [self.create_embedding(text, retry=retry) for text in batch]
        return await asyncio.gather(*tasks)