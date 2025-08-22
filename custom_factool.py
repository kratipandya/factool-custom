import asyncio
import json
import sys
import os
import openai

# Set OpenRouter API key properly
OPENROUTER_API_KEY = 'sk-or-v1-...'

# Configure OpenRouter settings
os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENROUTER_API_KEY
openai.default_headers = {
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "Factool App",
}

print(f"Using API key: {OPENROUTER_API_KEY[:10]}...")  # Show first 10 chars for verification

async def main():
    # Import and set the global configuration first - BEFORE importing any factool modules
    from factool.utils.openai_wrapper import set_openrouter_config
    set_openrouter_config("https://openrouter.ai/api/v1", OPENROUTER_API_KEY)
    
    # Use the local-only pipeline
    from factool.knowledge_qa.pipeline_local import knowledge_qa_pipeline
    
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python custom_factool.py <pdf_file_path> <claims_json_file> <output_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    claims_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Load claims from JSON file
    with open(claims_file, 'r', encoding='utf-8') as f:
        claims = json.load(f)
    
    print(f"Loaded {len(claims)} claims from {claims_file}")
    print(f"Using PDF source: {pdf_file}")
    
    # Initialize the pipeline with API configuration
    pipeline = knowledge_qa_pipeline(
        foundation_model="deepseek/deepseek-chat",
        snippet_cnt=5,
        data_link=pdf_file
    )
    
    # Run fact-checking
    print("Starting fact-checking process...")
    results = await pipeline.run_with_pre_extracted_claims(claims, pdf_file)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    factual_count = sum(1 for result in results if result and result.get('factuality', False))
    print(f"Factual claims: {factual_count}/{len(claims)}")

if __name__ == "__main__":
    asyncio.run(main())
