import asyncio
import json
import sys
import os
import openai

# Set OpenRouter API key properly - MAKE SURE TO USE YOUR ACTUAL KEY
OPENROUTER_API_KEY = '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'  # ← Replace with your actual OpenRouter API key

# Configure OpenRouter settings - THIS IS CRITICAL
os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENROUTER_API_KEY
openai.default_headers = {
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "Factool App",
}

print(f"Using API key: {OPENROUTER_API_KEY[:10]}...")  # Show first 10 chars for verification

async def debug_main():
    # Import and set the global configuration first
    from factool.utils.openai_wrapper import set_openrouter_config
    set_openrouter_config("https://openrouter.ai/api/v1", OPENROUTER_API_KEY)
    
    # Use the local-only pipeline
    from factool.knowledge_qa.pipeline_local import knowledge_qa_pipeline
    
    pdf_file = sys.argv[1]
    claims_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Check if files exist with better path handling
    if not os.path.exists(pdf_file):
        print(f"PDF file not found: {pdf_file}")
        print("Looking for PDF in current directory...")
        
        # Try different possible locations
        possible_paths = [
            pdf_file,
            os.path.join(os.getcwd(), pdf_file),
            os.path.join(os.getcwd(), "factool", pdf_file),
            os.path.join(os.path.dirname(os.getcwd()), pdf_file)
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                pdf_file = path
                found = True
                print(f"Found PDF at: {pdf_file}")
                break
        
        if not found:
            print("PDF file not found in any common locations.")
            print("Please make sure the PDF file is in the same directory as this script.")
            return
    
    if not os.path.exists(claims_file):
        print(f"Claims file not found: {claims_file}")
        return
    
    # Load claims from JSON file
    with open(claims_file, 'r', encoding='utf-8') as f:
        claims = json.load(f)
    
    print(f"Loaded {len(claims)} claims")
    print(f"Using PDF: {pdf_file}")
    
    # Test with just a few claims first
    test_claims = claims[:2]  # Test with first 2 claims only
    print(f"Testing with {len(test_claims)} claims:")
    for i, claim in enumerate(test_claims):
        print(f"  {i+1}. {claim['claim'][:100]}...")
    
    # Initialize pipeline
    try:
        pipeline = knowledge_qa_pipeline(
            foundation_model="deepseek/deepseek-chat",
            snippet_cnt=3,  # Reduced for testing
            data_link=pdf_file
        )
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return
    
    # Test query generation first
    print("Testing query generation...")
    try:
        queries = await pipeline._query_generation(test_claims)
        print("Generated queries:")
        for i, query in enumerate(queries):
            if query:
                print(f"  {i+1}. {query}")
            else:
                print(f"  {i+1}. [Failed to generate query]")
    except Exception as e:
        print(f"Query generation failed: {e}")
        return
    
    # Test search
    print("Testing search...")
    try:
        search_results = await pipeline.tool.run(queries)
        print("Search results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Found {len(result)} evidence snippets")
            for j, snippet in enumerate(result[:1]):  # Show just first snippet
                print(f"     {j+1}. {snippet['content'][:100]}...")
    except Exception as e:
        print(f"Search failed: {e}")
        return
    
    # Test verification
    print("Testing verification...")
    try:
        evidences = []
        for result in search_results:
            if isinstance(result, list):
                evidences.append([item['content'] for item in result])
            else:
                evidences.append([result['content']])
        
        verification_results = await pipeline._verification(test_claims, evidences)
        print("Verification results:")
        for i, result in enumerate(verification_results):
            if result:
                print(f"  {i+1}. Factuality: {result.get('factuality', 'Unknown')}")
                print(f"     Reasoning: {result.get('reasoning', 'No reasoning')[:100]}...")
            else:
                print(f"  {i+1}. Verification failed")
    except Exception as e:
        print(f"Verification failed: {e}")
        return
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(debug_main())