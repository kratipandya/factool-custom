import asyncio
from factool.knowledge_qa.tool import local_search

async def test_search():
    pdf_file = "History_of_Lingusitics.pdf"
    
    # Test with some specific queries that should be in the PDF
    test_queries = [
        "linguistics",
        "grammar", 
        "language",
        "Saussure",
        "Chomsky",
        "Panini",
        "syntax",
        "semantics"
    ]
    
    search_tool = local_search(snippet_cnt=3, data_link=pdf_file)
    await search_tool.initialize()
    
    print("Testing search with common linguistics terms:")
    for query in test_queries:
        results = await search_tool.search(query)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results[:2]):  # Show first 2 results
            print(f"  {i+1}. {result['content'][:100]}...")
    
    # Test with some of your actual claims
    claim_queries = [
        "linguistics as a scientific study began",
        "ancient Indian scholar Panini",
        "Ferdinand de Saussure introduced"
    ]
    
    print("\n\nTesting search with claim-based queries:")
    for query in claim_queries:
        results = await search_tool.search(query)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test_search())