import PyPDF2
import re
import sys

def analyze_pdf(pdf_path):
    print(f"Analyzing PDF: {pdf_path}")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num in range(min(5, len(pdf_reader.pages))):  # First 5 pages
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n\n"
                    print(f"\n--- Page {page_num + 1} ---")
                    print(page_text[:500] + "..." if len(page_text) > 500 else page_text)
            
        # Analyze content
        sentences = re.split(r'[.!?]+', text)
        words = re.findall(r'\b\w+\b', text.lower())
        
        print(f"\n=== PDF Analysis ===")
        print(f"Total characters: {len(text)}")
        print(f"Total sentences: {len(sentences)}")
        print(f"Total words: {len(words)}")
        print(f"Unique words: {len(set(words))}")
        
        # Show most common words
        from collections import Counter
        word_counts = Counter(words)
        print(f"\nTop 20 words: {word_counts.most_common(20)}")
        
        # Check for linguistics terms
        linguistics_terms = ['linguistics', 'grammar', 'language', 'syntax', 'semantics', 
                           'phonetics', 'phonology', 'morphology', 'pragmatics', 'saussure',
                           'chomsky', 'panini', 'structuralism', 'generative', 'comparative']
        
        print(f"\nLinguistics terms found:")
        for term in linguistics_terms:
            if term in words:
                print(f"  {term}: {words.count(term)} occurrences")
                
    except Exception as e:
        print(f"Error analyzing PDF: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_pdf.py <pdf_file_path>")
    else:
        analyze_pdf(sys.argv[1])