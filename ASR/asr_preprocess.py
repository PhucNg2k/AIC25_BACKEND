from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=[". ", "! ", "? ", " "],  # Try sentences first, then words
    chunk_size=50,                       # Optimal for news segments (30-60 seconds)
    chunk_overlap=15,                     # Good overlap for context
    length_function=lambda x: len(x.split())
)