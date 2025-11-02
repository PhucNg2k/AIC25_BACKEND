from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


def process_string(text: str) -> str:
    return text.strip()

# Configure word-based splitter
splitter = CharacterTextSplitter(
    separator=" ",                      # split on spaces (word-based)
    chunk_size=2,                       # 8 words per chunk
    chunk_overlap=0,                    # 1 word overlap
    length_function=lambda x: len(x.split())  # count words instead of chars
)
