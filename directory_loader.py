from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os
# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the assets folder
assets_path = os.path.join(script_dir, 'assets')

loader = DirectoryLoader(
    path=assets_path,
    glob='*.pdf',
    loader_cls=PyPDFLoader
)


# Added error handling in case no files are found
try:
    docs = loader.load()
    if docs:
        print(f"Loaded {len(docs)} documents.")
        print(docs[0].page_content[:500])
    else:
        print("No PDF documents found in the specified directory.")
except FileNotFoundError:
    print(f"Error: The directory was not found at {assets_path}")
except Exception as e:
    print(f"An error occurred during loading: {e}")