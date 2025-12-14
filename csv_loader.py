from langchain_community.document_loaders import CSVLoader
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'assets', 'abc.csv')

    loader = CSVLoader(file_path=csv_file_path)
    docs = loader.load()

    print(docs[0].page_content)

except Exception as e:
    print(e)