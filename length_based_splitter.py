from langchain_text_splitters import CharacterTextSplitter

large_text = """
LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and more. LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.
We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use LangGraph, our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.
LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.
"""

text_splitter = CharacterTextSplitter(
    separator="\n",        # Try splitting on new lines first
    chunk_size=100,        # Max size of each chunk in characters
    chunk_overlap=20,      # Number of characters to overlap between chunks
    length_function=len,   # The function used to measure length (defaults to len)
    is_separator_regex=False,
)

result = text_splitter.split_text(large_text)

for i, chunk in enumerate(result):
    print(chunk, '\n')
