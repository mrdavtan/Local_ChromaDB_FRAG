from dataset import load_data
from vectordb import create_collection, add_data_to_collection, query_database
from semantic_cache import SemanticCache
from llm_module import LLMModule  # Import the LLMModule class
import chromadb

def main():
    # Load the dataset
    max_rows = 15000
    data = load_data(max_rows)

    # Create ChromaDB collection
    chroma_client = chromadb.Client()
    collection_name = "news_collection"
    collection = create_collection(chroma_client, collection_name)

    # Add data to the collection
    document_column = "Answer"
    topic_column = "qtype"
    add_data_to_collection(collection, data, document_column, topic_column, max_rows)

    # Initialize semantic cache
    cache = SemanticCache()

    # Load the LLM using the LLMModule class
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    llm = LLMModule(model_name=model_name)
    llm.load_model()
    llm.load_pipelines()

    print("Welcome to the chat! Type 'quit' to exit.")

    while True:
        user_input = input("User: ")

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Retrieve results from the semantic cache or ChromaDB
        results = cache.ask(user_input, lambda q: query_database(collection, q, 1))

        # Create the prompt
        prompt_template = f"Relevant context: {results}\n\n The user's question: {user_input}"

        # Generate the response using the LLMModule
        response = llm.generate_text(prompt_template)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
