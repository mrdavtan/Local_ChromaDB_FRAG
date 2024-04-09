import chromadb

def create_collection(chroma_client, collection_name):
    if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)
    return chroma_client.create_collection(name=collection_name)

def add_data_to_collection(collection, data, document_column, topic_column, max_rows):
    collection.add(
        documents=data[document_column].tolist(),
        metadatas=[{topic_column: topic} for topic in data[topic_column].tolist()],
        ids=[f"id{x}" for x in range(max_rows)],
    )

def query_database(collection, query_text, n_results=10):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results
