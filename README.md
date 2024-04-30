ChromaDB RAG Chat Application
=============================

![0ba62820-eac8-4de9-85bd-91394ca7d69f](https://github.com/mrdavtan/Local_ChromaDB_FRAG/assets/21132073/be173a80-8fb9-472e-859c-23a42103e3c4)

This project is a modularized version of the ChromaDB Retrieval-Augmented Generation (RAG) chat application based on the Hugging Face cookbook tutorial: [Semantic Cache with Chroma Vector Database](https://huggingface.co/learn/cookbook/semantic_cache_chroma_vector_database). The application combines semantic caching and language model-based text generation to provide relevant and contextual responses to user queries.

Description
-----------

The ChromaDB RAG Chat Application utilizes the following components:

1.  **Dataset**: The application loads a dataset (`keivalya/MedQuad-MedicalQnADataset`) using the `datasets` library and prepares it for further processing.
2.  **Vector Database**: The loaded dataset is stored in a ChromaDB collection, which serves as a vector database for efficient retrieval of relevant documents based on user queries.
3.  **Semantic Cache**: The application implements a semantic cache (`SemanticCache`) that stores previously asked questions, their embeddings, answers, and response texts. The cache uses the FAISS library for efficient similarity search and the Sentence Transformers library for encoding questions into embeddings.
4.  **Language Model**: The application utilizes the `mistralai/Mistral-7B-Instruct-v0.1` language model for generating responses based on the retrieved context from the vector database or the semantic cache.

Unique Features
---------------

1.  **Modularized Structure**: The application follows a modularized structure, separating different functionalities into individual files. This modular approach enhances code organization, reusability, and maintainability.
2.  **Semantic Caching**: The application employs a semantic cache that stores previous user queries, their embeddings, and corresponding responses. When a new query is asked, the cache is searched for similar questions using the FAISS library. If a similar question is found, the cached response is returned, reducing the need for database retrieval and language model inference.
3.  **Vector Database**: The application uses ChromaDB, a vector database, to store and retrieve relevant documents based on user queries. ChromaDB enables efficient similarity search, allowing the application to find the most relevant context for generating responses.
4.  **Language Model Integration**: The application integrates the `mistralai/Mistral-7B-Instruct-v0.1` language model using the `LLMModule` class. The language model is used to generate contextual responses based on the retrieved context from the vector database or the semantic cache.

Usage
-----

1.  Install the required dependencies:
    -   `datasets`
    -   `chromadb`
    -   `faiss`
    -   `sentence_transformers`
    -   `transformers`
2.  Run the `main.py` script to start the chat application.
3.  Enter user queries in the chat interface. The application will retrieve relevant context from the semantic cache or the ChromaDB vector database and generate responses using the language model.
4.  To exit the chat, type 'quit'.

File Structure
--------------

-   `dataset.py`: Contains functions for loading and preparing the dataset.
-   `vectordb.py`: Defines functions for creating and interacting with the ChromaDB vector database.
-   `semantic_cache.py`: Implements the semantic cache functionality using FAISS and Sentence Transformers.
-   `llm_module.py`: Defines the `LLMModule` class for loading and utilizing the language model.
-   `main.py`: The main script that orchestrates the chat application by combining the dataset, vector database, semantic cache, and language model.


