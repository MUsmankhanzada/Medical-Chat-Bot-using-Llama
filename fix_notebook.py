# Add this code to your notebook to fix the error:

# Cell 20: Initialize the LLM (add this after downloading the model)
llm = ctransformers.CTransformers(
    model=model_path,
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.1,
        'context_length': 2048,
        'gpu_layers': 0  # Set to 0 for CPU, or higher number for GPU layers
    }
)

# Cell 21: Create the QA chain (add this after setting up vector_store)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.4}
    )
)

# Enable source docs separately
qa.return_source_documents = True

# Cell 22: Test the chatbot
def ask_question(question):
    """Function to ask questions to the medical chatbot"""
    try:
        result = qa({"query": question})
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['result']}")
        if result.get('source_documents'):
            print("\nSources:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"{i}. {doc.page_content[:200]}...")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test with a sample question
test_question = "What are the symptoms of diabetes?"
ask_question(test_question) 