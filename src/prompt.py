prompt_template = """
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""