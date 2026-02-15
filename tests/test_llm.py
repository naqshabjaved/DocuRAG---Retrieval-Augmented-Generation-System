from app.services.llm import LLMService

if __name__ == "__main__":
    llm = LLMService()

    prompt = """
Use the following context to answer the question.
If the answer is not contained in the context, say "I don't know."

Context:
The capital of France is Paris.

Question:
What is the capital of France?

Answer:
"""

    response = llm.generate(prompt)

    print("LLM Response:\n")
    print(response)

