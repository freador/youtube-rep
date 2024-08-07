import ollama

prompt = f"""Qual é o beneficio de usar você?"""
response = ollama.generate(model="llama3", prompt=prompt)  # Using llama2:13b as a proxy for llama3
resp_gen = response['response'].strip()
print(resp_gen)