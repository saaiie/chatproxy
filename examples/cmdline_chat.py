# Very simple command line chat client
import os
import openai
import chatproxy


openai.api_key = os.environ["OPENAI_API_KEY"]

print("To exit the chat, type 'exit'\n")

session = chatproxy.Session()
while (prompt := input("you : ")) != "exit":
    message = session.send(prompt)
    print(f"\n{message['role']} : {message['content']}\n")
