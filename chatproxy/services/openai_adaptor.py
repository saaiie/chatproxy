import openai
from os import environ as env


class OpenAIAdaptor:
    _initialized = False

    @classmethod
    def init(cls):
        openai.api_key = env["CHATPROXY_OPENAI_API_KEY"]
        cls._initialized = True

    @classmethod
    @property
    def initialized(cls):
        return cls._initialized

    @classmethod
    def default_system_prompt(cls):
        return "You are a helpful assistant"

    @classmethod
    def get_vector(cls, query):
        embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")
        vector = embedding["data"][0]["embedding"]
        return vector

    @classmethod
    def send(cls, model, messages, temperature, max_tokens):
        # print("REQ:", messages)
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # print("RES:", resp)
        message = dict(resp["choices"][0]["message"])
        return {"response": resp, "message": message, "content": message["content"]}
