import importlib
from . import common


class Session:
    def __init__(
        self,
        service,
        model="gpt-3.5-turbo",
        system_prompt=None,
        temperature=0.3,
        max_tokens=400,
    ):
        if service not in common.ADAPTORS:
            raise RuntimeError(f"Unkown service name")
        if (adaptor := common.ADAPTORS[service]) and not adaptor.initialized:
            raise RuntimeError(
                f"ChatProxy is not initialized properly. call chatproxy.init(services=['service', ...]) first"
            )
        self.service_adaptor = adaptor
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not system_prompt:
            self.system_prompt = "You are a helpful assistant"
        else:
            self.system_prompt = system_prompt
        self.messages = []

    def vector(self, query):
        return self.service_adaptor.get_vector(query)

    def clear(self):
        self.messages = []

    def send(self, prompt=None):
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        response = self.service_adaptor.send(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.messages.append(response["message"])
        return response

    def generate(self, prompt=None):
        if not self.messages:
            self.messages = self.system_prompt

        if prompt:
            self.messages += prompt

        response = self.service_adaptor.send(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.messages += response["message"]
        return response
