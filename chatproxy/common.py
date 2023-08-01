from .services.openai_adaptor import OpenAIAdaptor
from .services.tgi_adaptor import TGIAdaptor

ADAPTORS = {"openai": OpenAIAdaptor, "tgi": TGIAdaptor}


def init(services):
    for service in services:
        if service not in ADAPTORS:
            raise RuntimeError("Unkown service.")
        ADAPTORS[service].init()
