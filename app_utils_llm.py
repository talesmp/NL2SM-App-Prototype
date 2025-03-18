# llm_utils.py
import os
from enum import Enum
from typing import Iterator

# If you have these libs installed:
import anthropic
from openai import OpenAI
# import google.generativeai
# import ollama

# Global environment-based API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

# Prepare clients (you might want to handle the case when they are None)
openai = None
claude = None
ollama_via_openai = None

if openai_api_key:
    openai = OpenAI(api_key=openai_api_key)

if anthropic_api_key:
    claude = anthropic.Anthropic(api_key=anthropic_api_key)

# Example: If you have an Ollama server locally
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


# Enums for LLM providers and models
class LLMProvider(Enum):
    OPENAI = "OpenAI"
    # META = "Meta AI (via Ollama)"
    ANTHROPIC = "Anthropic"
    # GOOGLE = "Google" # If you want to re-enable

class OpenAIModels(Enum):
    GPT4_MINI = "GPT 4o mini"
    GPT4 = "GPT 4o"
    O1_MINI = "o1 mini"
    # O1 = "o1"
    # O3_MINI = "o3 mini"

# class MetaAIModels(Enum):
#     LLAMA32 = "Llama 3.2 3B"

class AnthropicModels(Enum):
    CLAUDE35_HAIKU = "Claude 3.5 Haiku"
    CLAUDE35_SONNET = "Claude 3.5 Sonnet"


def get_models_for_provider(provider: LLMProvider) -> list[str]:
    """Return a list of model names for a given provider Enum."""
    if provider == LLMProvider.OPENAI:
        return [m.value for m in OpenAIModels]
    # elif provider == LLMProvider.META:
    #     return [m.value for m in MetaAIModels]
    elif provider == LLMProvider.ANTHROPIC:
        return [m.value for m in AnthropicModels]
    return []


def process_openai_request(prompt: str, model: OpenAIModels, temperature: float, stream: bool=True) -> Iterator[str]:
    """
    Process request using OpenAI's API, yielding chunks as they stream in.
    """
    model_mapping = {
        OpenAIModels.GPT4.value: "gpt-4o",
        OpenAIModels.GPT4_MINI.value: "gpt-4o-mini",
        OpenAIModels.O1_MINI.value: "o1-mini",
        # ...
    }
    try:
        if not openai:
            yield f"[Error] OpenAI API key not configured."
            return

        if stream:
            response = openai.chat.completions.create(
                model=model_mapping[model],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True
            )
            full_response = ""
            for chunk in response:
                if chunk and chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    yield full_response
        else:
            response = openai.chat.completions.create(
                model=model_mapping[model],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False
            )
            text = response.choices[0].message.content
            yield text
    except Exception as e:
        yield f"Error processing OpenAI request: {str(e)}"


# def process_meta_request(prompt: str, model: MetaAIModels, temperature: float) -> Iterator[str]:
#     """
#     Process request using a local Llama-based model via Ollama (exposed with OpenAI-compatible endpoint).
#     """
#     try:
#         if not ollama_via_openai:
#             yield f"[Error] Ollama local server not configured."
#             return

#         response = ollama_via_openai.chat.completions.create(
#             model="llama3.2",  # or your local model name
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature,
#             stream=True
#         )
#         full_response = ""
#         for chunk in response:
#             if chunk and chunk.choices and chunk.choices[0].delta.content:
#                 full_response += chunk.choices[0].delta.content
#                 yield full_response
#     except Exception as e:
#         yield f"Error processing Meta AI (via Ollama) request: {str(e)}"


def process_anthropic_request(prompt: str, model: AnthropicModels, temperature: float, stream: bool=True) -> Iterator[str]:
    """
    Process request using Anthropic's Claude model.
    """
    model_mapping = {
        AnthropicModels.CLAUDE35_HAIKU.value: "claude-3-5-haiku-latest",
        AnthropicModels.CLAUDE35_SONNET.value: "claude-3-5-sonnet-latest"
    }

    try:
        if not claude:
            yield f"[Error] Anthropic API key not configured."
            return

        if stream:
            with claude.messages.stream(
                model=model_mapping[model],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            ) as stream_resp:
                full_response = ""
                for text in stream_resp.text_stream:
                    full_response += text
                    yield full_response
        else:
            response = claude.messages.create(
                model=model_mapping[model],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            )
            yield response.content[0].text
    except Exception as e:
        yield f"Error processing Anthropic request: {str(e)}"


def process_llm_request(provider: str, model: str, prompt: str, temperature: float, stream: bool = True):
    """
    Dispatch to the correct LLM function based on selected provider.
    """
    if provider == LLMProvider.OPENAI.value:
        yield from process_openai_request(prompt, model, temperature, stream=stream)
    # elif provider == LLMProvider.META.value:
    #     yield from process_meta_request(prompt, model, temperature)
    elif provider == LLMProvider.ANTHROPIC.value:
        yield from process_anthropic_request(prompt, model, temperature, stream=stream)
    else:
        yield f"[Error] Unknown provider: {provider}"
