#############################         IMPORTS        ###########################
import os
# from google.colab import userdata

from typing import Iterator
from dataclasses import dataclass
from enum import Enum
import time

import gradio as gr
from plantweb.render import render_file

from openai import OpenAI
import anthropic
# import google.generativeai
# import ollama

import duckdb as ddb
import sqlite3



#########################         API KEYS SETUP        #########################
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
openai = OpenAI(api_key=openai_api_key)

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")
claude = anthropic.Anthropic(api_key=anthropic_api_key)

# google_api_key = os.getenv('GEMINI_API_KEY')
# if google_api_key:
#     print(f"Google API Key exists and begins {google_api_key[:8]}")
# else:
#     print("Google API Key not set")
# google.generativeai.configure()

ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')



##########################         SQLITE SETUP        ##########################
# Connect to or create a SQLite database file
DB_PATH = "nl2sm-app-prototype/nl2sm_oltp.sqlite3"

with sqlite3.connect(DB_PATH) as db_conn:
    db_curs = db_conn.cursor()
    db_curs.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id                              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type                      TEXT,
            llm_provider                    TEXT,
            llm_model                       TEXT,
            temperature                     REAL,
            srs_id                          TEXT,
            original_srs_text               TEXT,
            prompt_distill_srs              TEXT,
            distilled_srs_text              TEXT,
            plantuml_script_orig_srs        TEXT,
            plantuml_script_dist_srs        TEXT,
            uml_image_data_orig_srs         BLOB,
            uml_image_data_dist_srs         BLOB,
            extracted_uml_descr_orig_srs    TEXT,
            extracted_uml_descr_dist_srs    TEXT,
            similarity_descr_orig_srs       TEXT,
            similarity_score_orig_srs       REAL,
            similarity_descr_dist_srs       TEXT,
            similarity_score_dist_srs       REAL,
            created_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

# Store prompts for each interaction



##########################   CONSTANTS AND CLASSES     ##########################
# Constants
class LLMProvider(Enum):
    OPENAI = "OpenAI"
    META = "Meta AI (via Ollama)"
    ANTHROPIC = "Anthropic"
    # GOOGLE = "Google"

class OpenAIModels(Enum):
    GPT4_MINI = "GPT 4o mini"
    GPT4 = "GPT 4o"
    # O1 = "o1"
    O1_MINI = "o1 mini"
    # O3_MINI = "o3 mini"

class MetaAIModels(Enum):
    LLAMA32 = "Llama 3.2 3B"

class AnthropicModels(Enum):
    CLAUDE35_HAIKU = "Claude 3.5 Haiku"
    CLAUDE35_SONNET = "Claude 3.5 Sonnet"

@dataclass
class ConversationState:
    original_srs: str = ""
    distilled_srs: str = ""
    generated_plantuml: str = ""
    chat_history: list = None

    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []

# Initialize state
state = ConversationState()

##########################         HELPER FUNCTIONS        ##########################
def get_available_models(provider: LLMProvider) -> list:
    """Return available models based on selected provider."""
    if provider == LLMProvider.OPENAI.value:
        return [model.value for model in OpenAIModels]
    elif provider == LLMProvider.META.value:
        return [model.value for model in MetaAIModels]
    elif provider == LLMProvider.ANTHROPIC.value:
        return [model.value for model in AnthropicModels]
    return []

def process_openai_request(prompt: str, model: OpenAIModels, temperature: float, stream: bool=True) -> Iterator[str]:
    """Process request using OpenAI's API."""
    # Map UI model names to actual OpenAI model identifiers
    model_mapping = {
        OpenAIModels.GPT4.value: "gpt-4o",
        OpenAIModels.GPT4_MINI.value: "gpt-4o-mini",
        # OpenAIModels.O1.value: "o1",
        OpenAIModels.O1_MINI.value: "o1-mini",
        # OpenAIModels.O3_MINI.value: "o3-mini"
    }
    try:
        if stream:
            # Streaming approach
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
            # Non-stream approach
            response = openai.chat.completions.create(
                model=model_mapping[model],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False
            )
            # The entire completion is in response.choices[0].message.content
            text = response.choices[0].message.content
            yield text
    except Exception as e:
        yield f"Error processing OpenAI request: {str(e)}"

def process_meta_request(prompt: str, model: MetaAIModels, temperature: float) -> Iterator[str]:
    """Process request using Meta AI's local Llama model running through Ollama and called through OpenAI API interface."""
    try:
        # Assuming Llama is running locally with OpenAI API compatibility
        response = ollama_via_openai.chat.completions.create(
            model="llama3.2",  # Local model identifier
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True
        )
        full_response = ""
        for chunk in response:
            if chunk and chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                yield full_response
    except Exception as e:
        yield f"Error processing Meta AI (via Ollama) request: {str(e)}"

def process_anthropic_request(prompt: str, model: AnthropicModels, temperature: float, stream: bool=True) -> Iterator[str]:
    """Process request using Anthropic's API."""
    model_mapping = {
        AnthropicModels.CLAUDE35_HAIKU.value: "claude-3-5-haiku-latest",
        AnthropicModels.CLAUDE35_SONNET.value: "claude-3-5-sonnet-latest"
    }
    try:
        if stream:
            with claude.messages.stream(
                model=model_mapping[model],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
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

def generate_uml_prompt(srs_text: str) -> str:
    """Generate prompt for UML class diagram generation."""
    return f"""Please analyze the following Software Requirements Specification and generate a PlantUML script for a Class Diagram that accurately represents the described system:

{srs_text}

Generate only the PlantUML script, starting with @startuml and ending with @enduml. Include appropriate classes, attributes, methods, and relationships."""

def check_similarity_prompt(srs_text: str, plantuml_script: str) -> str:
    """Generate prompt for similarity checking."""
    return f"""Task 1: First, analyze the following PlantUML script and extract a textual description of the system it represents:

{plantuml_script}

Task 2: Compare the extracted description with the original SRS text below:

{srs_text}

Provide:
1. A similarity score between 0 and 100
2. A list of specific discrepancies between the original SRS and the UML representation
3. Any missing or additional elements in the UML diagram that weren't specified in the SRS

Format your response as:
Similarity Score: X/100

Discrepancies:
- [List specific differences]

Missing Elements:
- [List elements from SRS missing in UML]

Additional Elements:
- [List elements in UML not specified in SRS]"""

def process_llm_request(provider: LLMProvider, model: str, prompt: str, temperature: float, stream: bool = True):
    """Route request to appropriate provider."""
    if provider == LLMProvider.OPENAI.value:
        yield from process_openai_request(prompt, model, temperature, stream=stream)
    elif provider == LLMProvider.META.value:
        yield from process_meta_request(prompt, model, temperature)
    elif provider == LLMProvider.ANTHROPIC.value:
        yield from process_anthropic_request(prompt, model, temperature, stream=stream)

def extract_plantuml_code(raw_text: str) -> str:
    """
    Extracts a clean PlantUML script from raw_text:
    - Strips triple quotes or triple backticks if wrapping the text
    - Finds the substring from @startuml to @enduml
    """
    s = raw_text.strip()

    # 1) Strip triple single quotes or backticks if they wrap the entire text
    if (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    elif (s.startswith('```') and s.endswith('```')):
        s = s[3:-3].strip()

    # 2) Find the @startuml / @enduml block
    start_idx = s.find("@startuml")
    end_idx   = s.rfind("@enduml")  # rfind() ensures we catch the last one if multiple occur

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        # If we can't find a valid block, just return an empty string (or original s if you prefer)
        return ""

    # 3) Return substring from @startuml to @enduml
    return s[start_idx:end_idx + len("@enduml")]

def generate_plantuml_diagram(script: str) -> str:
    """
    Generates a UML diagram using PlantWeb and returns a file path to the rendered image.
    Gradio will display this file path in an Image component set to `type="filepath"`.
    """
    try:
        # Ensure @startuml/@enduml is present
        if not script.strip().startswith("@startuml"):
            script = f"@startuml\n{script}\n@enduml"

        state.generated_plantuml = script

        infile = 'mygraph.dot'
        with open(infile, 'wb') as fd:
          fd.write(script.encode('utf-8'))

        outfile = render_file(
          infile,
          renderopts={
              'engine': 'plantuml',
              'format': 'png'
          },
          cacheopts={
              'use_cache': False
        }
        )

        return outfile

    except Exception as e:
        print(f"Error generating diagram with PlantWeb: {e}")
        return ""

def handle_generate_uml(srs_type: str, plantuml_script: str, db_store: bool = True):
    """Generate UML from script and return a PNG image."""
    if srs_type == "original":
        if not plantuml_script.strip():
            return None
        outfile = generate_plantuml_diagram(plantuml_script)

        image_data = None
        if outfile:
            with open(outfile, 'rb') as f:
                image_data = f.read()

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        plantuml_script_orig_srs, 
                        uml_image_data_orig_srs
                    ) 
                    VALUES (?, ?, ?)
                    """, 
                    [
                        "GENERATE_UML_ORIG",
                        plantuml_script,
                        image_data  # BLOB data
                    ]
                )
        # ----------------------------
    
    if srs_type == "distilled":
        if not plantuml_script.strip():
            return None
        outfile = generate_plantuml_diagram(plantuml_script)

        image_data = None
        if outfile:
            with open(outfile, 'rb') as f:
                image_data = f.read()

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        plantuml_script_dist_srs, 
                        uml_image_data_dist_srs
                    ) 
                    VALUES (?, ?, ?)
                    """, 
                    [
                        "GENERATE_UML_DIST",
                        plantuml_script,
                        image_data  # BLOB data
                    ]
                )
        # ----------------------------
    return outfile

def handle_send(srs_type: str, srs_text: str, provider: LLMProvider, model: str, temperature: float, chatbot: list, db_store: bool = True):
    # srs_id: str,
    if not srs_text.strip():
        return "", chatbot, ""

    if srs_type == "original":
        # Store original SRS
        state.original_srs = srs_text

        # 1. Add SRS text to chatbot
        chatbot.append((srs_text, ""))

        # 2. Make your prompt and get the LLM response
        prompt = generate_uml_prompt(srs_text)
        response_generator = process_llm_request(provider, model, prompt, temperature)

        full_response = ""
        for partial_chunk in response_generator:
            full_response = partial_chunk
            # Update chatbot with partial chunk
            chatbot[-1] = (srs_text, partial_chunk)
            yield "", chatbot, full_response

        # 3. Now that we have the final LLM response, extract @startuml..@enduml
        clean_uml = extract_plantuml_code(full_response)

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        original_srs_text, 
                        plantuml_script_orig_srs,
                        llm_provider, 
                        llm_model, 
                        temperature
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "SEND_LLM_ORIG_SRS",
                        srs_text,
                        clean_uml,
                        provider,
                        model,
                        temperature
                    ]
                )
        # ----------------------------

    if srs_type == "distilled":
        # Store original SRS
        state.distilled_srs = srs_text

        # 1. Add SRS text to chatbot
        chatbot.append((srs_text, ""))

        # 2. Make your prompt and get the LLM response
        prompt = generate_uml_prompt(srs_text)
        response_generator = process_llm_request(provider, model, prompt, temperature)

        full_response = ""
        for partial_chunk in response_generator:
            full_response = partial_chunk
            # Update chatbot with partial chunk
            chatbot[-1] = (srs_text, partial_chunk)
            yield "", chatbot, full_response

        # 3. Now that we have the final LLM response, extract @startuml..@enduml
        clean_uml = extract_plantuml_code(full_response)

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        distilled_srs_text, 
                        plantuml_script_dist_srs,
                        llm_provider, 
                        llm_model, 
                        temperature
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "SEND_LLM_DIST_SRS",
                        srs_text,
                        clean_uml,
                        provider,
                        model,
                        temperature
                    ]
                )
        # ----------------------------

    # 4. Return the cleaned UML script in the plantuml_script TextArea
    yield "", chatbot, clean_uml

def handle_check_similarity(srs_type: str, provider: LLMProvider, model: str, temperature: float, db_store: bool = True) -> str:
    """Handle Check Similarity button click."""

    if srs_type == "original":
        if not state.original_srs or not state.generated_plantuml:
            return "Please ensure both SRS text and PlantUML script are available."

        prompt = check_similarity_prompt(state.original_srs, state.generated_plantuml)

        # Process request and return response
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
        similarity_text = "".join(list(response_generator))

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        original_srs_text, 
                        plantuml_script_orig_srs, 
                        similarity_descr_orig_srs,
                        llm_provider,
                        llm_model,
                        temperature
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "CHECK_SIMILARITY_ORIG",
                        state.original_srs,
                        state.generated_plantuml,
                        similarity_text,
                        provider,
                        model,
                        temperature
                    ]
                )
        # ----------------------------

    if srs_type == "distilled":
        if not state.distilled_srs or not state.generated_plantuml:
            return "Please ensure both Distilled SRS text and PlantUML script are available."

        prompt = check_similarity_prompt(state.distilled_srs, state.generated_plantuml)

        # Process request and return response
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
        similarity_text = "".join(list(response_generator))

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        distilled_srs_text, 
                        plantuml_script_dist_srs, 
                        similarity_descr_dist_srs,
                        llm_provider,
                        llm_model,
                        temperature
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "CHECK_SIMILARITY_DIST",
                        state.distilled_srs,
                        state.generated_plantuml,
                        similarity_text,
                        provider,
                        model,
                        temperature
                    ]
                )
        # ----------------------------

    return similarity_text

def handle_send_non_stream(srs_type: str, srs_text: str, provider: LLMProvider, model: str, temperature: float, chatbot: list, db_store: bool = True):
    """
    Non-stream version of handle_send. 
    Returns final UML script plus updated chatbot.
    """
    if not srs_text.strip():
        return "", chatbot, ""


    if srs_type == "original":
        # Store original SRS
        state.original_srs = srs_text

        # 1. Add SRS text to chatbot
        chatbot.append((srs_text, ""))

        # 2. Make your prompt and get the LLM response (no streaming).
        prompt = generate_uml_prompt(srs_text)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)

        # Combine the response into one string.
        full_response = "".join(list(response_generator))
        chatbot[-1] = (srs_text, full_response)

        # 3. Extract the final UML script
        clean_uml = extract_plantuml_code(full_response)

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        original_srs_text, 
                        plantuml_script_orig_srs,
                        llm_provider, 
                        llm_model, 
                        temperature
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "ALL_IN_ONE_ORIG_SRS",
                        srs_text,
                        clean_uml,
                        provider,
                        model,
                        temperature
                    ]
                )
        # -----------------------------------------------------------

    if srs_type == "distilled":
        # Store original SRS
        state.distilled_srs = srs_text

        # 1. Add SRS text to chatbot
        chatbot.append((srs_text, ""))

        # 2. Make your prompt and get the LLM response (no streaming).
        prompt = generate_uml_prompt(srs_text)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)

        # Combine the response into one string.
        full_response = "".join(list(response_generator))
        chatbot[-1] = (srs_text, full_response)

        # 3. Extract the final UML script
        clean_uml = extract_plantuml_code(full_response)

        # ----------------------------
        # SQLITE DATA STORAGE
        # ----------------------------
        if db_store:
            with sqlite3.connect(DB_PATH) as db_conn:
                db_curs = db_conn.cursor()
                db_curs.execute("""
                    INSERT INTO usage_logs 
                    (
                        event_type, 
                        distilled_srs_text, 
                        plantuml_script_dist_srs,
                        llm_provider, 
                        llm_model, 
                        temperature
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "ALL_IN_ONE_DIST_SRS",
                        srs_text,
                        clean_uml,
                        provider,
                        model,
                        temperature
                    ]
                )
        # -----------------------------------------------------------

    # Return SRS text, updated chatbot, and the final UML script
    return srs_text, chatbot, clean_uml

def handle_all_in_one(srs_type: str, srs_text: str, provider: LLMProvider, model, temperature, chatbot):
    """
    Calls:
    1) handle_send_non_stream   -> gets UML script
    2) handle_generate_uml      -> converts UML script to UML diagram
    3) handle_check_similarity  -> compares original SRS & textual description from the UML diagram
    Returns multiple outputs for Gradio.
    """

    if srs_type == "original":
        # Step 1: Get final UML script
        new_srs_text, new_chatbot, uml_script = handle_send_non_stream(srs_type, srs_text, provider, model, temperature, chatbot, db_store=False)

        # Step 2: Generate UML diagram (returns a file path or None)
        outfile = handle_generate_uml(srs_type, uml_script, db_store=False)

        # read the image data for storing
        image_data = None
        if outfile:
            with open(outfile, 'rb') as f:
                image_data = f.read()

        # Step 3: Check similarity
        similarity_text = handle_check_similarity(srs_type, provider, model, temperature, db_store=False)

        with sqlite3.connect(DB_PATH) as db_conn:
            # add a timer to check the time takes from the start to the end of the DB storage
    
            db_curs = db_conn.cursor()
            db_curs.execute("""
                INSERT INTO usage_logs 
                (
                    event_type, 
                    original_srs_text, 
                    plantuml_script_orig_srs,
                    uml_image_data_orig_srs,
                    similarity_descr_orig_srs,
                    llm_provider,
                    llm_model,
                    temperature
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                "ALL_IN_ONE_ORIG_SRS",
                new_srs_text,
                uml_script,
                image_data,
                similarity_text,
                provider,
                model,
                temperature
                ]
            )
    
    if srs_type == "distilled":
        # Step 1: Get final UML script
        new_srs_text, new_chatbot, uml_script = handle_send_non_stream(srs_type, srs_text, provider, model, temperature, chatbot, db_store=False)

        # Step 2: Generate UML diagram (returns a file path or None)
        outfile = handle_generate_uml(srs_type, uml_script, db_store=False)

        # read the image data for storing
        image_data = None
        if outfile:
            with open(outfile, 'rb') as f:
                image_data = f.read()

        # Step 3: Check similarity
        similarity_text = handle_check_similarity(srs_type, provider, model, temperature, db_store=False)

        with sqlite3.connect(DB_PATH) as db_conn:
            # add a timer to check the time takes from the start to the end of the DB storage
    
            db_curs = db_conn.cursor()
            db_curs.execute("""
                INSERT INTO usage_logs 
                (
                    event_type, 
                    distilled_srs_text, 
                    plantuml_script_dist_srs,
                    uml_image_data_dist_srs,
                    similarity_descr_dist_srs,
                    llm_provider,
                    llm_model,
                    temperature
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                "ALL_IN_ONE_DIST_SRS",
                new_srs_text,
                uml_script,
                image_data,
                similarity_text,
                provider,
                model,
                temperature
                ]
            )

    # Return everything that your UI expects
    return new_srs_text, new_chatbot, uml_script, outfile, similarity_text

def TODO_handle_all_in_one_original_and_distilled(srs_text, provider, model, temperature, chatbot):
    """
    Calls:
    1) handle_distill_non_stream    ->  gets SRSd                                   ===============> new function
    2.1) handle_send_non_stream     ->  gets UMLo script                            ===============> implement argument to differentiate between O and D, important for DB storage
    2.2) handle_send_non_stream     ->  gets UMLd script                            ===============> implement argument to differentiate between O and D, important for DB storage
    3.1) handle_generate_uml        ->  converts UMLo script to UMLo Diagram        ===============> implement argument to differentiate between O and D, important for DB storage
    3.2) handle_generate_uml        ->  converts UMLd script to UMLd Diagram        ===============> implement argument to differentiate between O and D, important for DB storage
    4.1) handle_check_similarity    ->  compares SRSo & UMLo extracted description  ===============> implement argument to differentiate between O and D, important for DB storage
    4.2) handle_check_similarity    ->  compares SRSd & UMLd extracted description  ===============> implement argument to differentiate between O and D, important for DB storage
    Returns multiple outputs for Gradio.
    """

    return srs_text_original, srs_text_distilled, plantuml_script_original, plantuml_script_distilled, uml_image_original, uml_image_distilled, similarity_result_original, similarity_result_distilled, chatbot



def create_ui():
    """Create and configure the Gradio interface."""
    with gr.Blocks() as app:
        with gr.Row():

            with gr.Column(scale=2):
                # Sidebar
                # Provider and Model selection and automatic updates
                provider = gr.Dropdown(
                    choices=[p.value for p in LLMProvider],
                    label="LLM Provider",
                    value=LLMProvider.OPENAI.value
                )
                model = gr.Dropdown(
                    choices=get_available_models(LLMProvider.OPENAI.value),
                    label="LLM Model",
                    value=OpenAIModels.GPT4_MINI.value
                )
                def update_model(provider):
                    if provider == LLMProvider.OPENAI.value:
                        return gr.update(choices=get_available_models(provider), value=OpenAIModels.GPT4_MINI.value)
                    elif provider == LLMProvider.META.value:
                        return gr.update(choices=get_available_models(provider), value=MetaAIModels.LLAMA32.value)
                    elif provider == LLMProvider.ANTHROPIC.value:
                        return gr.update(choices=get_available_models(provider), value=AnthropicModels.CLAUDE35_HAIKU.value)
                    return gr.update(choices=get_available_models(provider))
                provider.change(
                    fn=update_model,
                    inputs=provider,
                    outputs=model
                )
                provider.change(
                fn=lambda p: gr.Dropdown(choices=get_available_models(p)),
                inputs=provider,
                outputs=model
                )
                
                # Temperature slider and automatic updates based on model
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    label="Temperature"
                )
                srs_type = gr.Dropdown(
                    choices=["original", "distilled"],
                    label="SRS Type"
                )

                def update_temperature(model):
                    if model == OpenAIModels.O1_MINI.value:
                        return gr.update(value=1.0, minimum=1.0, maximum=1.0, step=0.0)
                    else:
                        return gr.update(value=0.5, minimum=0.0, maximum=1.0, step=0.1)

                model.change(
                    fn=update_temperature,
                    inputs=model,
                    outputs=temperature
                )

                # srs_id = gr.Textbox(label="SRS ID") # <--- add SRS ID once SQLite is integrated
                srs_placeholder = """
                Suppose the financial system of a country is composed of banks. 
                * Each bank has branches, which may be spread throughout the country, but there may not be more than three branches of the same bank in the same city. 
                * The structure of a bank's branches is hierarchical, so that a branch may have subordinate branches (of the same bank), either in the same city or in other cities. 
                * The customers of a branch can be either the individuals or companies that have an account at the branch. Each customer may have one or more accounts in any of the branches of a bank. Each account can have only one account holder, the customer who opens the account. 
                * Each branch owns an account, in which it stores its assets. 
                * Each account has a balance, which must be positive, unless it is a "credit" account. 
                * Credit accounts allow their balance to be negative, with a limit that is established when the account is opened. 
                * A customer may request a change in this limit from the branch where the account is held, and the branch must request authorization from the branch directly responsible for it (except the head office, at the root of the hierarchy, which can make decisions directly). Changes in the credit limit will be authorized as long as the new limit is lower than the previous one, or if it is only 10% higher than the one the account already had, and the current balance exceeds the new limit (e.g., if the credit limit is 1,000 Euros and you request to increase it to 1,005 Euros with a balance of 1,100 euros on the account, the branch will authorize the change).
                * Customer can perform transactions with their accounts (request balance, deposit or withdraw money, and transfer money to another account). hey can also open accounts at any branch. When opening an account, the initial balance is 0. If the account is a credit account, the initial limit is 10 euros. 
                * The accounts have a maintenance fee of 1% per year. This means that, once a year (on January 1), each branch deducts 1% from the current balance of all its accounts. If the balance is 0 and the account is not a credit account, no money is deducted. In case of credit accounts, if the resulting balance is negative the branch will deduct 1% of the account's credit limit instead. The deducted money from the becomes the property of the branch and is stored in your account.
                * All customers can transfer money from any of the accounts the own, to any other account -- no matter who the owner is, or the bank the destination account belongs to. Transfers between accounts of different banks have a 2% commission, i.e., if you transfer 1,000 euros, then 980 euros are deposited in the destination account. That 2% of the money becomes the property of the origin and destination branches (1% for each). Transfers between accounts of the same bank are free of charge. 
                * Companies with a balance of more than 1 million euros in one of their accounts are considered VIP by that bank and have a number of advantages. Firstly, in a transfer between banks, the part corresponding to the account of origin or destination that is of that bank is not taxed with the corresponding 1%. Secondly, these accounts do not pay an annual maintenance fee. Only companies, and not individuals, can be considered VIP.
                * Finally, accounts whose holders are bank branches pay neither annual fee nor transfer commission in any bank. 
                """
                srs_text = gr.TextArea(label="SRS Textual Description", lines=24, max_lines=24, value=srs_placeholder)
                all_in_one_btn = gr.Button("All in One")
                send_btn = gr.Button("Send to LLM")
                generate_btn = gr.Button("Generate UML")
                check_btn = gr.Button("Check Similarity")

            with gr.Column(scale=4):
                # Main Area
                with gr.Row():
                    # First Row with whole width
                    with gr.Column(scale=4):
                        # Upper left
                        chatbot = gr.Chatbot(height=400)

                with gr.Row():
                    with gr.Column():
                        # Upper right
                        plantuml_script = gr.TextArea(label="PlantUML Script", lines=12, max_lines=12, show_copy_button=True)
                    with gr.Column():
                        uml_image = gr.Image(type="pil", label="UML Diagram", height=306, show_download_button=True, show_fullscreen_button=True)
               
                with gr.Row(scale=4):
                        # Lower right
                        
                        similarity_result = gr.TextArea(label="Similarity Analysis", lines=12)

        # Event handlers

        all_in_one_btn.click(
            fn=handle_all_in_one,
            inputs=[srs_type, srs_text, provider, model, temperature, chatbot],
            # Return 5 outputs: updated SRS text, chatbot, UML script, UML image, and similarity text
            outputs=[srs_text, chatbot, plantuml_script, uml_image, similarity_result]
        )

        send_btn.click(
            fn=handle_send,
            inputs=[srs_type, srs_text, provider, model, temperature, chatbot], # <--- add srs_id
            outputs=[srs_text, chatbot, plantuml_script]  # <--- now we also update plantuml_script
        )

        generate_btn.click(
            fn=handle_generate_uml,
            inputs=[srs_type, plantuml_script],
            outputs=uml_image
        )

        check_btn.click(
            fn=handle_check_similarity,
            inputs=[srs_type, provider, model, temperature],
            outputs=similarity_result
        )

    return app

def create_new_ui():
    """Create and configure the Gradio interface."""
    with gr.Blocks(fill_width=True) as app:
        # LLM and Temperature selection
        with gr.Row():
            with gr.Column():
                provider = gr.Dropdown(
                    choices=[p.value for p in LLMProvider],
                    label="LLM Provider",
                    value=LLMProvider.OPENAI.value
                )
            with gr.Column():
                model = gr.Dropdown(
                    choices=get_available_models(LLMProvider.OPENAI.value),
                    label="LLM Model",
                    value=OpenAIModels.GPT4_MINI.value
                )
                def update_model(provider):
                    if provider == LLMProvider.OPENAI.value:
                        return gr.update(choices=get_available_models(provider), value=OpenAIModels.GPT4_MINI.value)
                    elif provider == LLMProvider.META.value:
                        return gr.update(choices=get_available_models(provider), value=MetaAIModels.LLAMA32.value)
                    elif provider == LLMProvider.ANTHROPIC.value:
                        return gr.update(choices=get_available_models(provider), value=AnthropicModels.CLAUDE35_HAIKU.value)
                    return gr.update(choices=get_available_models(provider))
                provider.change(
                    fn=update_model,
                    inputs=provider,
                    outputs=model
                )
                provider.change(
                fn=lambda p: gr.Dropdown(choices=get_available_models(p)),
                inputs=provider,
                outputs=model
                )
            with gr.Column():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    label="Temperature"
                )
                def update_temperature(model):
                    if model == OpenAIModels.O1_MINI.value:
                        return gr.update(value=1.0, minimum=1.0, maximum=1.0, step=0.0)
                    else:
                        return gr.update(value=0.5, minimum=0.0, maximum=1.0, step=0.1)
                model.change(
                    fn=update_temperature,
                    inputs=model,
                    outputs=temperature
                )
        
        # Original SRS, 'Send to LLM' button and Distilled SRS
        with gr.Row():
            # Original SRS and 'Send to LLM' button
            with gr.Column():
                srs_text_original = gr.TextArea(label="Original SRS Textual Description", lines=18, max_lines=18)
                all_in_one_original_and_distilled_btn = gr.Button("Send to LLM (All in One)")
                # all_in_one_btn = gr.Button("Send to LLM (All in One)")
                # srs_id = gr.Textbox(label="SRS ID") # <--- add SRS ID once DuckDB is integrated
            # Distilled SRS
            with gr.Column():
                srs_text_distilled = gr.TextArea(label="Distilled SRS", lines=21, max_lines=21)

        # PlantUML Scripts
        with gr.Row():
            # Original PlantUML Script
            with gr.Column():
                plantuml_script_original = gr.TextArea(label="PlantUML Script (SRSo)", lines=12, max_lines=12, show_copy_button=True)

            # Distilled PlantUML Script
            with gr.Column():
                plantuml_script_distilled = gr.TextArea(label="PlantUML Script (SRSd)", lines=12, max_lines=12, show_copy_button=True)

        # Original SRS, Send button and Distilled SRS
        with gr.Row():
            # Original PlantUML Diagram
            with gr.Column():
                uml_image_original = gr.Image(type="pil", label="PlantUML Diagram (SRSo)", height=306, show_download_button=True, show_fullscreen_button=True)
            # Distilled PlantUML Diagram
            with gr.Column():
                uml_image_distilled = gr.Image(type="pil", label="PlantUML Diagram (SRSd)", height=306, show_download_button=True, show_fullscreen_button=True)

        # Original and Distilled Semantic Similarity Analysis
        with gr.Row():
            # Original Semantic Similarity Analysis
            with gr.Column():
                similarity_result_original = gr.TextArea(label="Similarity Analysis (SRSo)", lines=12)
            # Distilled Semantic Similarity Analysis
            with gr.Column():
                similarity_result_distilled = gr.TextArea(label="Similarity Analysis (SRSd)", lines=12)   

        # Chatbot (to remove)
        with gr.Row():
            # First Row with whole width
            with gr.Column(scale=4):
                # Upper left
                chatbot = gr.Chatbot(height=400)


        # Event handlers

        # all_in_one_btn.click(
        #     fn=handle_all_in_one,
        #     inputs=[srs_text_original, provider, model, temperature, chatbot],
        #     # Return 5 outputs: updated SRS text, chatbot, UML script, UML image, and similarity text
        #     outputs=[srs_text_original, chatbot, plantuml_script_original, uml_image_original, similarity_result_original]
        # )

        all_in_one_original_and_distilled_btn.click(
            fn=handle_all_in_one_original_and_distilled,
            inputs=[srs_text_original, provider, model, temperature, chatbot],
            # Return 9 outputs: original and distilled SRS text, original and distilled PlantUML script, original and distilled UML image, original and distilled similarity text, and chatbot
            outputs=[srs_text_original, srs_text_distilled, plantuml_script_original, plantuml_script_distilled, uml_image_original, uml_image_distilled, similarity_result_original, similarity_result_distilled, chatbot]

            # srs_text_distilled                =========> should be the first call to the LLM    ===> new function

            # plantuml_script_distilled         =========> uses the exact same function as plantuml_script_original, need to worry about concurrency with the LLM calls
            
            # uml_image_distilled               =========> uses the exact same function as uml_image_original, need to worry about concurrency with the PlantWeb calll
            
            # similarity_result_distilled       =========> uses the exact same function as similarity_result_original, need to worry about concurrency with the LLM calls
        )


    return app



##########################          UI LAUNCH            ##########################
if __name__ == "__main__":

    # Create and launch the interface
    app = create_ui()
    # app = create_new_ui()
    app.launch()

