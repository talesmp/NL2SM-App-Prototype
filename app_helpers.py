# app_helpers.py

import os
from dataclasses import dataclass
from typing import Iterator, List
import sqlite3

import gradio as gr
from plantweb.render import render_file
import uuid

from llm_utils import (
    LLMProvider,
    OpenAIModels,
    # MetaAIModels,
    AnthropicModels,
    process_llm_request
)
from db_utils import store_usage_log, DB_PATH

###############################################################################
# Conversation State
###############################################################################
@dataclass
class ConversationState:
    original_srs: str = ""
    distilled_srs: str = ""
    generated_plantuml: str = ""
    chat_history: list = None

    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []

# We create a single shared instance.  Alternatively, you could
# pass it around as needed, or store it in a Gradio State object.
state = ConversationState()

###############################################################################
# Helper Functions for prompts, UML generation, etc.
###############################################################################
def distill_srs_text(provider: LLMProvider, model: str, temperature: float, original_srs_text: str, db_store: bool = True) -> str:
    """
    Calls the selected LLM to produce a distilled SRS from the original SRS text.
    """
    # Create the prompt
    prompt = f"""
    From now on, you will take on the role of a software requirements analyst.
    You are very proficient at extracting the information necessary to build UML Class Diagrams models from Natural Language textual requirements.

    The following content enclosed in triple quotes is a software requirement description, which we call R.

    \"\"\" 
    {original_srs_text}
    \"\"\"

    You don't need to provide any explanation for it.

    Please extract only the information necessary to build a UML Class Diagram, such as entities, relationships, constraints and rules based on the above software requirements.

    Provide me with a concise and structured version of the software requirements in order to build an UML Class Diagram.
    Do not proceed with building the UML Class Diagram yet.
        """.strip()

    # We do a non-stream call here to get the final distilled SRS text
    response_gen = process_llm_request(provider, model, prompt, temperature, stream=False)
    distilled_srs = "".join(list(response_gen))  # gather chunks into one string

    # (Optional) log it in the DB
    if db_store:
        store_usage_log(
            event_type="DISTILL_SRS",
            llm_provider=provider,
            llm_model=model,
            temperature=temperature,
            original_srs_text=original_srs_text,
            distilled_srs_text=distilled_srs
        )

    return prompt, distilled_srs


def generate_uml_prompt(srs_text: str) -> str:
    """Generate prompt for UML class diagram generation."""
    return f"""Please analyze the following Software Requirements Specification and generate a PlantUML script for a Class Diagram that accurately represents the described system:

{srs_text}

Generate only the PlantUML script, starting with @startuml and ending with @enduml. Include appropriate classes, attributes, methods, and relationships."""


def check_similarity_prompt(srs_text: str, plantuml_script: str) -> str:
    """Generate prompt for similarity checking."""
    return f"""Task 1: Analyze the following PlantUML script and extract a textual description of the system it represents:

{plantuml_script}

Task 2: Compare the extracted description with the original SRS text below:

{srs_text}

Provide:
1. A similarity score between 0 and 100
2. A list of specific discrepancies
3. Any missing or additional elements in the UML diagram that weren't specified in the SRS

Format your response as:
Similarity Score: X/100

Discrepancies:
- ...

Missing Elements:
- ...

Additional Elements:
- ...
"""


def extract_plantuml_code(raw_text: str) -> str:
    """
    Extracts a clean PlantUML script from raw_text:
    - Strips triple quotes or triple backticks if wrapping the text
    - Finds the substring from @startuml to @enduml
    """
    s = raw_text.strip()

    # 1) Strip triple quotes/backticks if they wrap the entire text
    if (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    elif (s.startswith('```') and s.endswith('```')):
        s = s[3:-3].strip()

    # 2) Find the @startuml / @enduml block
    start_idx = s.find("@startuml")
    end_idx   = s.rfind("@enduml")  # rfind() for last occurrence

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return ""

    return s[start_idx:end_idx + len("@enduml")]


def generate_plantuml_diagram(script: str) -> str:
    """
    Generates a UML diagram using PlantWeb and returns a file path to the rendered image.
    """
    try:
        if not script.strip().startswith("@startuml"):
            script = f"@startuml\n{script}\n@enduml"

        state.generated_plantuml = script

        # Create a unique file name
        unique_id = uuid.uuid4().hex
        dot_file = f"mygraph_{unique_id}.dot"
        with open(dot_file, 'wb') as fd:
            fd.write(script.encode('utf-8'))

        outfile = render_file(
            dot_file,
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


###############################################################################
# Higher-level event handlers
###############################################################################
def handle_generate_uml(srs_type: str, plantuml_script: str, db_store: bool = True):
    """Generate UML from script and return a PNG image."""
    if not plantuml_script.strip():
        return None

    outfile = generate_plantuml_diagram(plantuml_script)
    image_data = None
    if outfile:
        with open(outfile, 'rb') as f:
            image_data = f.read()

    if db_store:
        if srs_type == "original":
            store_usage_log(
                event_type="GENERATE_UML_ORIG",
                plantuml_script_orig_srs=plantuml_script,
                uml_image_data_orig_srs=image_data
            )
        if srs_type == "distilled":
            store_usage_log(
                event_type="GENERATE_UML_DIST",
                plantuml_script_dist_srs=plantuml_script,
                uml_image_data_dist_srs=image_data
            )
    return outfile


def handle_send(srs_type: str, srs_text: str, provider: LLMProvider, model: str, temperature: float, chatbot: list, db_store: bool = True):
    """
    Submit SRS to LLM, get partial streaming responses, extract final UML script,
    store to DB.  Gradio generator function (hence 'yield').
    """
    if not srs_text.strip():
        return "", chatbot, ""

    # Add SRS text to chatbot
    chatbot.append((srs_text, ""))

    if srs_type == "original":
        state.original_srs = srs_text
    else:
        state.distilled_srs = srs_text

    # Make the UML prompt
    prompt = generate_uml_prompt(srs_text)
    response_generator = process_llm_request(provider, model, prompt, temperature)

    full_response = ""
    for partial_chunk in response_generator:
        full_response = partial_chunk
        chatbot[-1] = (srs_text, partial_chunk)
        yield "", chatbot, full_response

    # Extract UML
    clean_uml = extract_plantuml_code(full_response)

    if db_store:
        if srs_type == "original":
            store_usage_log(
                event_type="SEND_LLM_ORIG_SRS",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                original_srs_text=srs_text,
                plantuml_script_orig_srs=clean_uml
            )
        else:
            store_usage_log(
                event_type="SEND_LLM_DIST_SRS",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                distilled_srs_text=srs_text,
                plantuml_script_dist_srs=clean_uml
            )

    yield "", chatbot, clean_uml


def handle_check_similarity(srs_type: str, provider: LLMProvider, model: str, temperature: float, db_store: bool = True) -> str:
    """Compare SRS text to the UML diagram content in state."""
    if srs_type == "original":
        if not state.original_srs or not state.generated_plantuml:
            return "Please provide the original SRS text and UML script first."

        prompt = check_similarity_prompt(state.original_srs, state.generated_plantuml)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
        similarity_text = "".join(list(response_generator))  # Collect all pieces

        if db_store:
            store_usage_log(
                event_type="CHECK_SIMILARITY_ORIG",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                original_srs_text=state.original_srs,
                plantuml_script_orig_srs=state.generated_plantuml,
                similarity_descr_orig_srs=similarity_text
            )

        return similarity_text

    else:
        if not state.distilled_srs or not state.generated_plantuml:
            return "Please provide the distilled SRS text and UML script first."

        prompt = check_similarity_prompt(state.distilled_srs, state.generated_plantuml)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
        similarity_text = "".join(list(response_generator))

        if db_store:
            store_usage_log(
                event_type="CHECK_SIMILARITY_DIST",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                distilled_srs_text=state.distilled_srs,
                plantuml_script_dist_srs=state.generated_plantuml,
                similarity_descr_dist_srs=similarity_text
            )

        return similarity_text


def handle_send_non_stream(srs_type: str, srs_text: str, provider: LLMProvider, model: str, temperature: float, chatbot: list, db_store: bool = True):
    """
    Non-stream version of handle_send. 
    """
    if not srs_text.strip():
        return "", chatbot, ""

    chatbot.append((srs_text, ""))

    if srs_type == "original":
        state.original_srs = srs_text
    if srs_type == "distilled":
        state.distilled_srs = srs_text

    prompt = generate_uml_prompt(srs_text)
    response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)

    full_response = "".join(list(response_generator))
    chatbot[-1] = (srs_text, full_response)

    clean_uml = extract_plantuml_code(full_response)

    if db_store:
        if srs_type == "original":
            store_usage_log(
                event_type="ALL_IN_ONE_ORIG_SRS",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                original_srs_text=srs_text,
                plantuml_script_orig_srs=clean_uml
            )
        if srs_type == "distilled":
            store_usage_log(
                event_type="ALL_IN_ONE_DIST_SRS",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                distilled_srs_text=srs_text,
                plantuml_script_dist_srs=clean_uml
            )

    return srs_text, chatbot, clean_uml


def handle_all_in_one(srs_type: str, srs_text: str, provider: LLMProvider, model: str, temperature: float, chatbot: list):
    """
    1) handle_send_non_stream => obtains UML script
    2) handle_generate_uml => renders UML diagram
    3) handle_check_similarity => obtains similarity analysis
    """
    # Step 1: Non-stream SRS => UML script
    new_srs_text, new_chatbot, uml_script = handle_send_non_stream(
        srs_type, srs_text, provider, model, temperature, chatbot, db_store=False
    )

    # Step 2: Generate UML diagram
    outfile = handle_generate_uml(srs_type, uml_script, db_store=False)
    image_data = None
    if outfile:
        with open(outfile, 'rb') as f:
            image_data = f.read()

    # Step 3: Check similarity
    similarity_text = handle_check_similarity(srs_type, provider, model, temperature, db_store=False)

    # We store everything at once
    if srs_type == "original":
        store_usage_log(
            event_type="ALL_IN_ONE_ORIG_SRS",
            llm_provider=provider,
            llm_model=model,
            temperature=temperature,
            original_srs_text=new_srs_text,
            plantuml_script_orig_srs=uml_script,
            uml_image_data_orig_srs=image_data,
            similarity_descr_orig_srs=similarity_text
        )

    # Return what the UI expects:
    return new_srs_text, new_chatbot, uml_script, outfile, similarity_text


def handle_distill_and_all_in_one(original_srs_text: str, provider: LLMProvider, model: str, temperature: float, chatbot: list):
    """
    1) Distills the original SRS.
    2) Calls handle_all_in_one(...) with srs_type='distilled'.
    """
    # Step 1: Distill
    distill_prompt, distilled_srs = distill_srs_text(provider, model, temperature, original_srs_text, db_store=False)

    # Step 2: Pass the distilled SRS to handle_all_in_one
    #         (We set srs_type="distilled")
    new_srs_text, new_chatbot, uml_script, outfile, similarity_text = handle_all_in_one(
        srs_type="distilled",
        srs_text=distilled_srs,
        provider=provider,
        model=model,
        temperature=temperature,
        chatbot=chatbot
    )
    
    image_data = None
    if outfile:
        with open(outfile, 'rb') as f:
            image_data = f.read()

    store_usage_log(
            event_type="ALL_IN_ONE_DIST_SRS",
            original_srs_text=original_srs_text,
            llm_provider=provider,
            llm_model=model,
            temperature=temperature,
            prompt_distill_srs=distill_prompt,
            distilled_srs_text=distilled_srs,
            plantuml_script_dist_srs=uml_script,
            uml_image_data_dist_srs=image_data,
            similarity_descr_dist_srs=similarity_text
        )

    # Return them. Notice we return distilled_srs as the *first* item
    return (
        distilled_srs,       # <--- The newly distilled SRS
        new_srs_text,        # from handle_all_in_one
        new_chatbot,         # from handle_all_in_one
        uml_script,          # from handle_all_in_one
        outfile,             # from handle_all_in_one
        similarity_text      # from handle_all_in_one
    )


def handle_all_in_one_original_and_distilled(
    original_srs_text: str,
    provider: LLMProvider,
    model: str,
    temperature: float,
    chatbot: list
):
    """
    1) Run the 'non-stream' UML generation & similarity steps on the original SRS (like handle_all_in_one),
       but with db_store=False to avoid partial logs.
    2) Distill the same original SRS (db_store=False).
    3) Run the 'non-stream' UML generation & similarity steps again for the distilled SRS, also with db_store=False.
    4) Finally, log EVERYTHING in a single store_usage_log call with event_type="ALL_IN_ONE_ORIG_AND_DIST".

    Returns a tuple containing the data for both flows:
      (
        # -- original flow --
        orig_srs_text,
        orig_chatbot,
        orig_uml_script,
        orig_outfile,
        orig_sim_text,

        # -- distillation --
        dist_prompt,
        dist_srs_text,

        # -- distilled flow --
        dist_final_srs_text,
        dist_chatbot,
        dist_uml_script,
        dist_outfile,
        dist_sim_text
      )
    """

    if model == "o1 mini":
        temperature = 1.0  # Override temperature for o1-mini

    ################################################################
    # (A) Original Flow (non-stream, UML generation, similarity)
    ################################################################
    # Step A1: Non-stream handle_send => get UML script (no DB logs)
    orig_srs_text, orig_chatbot, orig_uml_script = handle_send_non_stream(
        srs_type="original",
        srs_text=original_srs_text,
        provider=provider,
        model=model,
        temperature=temperature,
        chatbot=chatbot,
        db_store=False
    )

    # Step A2: Generate UML diagram (no DB logs)
    orig_outfile = handle_generate_uml("original", orig_uml_script, db_store=False)
    orig_image_data = None
    if orig_outfile:
        with open(orig_outfile, 'rb') as f:
            orig_image_data = f.read()

    # Step A3: Check similarity for original (no DB logs)
    orig_sim_text = handle_check_similarity(
        srs_type="original",
        provider=provider,
        model=model,
        temperature=temperature,
        db_store=False
    )

    ################################################################
    # (B) Distillation
    ################################################################
    dist_prompt, dist_srs_text = distill_srs_text(
        provider=provider,
        model=model,
        temperature=temperature,
        original_srs_text=original_srs_text,
        db_store=False  # don't log yet
    )

    ################################################################
    # (C) Distilled Flow (non-stream, UML generation, similarity)
    ################################################################
    # Step C1: Non-stream handle_send => get UML script (no DB logs)
    dist_final_srs_text, dist_chatbot, dist_uml_script = handle_send_non_stream(
        srs_type="distilled",
        srs_text=dist_srs_text,
        provider=provider,
        model=model,
        temperature=temperature,
        chatbot=orig_chatbot,
        db_store=False
    )

    # Step C2: Generate UML diagram (no DB logs)
    dist_outfile = handle_generate_uml("distilled", dist_uml_script, db_store=False)
    dist_image_data = None
    if dist_outfile:
        with open(dist_outfile, 'rb') as f:
            dist_image_data = f.read()

    # Step C3: Check similarity for distilled (no DB logs)
    dist_sim_text = handle_check_similarity(
        srs_type="distilled",
        provider=provider,
        model=model,
        temperature=temperature,
        db_store=False
    )

    ################################################################
    # (D) Now log EVERYTHING at once
    ################################################################
    store_usage_log(
        event_type="ALL_IN_ONE_ORIG_AND_DIST",
        llm_provider=provider,
        llm_model=model,
        temperature=temperature,
        # SRS Original data
        original_srs_text=orig_srs_text,
        plantuml_script_orig_srs=orig_uml_script,
        uml_image_data_orig_srs=orig_image_data,
        similarity_descr_orig_srs=orig_sim_text,
        # SRS Distilled data
        prompt_distill_srs=dist_prompt,
        distilled_srs_text=dist_srs_text,
        plantuml_script_dist_srs=dist_uml_script,
        uml_image_data_dist_srs=dist_image_data,
        similarity_descr_dist_srs=dist_sim_text
    )

    ################################################################
    # (E) Return the results from both flows
    ################################################################
    return (
        # Original flow results
        orig_srs_text,       # 1) final original SRS text
        orig_chatbot,        # 2) updated chatbot after original UML generation
        orig_uml_script,     # 3) UML script from original flow
        orig_outfile,        # 4) path to UML diagram for original
        orig_sim_text,       # 5) similarity analysis for original

        # Distillation
        # dist_prompt,         # 6) the prompt used for distillation
        dist_srs_text,       # 7) the newly distilled SRS

        # Distilled flow results
        # dist_final_srs_text, # 8) final text from handle_send_non_stream (usually same as dist_srs_text)
        dist_chatbot,        # 9) updated chatbot after distilled UML generation
        dist_uml_script,     # 10) UML script from distilled flow
        dist_outfile,        # 11) path to UML diagram for distilled
        dist_sim_text        # 12) similarity analysis for distilled
    )

 


