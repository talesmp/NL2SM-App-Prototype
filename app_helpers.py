# app_helpers.py

from dataclasses import dataclass
import json

import gradio as gr
from plantweb.render import render_file
import uuid

from app_utils_llm import (
    LLMProvider,
    OpenAIModels,
    # MetaAIModels,
    AnthropicModels,
    process_llm_request
)
from app_utils_db import store_usage_log, DB_PATH, retrieve_usage_logs

###############################################################################
# Conversation State
###############################################################################
@dataclass
class ConversationState:
    original_srs: str = ""
    distilled_srs: str = ""
    orig_generated_plantuml: str = ""
    dist_generated_plantuml: str = ""
    chat_history: list = None

    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []

# We create a single shared instance.  Alternatively, you could
# pass it around as needed, or store it in a Gradio State object.
state_original = ConversationState()
state_distilled = ConversationState()

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
1. The extracted description of the system represented by the PlantUML script.
2. A similarity score between 0 and 100
3. A list of specific discrepancies
4. Any missing elements in the UML diagram that weren't specified in the SRS
5. Any additional elements in the UML diagram that weren't specified in the SRS

Format your response as JSON with five keys: 'extracted_description' 'similarity_score', 'discrepancies', 'missing_elements' and 'additional_elements'.
'extracted_description' should be a string.
'similarity_score' should be a float between 0 and 100.
'discrepancies', 'missing_elements' and 'additional_elements' should be lists of strings.

This is an example of the structure of the JSON that you must follow:
```json
    "extracted_description": "The system is a Book Store with classes for Book, Author, Publisher, and User. Books can be bought and sold by the Bookstore and Individuals. Users can borrow books from the Bookstore. Books have titles, authors, and ISBNs. Users have names and can register with the Bookstore.",
    "similarity_score": 75.0,
    "discrepancies": ["- The UML diagram doesn't explicitly show that Publishers cannot buy books (no purchase method)", "- The lending rules constraints aren't fully represented in the UML (e.g., can't show that Individuals cannot lend borrowed books)"],
    "missing_elements": ["- No indication of Copy's default owner being Publisher", "- No representation of inter-library lending capability"],
    "additional_elements": ["- Methods that weren't explicitly mentioned in SRS:  * transferOwnership() in Owner class  * sellBook() in Bookstore and Individual  * purchaseBook() in multiple classes", "- The User class is more detailed than specified in the SRS, including attributes like id, name, and registrationDate"]
```

Respond only with the JSON object, without any additional text or explanation.
"""


def clean_and_parse_json_response(text):
    # Remove the starting ```json if present
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    # Remove the ending ``` if present
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    try:
        parsed_data = json.loads(text)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return None
    return parsed_data


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

    if srs_type == "original":
        state_original.orig_generated_plantuml = plantuml_script
    if srs_type == "distilled":
        state_distilled.dist_generated_plantuml = plantuml_script

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

def handle_check_similarity(srs_type: str, provider: LLMProvider, model: str, temperature: float, db_store: bool = True) -> str:
    """Compare SRS text to the UML diagram content in state."""
    if srs_type == "original":
        if not state_original.original_srs or not state_original.orig_generated_plantuml:
            return "Please provide the original SRS text and UML script first."

        prompt = check_similarity_prompt(state_original.original_srs, state_original.orig_generated_plantuml)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
        similarity_text = "".join(list(response_generator))  # Collect all pieces

        if db_store:
            store_usage_log(
                event_type="CHECK_SIMILARITY_ORIG",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                original_srs_text=state_original.original_srs,
                plantuml_script_orig_srs=state_original.orig_generated_plantuml,
                similarity_descr_orig_srs=similarity_text
            )

        return similarity_text

    if srs_type == "distilled":
        if not state_distilled.distilled_srs or not state_distilled.dist_generated_plantuml:
            return "Please provide the distilled SRS text and UML script first."

        prompt = check_similarity_prompt(state_distilled.distilled_srs, state_distilled.dist_generated_plantuml)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
        similarity_text = "".join(list(response_generator))

        if db_store:
            store_usage_log(
                event_type="CHECK_SIMILARITY_DIST",
                llm_provider=provider,
                llm_model=model,
                temperature=temperature,
                distilled_srs_text=state_distilled.distilled_srs,
                plantuml_script_dist_srs=state_distilled.dist_generated_plantuml,
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
        state_original.original_srs = srs_text
    if srs_type == "distilled":
        state_distilled.distilled_srs = srs_text

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


def handle_all_in_one_original_and_distilled(
    srs_id: str,
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

    parsed_orig_sim_text = clean_and_parse_json_response(orig_sim_text)
    # if parsed_orig_sim_text:
        # Save the values in the required variables
    orig_extr_desc = parsed_orig_sim_text.get("extracted_description", "")
    orig_sim_score = parsed_orig_sim_text.get("similarity_score", "")
    
    # Concatenate discrepancies, missing_elements, and additional_elements with their keys
    orig_discrepancies = "\n".join(parsed_orig_sim_text.get("discrepancies", []))
    orig_missing_elements = "\n".join(parsed_orig_sim_text.get("missing_elements", []))
    orig_additional_elements = "\n".join(parsed_orig_sim_text.get("additional_elements", []))
    
    orig_sim_desc = f"discrepancies:\n{orig_discrepancies}\n\nmissing_elements:\n{orig_missing_elements}\n\nadditional_elements:\n{orig_additional_elements}"

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

    parsed_dist_sim_text = clean_and_parse_json_response(dist_sim_text)
    print(parsed_dist_sim_text)
    # if parsed_dist_sim_text:
        # Save the values in the required variables
    dist_extr_desc = parsed_dist_sim_text.get("extracted_description", "")
    dist_sim_score = parsed_dist_sim_text.get("similarity_score", "")
    
    # Concatenate discrepancies, missing_elements, and additional_elements with their keys
    dist_discrepancies = "\n".join(parsed_dist_sim_text.get("discrepancies", []))
    dist_missing_elements = "\n".join(parsed_dist_sim_text.get("missing_elements", []))
    dist_additional_elements = "\n".join(parsed_dist_sim_text.get("additional_elements", []))
    
    dist_sim_desc = f"discrepancies:\n{dist_discrepancies}\n\nmissing_elements:\n{dist_missing_elements}\n\nadditional_elements:\n{dist_additional_elements}"

    ################################################################
    # (D) Now log EVERYTHING at once
    ################################################################
    store_usage_log(
        event_type="FIRST_PASS",
        llm_provider=provider,
        llm_model=model,
        temperature=temperature,
        srs_id=srs_id,
        # SRS Original data
        original_srs_text=orig_srs_text,
        plantuml_script_orig_srs=orig_uml_script,
        uml_image_data_orig_srs=orig_image_data,
        extracted_uml_descr_orig_srs=orig_extr_desc,
        similarity_score_orig_srs=orig_sim_score,
        similarity_descr_orig_srs=orig_sim_desc,
        # SRS Distilled data
        distilled_srs_text=dist_srs_text,
        plantuml_script_dist_srs=dist_uml_script,
        uml_image_data_dist_srs=dist_image_data,
        extracted_uml_descr_dist_srs=dist_extr_desc,
        similarity_score_dist_srs=dist_sim_score,
        similarity_descr_dist_srs=dist_sim_desc
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


def handle_send_backfeed_non_stream(
        srs_type: str, 
        provider: LLMProvider, 
        model: str, 
        temperature: float, 
        chatbot: list, 
        original_srs_text: str = None,
        distilled_srs_text: str = None,
        plantuml_script_orig_srs: str = None,
        plantuml_script_dist_srs: str = None,
        similarity_descr_orig_srs: str = None,
        similarity_descr_dist_srs: str = None
    ):
    """
    Non-stream version of handle_backfeed_send. 
    This function calls the LLM to generate a new UML script from the SRS text, the previously generated UML script and the similarity analysis content.
    """
    if srs_type == "original":
        if not original_srs_text.strip():
            return "", chatbot, ""

        chatbot.append((original_srs_text, ""))
        state_original.original_srs = original_srs_text

        def regenerate_orig_plantuml_prompt(original_srs_text, plantuml_script_orig_srs, similarity_descr_orig_srs):
            return f"""
            You are a software requirements analyst. You have been provided with the following information regarding an iterative process to generate a UML Class Diagram from a Software Requirements Specification (SRS):
        1. Original Software Requirements Specification: 
        {original_srs_text}

        2. Previously generated UML script: 
        {plantuml_script_orig_srs}
        
        3. Previous Similarity Analysis highlighting discrepancies, missing elements, and additional elements:
        {similarity_descr_orig_srs}

        Based on this information, please generate a new UML script that accurately represents the system described in the Software Requirements Specification.
        Generate only the PlantUML script, starting with @startuml and ending with @enduml. Include appropriate classes, attributes, methods, and relationships.
        """
        prompt = regenerate_orig_plantuml_prompt(original_srs_text, plantuml_script_orig_srs, similarity_descr_orig_srs)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)

        full_response = "".join(list(response_generator))
        chatbot[-1] = (original_srs_text, full_response)

        clean_uml = extract_plantuml_code(full_response)

    if srs_type == "distilled":
        if not distilled_srs_text.strip():
            return "", chatbot, ""

        chatbot.append((distilled_srs_text, ""))
        state_distilled.distilled_srs = distilled_srs_text

        def regenerate_dist_plantuml_prompt(distilled_srs_text, plantuml_script_dist_srs, similarity_descr_dist_srs):
            return f"""
            You are a software requirements analyst. You have been provided with the following information regarding an iterative process to generate a UML Class Diagram from a Software Requirements Specification (SRS):
        1. Previously pruned Software Requirements Specification: 
        {distilled_srs_text}

        2. Previously generated UML script: 
        {plantuml_script_dist_srs}
        
        3. Previous Similarity Analysis highlighting discrepancies, missing elements, and additional elements:
        {similarity_descr_dist_srs}

        Based on this information, please generate a new UML script that accurately represents the system described in the Software Requirements Specification.
        Generate only the PlantUML script, starting with @startuml and ending with @enduml. Include appropriate classes, attributes, methods, and relationships.
        """
        prompt = regenerate_dist_plantuml_prompt(distilled_srs_text, plantuml_script_dist_srs, similarity_descr_dist_srs)
        response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)

        full_response = "".join(list(response_generator))
        chatbot[-1] = (distilled_srs_text, full_response)

        clean_uml = extract_plantuml_code(full_response)

    return chatbot, clean_uml


def handle_recheck_similarity(
        srs_type: str, 
        provider: LLMProvider, 
        model: str, 
        temperature: float,
        srs_text: str = None,
        new_uml_script: str = None,
        ) -> str:
    """Compare SRS text to the UML diagram content in state."""
    prompt = check_similarity_prompt(srs_text, new_uml_script)
    response_generator = process_llm_request(provider, model, prompt, temperature, stream=False)
    similarity_text = "".join(list(response_generator))  # Collect all pieces

    return similarity_text


def handle_backfeed_for_improvement(
    srs_id: str,
    provider: LLMProvider,
    model: str,
    temperature: float,
    chatbot: list
):
    """
    1) Using the SRS_ID, Provider, Model, and Temperature, retrieve from the latest log entry:
        - Original SRS text
        - Distilled SRS text
        - PlantUML script for original SRS
        - PlantUML script for distilled SRS
        - Similarity analysis for original SRS
        - Similarity analysis for distilled SRS
    """

    if model == "o1 mini":
        temperature = 1.0  # Override temperature for o1-mini

    # Retrieve the latest log entry
    original_srs_text, distilled_srs_text, plantuml_script_orig_srs, plantuml_script_dist_srs, similarity_descr_orig_srs, similarity_descr_dist_srs, record_count = retrieve_usage_logs(srs_id, provider, model, temperature)
    if original_srs_text is None:
        return None


    ################################################################
    # (A) Original Flow (non-stream, UML generation, similarity)
    ################################################################
    # Step A1: Non-stream handle_send => get UML script (no DB logs)
    orig_chatbot, new_orig_uml_script = handle_send_backfeed_non_stream(
        srs_type="original",
        provider=provider,
        model=model,
        temperature=temperature,
        chatbot=chatbot,
        original_srs_text=original_srs_text,
        plantuml_script_orig_srs=plantuml_script_orig_srs,
        similarity_descr_orig_srs=similarity_descr_orig_srs
    )

    # Step A2: Generate UML diagram (no DB logs)
    new_orig_outfile = handle_generate_uml("original", new_orig_uml_script, db_store=False)
    new_orig_image_data = None
    if new_orig_outfile:
        with open(new_orig_outfile, 'rb') as f:
            new_orig_image_data = f.read()

    # Step A3: Check similarity for original (no DB logs)
    new_orig_sim_text = handle_recheck_similarity(
        srs_type="original",
        provider=provider,
        model=model,
        temperature=temperature,
        srs_text=original_srs_text,
        new_uml_script=new_orig_uml_script
    )

    new_parsed_orig_sim_text = clean_and_parse_json_response(new_orig_sim_text)
    if new_parsed_orig_sim_text:
        # Save the values in the required variables
        new_orig_extr_desc = new_parsed_orig_sim_text.get("extracted_description", "")
        new_orig_sim_score = new_parsed_orig_sim_text.get("similarity_score", "")
        
        # Concatenate discrepancies, missing_elements, and additional_elements with their keys
        new_orig_discrepancies = "\n".join(new_parsed_orig_sim_text.get("discrepancies", []))
        new_orig_missing_elements = "\n".join(new_parsed_orig_sim_text.get("missing_elements", []))
        new_orig_additional_elements = "\n".join(new_parsed_orig_sim_text.get("additional_elements", []))
        
        new_orig_sim_desc = f"discrepancies:\n{new_orig_discrepancies}\n\nmissing_elements:\n{new_orig_missing_elements}\n\nadditional_elements:\n{new_orig_additional_elements}"

    ################################################################
    # (C) Distilled Flow (non-stream, UML generation, similarity)
    ################################################################
    # Step C1: Non-stream handle_send => get UML script (no DB logs)
    dist_chatbot, new_dist_uml_script = handle_send_backfeed_non_stream(
        srs_type="distilled",
        provider=provider,
        model=model,
        temperature=temperature,
        chatbot=orig_chatbot,
        distilled_srs_text=distilled_srs_text,
        plantuml_script_dist_srs=plantuml_script_dist_srs,
        similarity_descr_dist_srs=similarity_descr_dist_srs
    )

    # Step C2: Generate UML diagram (no DB logs)
    new_dist_outfile = handle_generate_uml("distilled", new_dist_uml_script, db_store=False)
    new_dist_image_data = None
    if new_dist_outfile:
        with open(new_dist_outfile, 'rb') as f:
            new_dist_image_data = f.read()

    # Step C3: Check similarity for distilled (no DB logs)
    new_dist_sim_text = handle_recheck_similarity(
        srs_type="distilled",
        provider=provider,
        model=model,
        temperature=temperature,
        srs_text=distilled_srs_text,
        new_uml_script=new_dist_uml_script
    )

    new_parsed_dist_sim_text = clean_and_parse_json_response(new_dist_sim_text)
    if new_parsed_dist_sim_text:
        # Save the values in the required variables
        new_dist_extr_desc = new_parsed_dist_sim_text.get("extracted_description", "")
        new_dist_sim_score = new_parsed_dist_sim_text.get("similarity_score", "")
        
        # Concatenate discrepancies, missing_elements, and additional_elements with their keys
        new_dist_discrepancies = "\n".join(new_parsed_dist_sim_text.get("discrepancies", []))
        new_dist_missing_elements = "\n".join(new_parsed_dist_sim_text.get("missing_elements", []))
        new_dist_additional_elements = "\n".join(new_parsed_dist_sim_text.get("additional_elements", []))
        
        new_dist_sim_desc = f"discrepancies:\n{new_dist_discrepancies}\n\nmissing_elements:\n{new_dist_missing_elements}\n\nadditional_elements:\n{new_dist_additional_elements}"

    ################################################################
    # (D) Now log EVERYTHING at once
    ################################################################
    store_usage_log(
        event_type=f"BACKFEED_IMPROVEMENT_{record_count}",
        llm_provider=provider,
        llm_model=model,
        temperature=temperature,
        srs_id=srs_id,
        # SRS Original data
        original_srs_text=original_srs_text,
        plantuml_script_orig_srs=new_orig_uml_script,
        uml_image_data_orig_srs=new_orig_image_data,
        extracted_uml_descr_orig_srs=new_orig_extr_desc,
        similarity_score_orig_srs=new_orig_sim_score,
        similarity_descr_orig_srs=new_orig_sim_desc,
        # SRS Distilled data
        distilled_srs_text=distilled_srs_text,
        plantuml_script_dist_srs=new_dist_uml_script,
        uml_image_data_dist_srs=new_dist_image_data,
        extracted_uml_descr_dist_srs=new_dist_extr_desc,
        similarity_score_dist_srs=new_dist_sim_score,
        similarity_descr_dist_srs=new_dist_sim_desc
    )

    ################################################################
    # (E) Return the results from both flows
    ################################################################
    return (
        # Original flow results
        original_srs_text,       # 1) final original SRS text
        orig_chatbot,        # 2) updated chatbot after original UML generation
        new_orig_uml_script,     # 3) UML script from original flow
        new_orig_outfile,        # 4) path to UML diagram for original
        new_orig_sim_text,       # 5) similarity analysis for original

        # Distillation
        # dist_prompt,         # 6) the prompt used for distillation
        distilled_srs_text,       # 7) the previously distilled SRS

        # Distilled flow results
        # dist_final_srs_text, # 8) final text from handle_send_non_stream (usually same as dist_srs_text)
        dist_chatbot,        # 9) updated chatbot after distilled UML generation
        new_dist_uml_script,     # 10) UML script from distilled flow
        new_dist_outfile,        # 11) path to UML diagram for distilled
        new_dist_sim_text        # 12) similarity analysis for distilled
    )

