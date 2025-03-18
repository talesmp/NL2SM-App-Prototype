
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
    if srs_type == "distilled":
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

