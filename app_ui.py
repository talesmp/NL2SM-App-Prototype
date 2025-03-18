# ui.py

import gradio as gr
from app_helpers import (
    handle_all_in_one_original_and_distilled,
    handle_send_non_stream,
    handle_generate_uml,
    handle_check_similarity,
    clean_and_parse_json_response,
    distill_srs_text,
    handle_backfeed_for_improvement,
    handle_send_backfeed_non_stream,
    handle_recheck_similarity,
    handle_send_backfeed_non_stream,
    state # So we have a reference if needed
)

from app_utils_llm import (
    LLMProvider,
    OpenAIModels,
    # MetaAIModels,
    AnthropicModels,
    process_llm_request
)

from app_utils_db import initialize_db, DB_PATH


def get_available_models(provider_value):
    """Return possible models for a given provider enum value."""
    if provider_value == LLMProvider.OPENAI.value:
        return [m.value for m in OpenAIModels]
    # elif provider_value == LLMProvider.META.value:
    #     return [m.value for m in MetaAIModels]
    elif provider_value == LLMProvider.ANTHROPIC.value:
        return [m.value for m in AnthropicModels]
    return []


def create_ui():
    """
    Your 'new' UI from the code. You can adapt as needed.
    """
    initialize_db()

    with gr.Blocks(fill_width=True) as app:
        with gr.Row(): # LLM provider, model and temperature selection
            with gr.Column():
                srs_id = gr.Textbox(
                    label="Software Artifact ID"
                )
            
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
                    return gr.update(
                        choices=get_available_models(provider),
                        value=get_available_models(provider)[0] if len(get_available_models(provider))>0 else None
                    )
                provider.change(
                    fn=update_model,
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
                def update_temperature(selected_model):
                    if selected_model == OpenAIModels.O1_MINI.value:
                        return gr.update(value=1.0, minimum=1.0, maximum=1.0, step=0.0)
                    else:
                        return gr.update(value=0.5, minimum=0.0, maximum=1.0, step=0.1)

                model.change(
                    fn=update_temperature,
                    inputs=model,
                    outputs=temperature
                )

        with gr.Row(): # Original SA text and Pruned SA text
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
            with gr.Column(): 
                srs_text_original = gr.TextArea(
                    label="Original Software Artifact (oSA)",
                    lines=18,
                    max_lines=18,
                    value=srs_placeholder
                )
                # all_in_one_original_and_distilled_btn = gr.Button("Send to LLM (All in One)")
                with gr.Row():
                    with gr.Column():
                        all_in_one_original_and_distilled_btn = gr.Button("Send to LLM (Original & Prune-based approaches)")
                    with gr.Column():
                        backfeed_for_improvement_btn = gr.Button("Back-feed for Improvement")
            with gr.Column():
                srs_text_distilled = gr.TextArea(
                    label="Pruned Software Artifact (pSA)",
                    lines=21,
                    max_lines=21
                )

        with gr.Row():  # Original and Pruned PlantUML scripts
            with gr.Column():
                plantuml_script_original = gr.TextArea(
                    label="PlantUML Script (oSA)",
                    lines=12,
                    max_lines=12,
                    show_copy_button=True
                )
            with gr.Column():
                plantuml_script_distilled = gr.TextArea(
                    label="PlantUML Script (pSA)",
                    lines=12,
                    max_lines=12,
                    show_copy_button=True
                )

        with gr.Row():  # Original and Pruned UML images
            with gr.Column():
                uml_image_original = gr.Image(
                    type="pil",
                    label="PlantUML Rendered Diagram (oSA)",
                    height=306,
                    show_download_button=True,
                    show_fullscreen_button=True
                )
            with gr.Column():
                uml_image_distilled = gr.Image(
                    type="pil",
                    label="PlantUML Rendered Diagram (pSA)",
                    height=306,
                    show_download_button=True,
                    show_fullscreen_button=True,
                    value=None
                )

        with gr.Row():  # Original and Pruned Similarity Analysis
            with gr.Column():
                similarity_result_original = gr.TextArea(
                    label="Similarity Analysis (oSA)",
                    lines=12
                )
            with gr.Column():
                similarity_result_distilled = gr.TextArea(
                    label="Similarity Analysis (pSA)",
                    lines=12
                )

        with gr.Row():  # Original and Pruned Chatbots
            with gr.Column():
                chatbot_original = gr.Chatbot(height=400)
            with gr.Column():
                chatbot_distilled = gr.Chatbot(height=400)

        # Wire up event

        all_in_one_original_and_distilled_btn.click(
            fn=handle_all_in_one_original_and_distilled,
            inputs=[
                srs_id,
                srs_text_original, 
                provider, 
                model, 
                temperature, 
                chatbot_original
            ],
            outputs=[
                srs_text_original, 
                chatbot_original,
                plantuml_script_original, 
                uml_image_original, 
                similarity_result_original, 
                srs_text_distilled, 
                chatbot_distilled, 
                plantuml_script_distilled,
                uml_image_distilled,
                similarity_result_distilled
            ]
        )

        backfeed_for_improvement_btn.click(
            fn=handle_backfeed_for_improvement,
            inputs=[
                srs_id,
                provider, 
                model, 
                temperature, 
                chatbot_original
            ],
            outputs=[
                srs_text_original, 
                chatbot_original,
                plantuml_script_original, 
                uml_image_original, 
                similarity_result_original, 
                srs_text_distilled, 
                chatbot_distilled, 
                plantuml_script_distilled,
                uml_image_distilled,
                similarity_result_distilled
            ]
        )

    return app
