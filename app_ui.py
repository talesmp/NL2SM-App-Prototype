# ui.py

import gradio as gr
import sqlite3
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
    state_original, # So we have a reference if needed
    state_distilled # So we have a reference if needed
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

def get_unique_sa_ids():
    """
    Connects to the database, queries unique SA IDs from the srs_id column in the usage_logs table,
    and returns them as a list.
    """
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()
        db_curs.execute("SELECT DISTINCT srs_id FROM usage_logs ORDER BY srs_id")
        rows = db_curs.fetchall()
    return [row[0] for row in rows]

def get_additional_options(selected_sa_id):
    """
    Given a selected SA ID, queries the database for unique values for:
      - LLM Provider,
      - LLM Model,
      - Temperature,
      - Event Type.
    Returns update objects for each dropdown to make them visible and populate their choices.
    """
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()
        # Query unique LLM Providers for the selected SA ID
        db_curs.execute("SELECT DISTINCT llm_provider FROM usage_logs WHERE srs_id = ? ORDER BY llm_provider", (selected_sa_id,))
        providers = [row[0] for row in db_curs.fetchall()]
        # Query unique LLM Models for the selected SA ID
        db_curs.execute("SELECT DISTINCT llm_model FROM usage_logs WHERE srs_id = ? ORDER BY llm_model", (selected_sa_id,))
        models = [row[0] for row in db_curs.fetchall()]
        # Query unique Temperature values for the selected SA ID
        db_curs.execute("SELECT DISTINCT temperature FROM usage_logs WHERE srs_id = ? ORDER BY temperature", (selected_sa_id,))
        temperatures = [row[0] for row in db_curs.fetchall()]
        # Query unique Event Types for the selected SA ID
        # db_curs.execute("SELECT DISTINCT event_type FROM usage_logs WHERE srs_id = ? ORDER BY event_type", (selected_sa_id,))
        # event_types = [row[0] for row in db_curs.fetchall()]
    return (
        gr.Dropdown.update(choices=providers, visible=True),
        gr.Dropdown.update(choices=models, visible=True),
        gr.Dropdown.update(choices=temperatures, visible=True),
        # gr.Dropdown.update(choices=event_types, visible=True)
    )

initial_sa_ids = get_unique_sa_ids()

def create_ui():
    """
    Your 'new' UI from the code. You can adapt as needed.
    """
    initialize_db()
    js_func = """
        function refresh() {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'light') {
                url.searchParams.set('__theme', 'light');
                window.location.href = url.href;
            }
        }
        """

    with gr.Blocks(fill_width=True, js=js_func, theme="ocean") as app: # theme="ocean"
        with gr.Tabs():

            with gr.TabItem("SA Transformation",):
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
                            interactive=True,
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
                                all_in_one_original_and_distilled_btn = gr.Button("Send to LLM", variant="primary")
                            with gr.Column():
                                backfeed_for_improvement_btn = gr.Button("Back-feed for Improvement", variant="primary")
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
                            lines=21,
                            max_lines=21
                        )
                    with gr.Column():
                        similarity_result_distilled = gr.TextArea(
                            label="Similarity Analysis (pSA)",
                            lines=21,
                            max_lines=21
                        )

                with gr.Row():  # Original and Pruned Chatbots
                    with gr.Column():
                        chatbot_original = gr.Chatbot(height=400, label="Original Chatbot")
                    with gr.Column():
                        chatbot_distilled = gr.Chatbot(height=400, label="Distilled Chatbot")


            with gr.TabItem("Log Retrieval"):
                gr.Markdown("""# How does Log Retrieval work?
                            1. When loading the tab, a call to the database should be done in order to retrieve a list of unique identifiers in `srs_id` for each Software Artifact and display this list in the field "SA ID";  
                            2. Upon selecting an SA ID, two columns containing following fields will be available for selection:
                                - LLM Provider, given a list of available unique providers given the selected SA ID;
                                - LLM Model, a list of available unique models given the selected SA ID;
                                - Temperature, a list of available unique temperature values given the selected SA ID.
                                - Event Type, a list of available unique event types given the selected SA ID.
                            3. Upon selecting the LLM Provider, LLM Model, Temperature, and Event Type, the user can click the "Retrieve Logs" button to retrieve logs.
                            4. The logs will be displayed in the following order, in two columns, so they can be compared side-by-side:
                                - Similarity Score
                                - Generated UML Diagram (Original)
                                - Similarity Analysis (oSA)
                                - Generated UML Diagram (Distilled)
                                - Similarity Analysis (pSA)
                            """)
                
                with gr.Row(): # SA ID, LLM Provider, Model and Temperature selection
                    with gr.Column():
                        sa_id_retrieval = gr.Dropdown(
                            label="Select Software Artifact ID",
                            choices=initial_sa_ids,
                            value=initial_sa_ids[0] if initial_sa_ids else None,
                            interactive=True
                        )
                    with gr.Column():
                        provider_for_sa_id = gr.Dropdown(label="LLM Provider", choices=[], visible=False)
                    with gr.Column():
                        model_for_sa_id = gr.Dropdown(label="LLM Model", choices=[], visible=False)
                    with gr.Column():
                        temperature_for_sa_id = gr.Dropdown(label="Temperature", choices=[], visible=False)

                    sa_id_retrieval.change(
                        fn=get_additional_options,
                        inputs=sa_id_retrieval,
                        outputs=[provider_for_sa_id, model_for_sa_id, temperature_for_sa_id] #, event_type_left_for_sa_id, event_type_right_for_sa_id]
                        )

                with gr.Row(): # Event Type selection
                    with gr.Column():
                        event_type_left_for_sa_id = gr.Dropdown(
                            label="Event Type (Left)"
                        )
                    with gr.Column():
                        event_type_right_for_sa_id = gr.Dropdown(
                            label="Event Type (Right)"
                        )
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                clear_options_button = gr.Button("Clear")
                            with gr.Column():
                                retrieve_logs_button = gr.Button("Retrieve Logs")

                with gr.Row(): # Left and Right Similarity Scores
                    with gr.Column():
                        similarity_score_left = gr.Textbox(
                            label="Similarity Score (Left)"
                        )
                    with gr.Column():
                        similarity_score_right = gr.Textbox(
                            label="Similarity Score (Right)"
                        )

                with gr.Row():  # Left and Right Original UML images
                    with gr.Column():
                        uml_image_original_left = gr.Image(
                            type="pil",
                            label="PlantUML Rendered Diagram (oSA)",
                            height=306,
                            show_download_button=True,
                            show_fullscreen_button=True
                        )
                    with gr.Column():
                        uml_image_original_right = gr.Image(
                            type="pil",
                            label="PlantUML Rendered Diagram (oSA)",
                            height=306,
                            show_download_button=True,
                            show_fullscreen_button=True
                        )

                with gr.Row():  # Left and Right Original Semantic Analysis
                    with gr.Column():
                        semantic_analysis_left = gr.TextArea(
                            label="Semantic Analysis (Left)"
                        )
                    with gr.Column():
                        semantic_analysis_right = gr.TextArea(
                            label="Semantic Analysis (Right)"
                        )
                
                with gr.Row():  # Left and Right Distilled/Pruned UML images
                    with gr.Column():
                        uml_image_distilled_left = gr.Image(
                            type="pil",
                            label="PlantUML Rendered Diagram (pSA)",
                            height=306,
                            show_download_button=True,
                            show_fullscreen_button=True
                        )
                    with gr.Column():
                        uml_image_distilled_right = gr.Image(
                            type="pil",
                            label="PlantUML Rendered Diagram (pSA)",
                            height=306,
                            show_download_button=True,
                            show_fullscreen_button=True
                        )
                    
                with gr.Row():  # Left and Right Distilled/Pruned Semantic Analysis
                    with gr.Column():
                        semantic_analysis_distilled_left = gr.TextArea(
                            label="Semantic Analysis (Left)"
                        )
                    with gr.Column():
                        semantic_analysis_distilled_right = gr.TextArea(
                            label="Semantic Analysis (Right)"
                        )

            
            with gr.TabItem("SA Management"):
                gr.Markdown("""# How does SA Management work?
                            1. When loading the tab, a call to the database should be done in order to retrieve a list of unique identifiers in `srs_id` for each Software Artifact and display this list in the field "SA ID";
                            2. Upon selecting an SA ID, the user can click the "Retrieve SA" button to retrieve the Software Artifact;
                            3. The Software Artifact will be displayed in the "SA Text", and it can be viewed by the user.  It shouldn't be editable;
                            4. The user can insert a new SA in the "SA Text" field and click the "Create New SA" button to update the Software Artifact;
                            """)
                # with gr.TabItem("Existing SAs"):
                #     with gr.Row():
                #         with gr.Column():
                #             sa_id = gr.Dropdown(
                #                 label="Software Artifact ID"
                #             )
                #         with gr.Column():
                #             retrieve_sa_button = gr.Button("Retrieve SA", variant="primary")
                #     with gr.Row():
                #         sa_text = gr.TextArea(
                #             label="SA Text",
                #             lines=18,
                #             max_lines=18
                #         )
                # with gr.TabItem("Insert New SA"):
                #     with gr.Row():
                #         new_sa_text = gr.TextArea(
                #             label="SA Text",
                #             lines=18,
                #             max_lines=18
                #         )
                #     with gr.Row():
                #         with gr.Column():
                #             create_new_sa_button = gr.Textbox(label="SA ID")
                #         with gr.Column():
                #             create_new_sa_button = gr.Button("Create New SA (check for existing SA ID)", variant="primary")


            with gr.TabItem("Prompt Management"):
                gr.Markdown("""# How does Prompt Management work?
                            """)


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

