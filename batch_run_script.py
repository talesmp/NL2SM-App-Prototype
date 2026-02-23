import app_utils_llm
import app_helpers
import app_utils_db
import os
from enum import Enum

def handle_all_in_one_for_all_providers_and_models(
    original_srs_text: str,
    srs_id: str = None,
    temperature: float = 1.0
):
    """
    Calls handle_all_in_one_original_and_distilled for EACH provider & model combination.
    Returns a list of dicts, each containing:
      - 'provider': str
      - 'model': str
      - 'result': the returned tuple from handle_all_in_one_original_and_distilled
    """
    all_runs_results = []

    # We'll pass an empty list as `chatbot` for each run (so each run is independent).
    # If you prefer to reuse a single conversation state, pass the same reference.
    for provider_enum in app_utils_llm.LLMProvider:
        model_names = app_utils_llm.get_models_for_provider(provider_enum)
        for model_name in model_names:
            # The handle_* functions expect provider to be a string like "OpenAI", so use .value
            run_result = app_helpers.handle_all_in_one_original_and_distilled(
                srs_id=srs_id,
                original_srs_text=original_srs_text,
                provider=provider_enum.value,
                model=model_name,
                temperature=temperature,
                chatbot=[]
            )

            # Print partial results right away:
            print(f"\n==== Results for Provider: {provider_enum.value}, Model: {model_name} ====")
            print("Original UML Script:", run_result[2])
            print("Original Similarity:", run_result[4])
            print("Distilled SRS:", run_result[5])
            print("Distilled UML Script:", run_result[7])
            print("Distilled Similarity:", run_result[9])

            all_runs_results.append({
                "provider": provider_enum.value,
                "model": model_name,
                "result": run_result
            })

    return all_runs_results


def manual_srs_input_batch_run():
    # original_srs = """
    # We want to model a system that represents chains of movie theaters in a given country, with the following restrictions. 

    # * Each chain owns several movie theaters in different cities of the country. 
    # * The laws of that country prevent more than two movie theaters of the same chain in the same city. 
    # * Each theater has a number of rooms where movies are shown. 
    # * Each film has a duration and a director, and the chains have to pay a fee to the producer of the film each time they show it. 
    # * Each room has a maximum number of seats to accommodate the audience. 
    # * Each room may screen a maximum of five movies on the same day, whether they are the same film or not. 
    # * At least 20 minutes must elapse between the end of one screening and the beginning of the next in the same room. 
    # * Another rule in the country prevents films by a single director from being shown in the same cinema on the same day. 
    # * Customers buy their tickets centrally at each cinema chain, indicating the movie they want to see, the cinema of the chain where they want to see it, the day, the room, the start time, and the number of tickets they want. 
    # * Ticket prices for each movie are set by the chain. The price of a movie is the same for all theaters of the same chain. 
    # * Customers cannot buy more tickets than are currently available in the desired room. 
    # * A customer who has purchased a series of tickets may also return them if he/she does so before the movie starts. 
    # * Each chain has a loyalty card, and offers discounts at its theaters for customers who have it. Discounts are decided by each chain, but cannot exceed 50% of the ticket value, nor be less than 20%. 
    # """
    original_srs = """
    We want to model a system composed of video stores, which are companies that have shops dedicated to renting movies. 

    * Each video store has a name and a set of associated shops, each with a different address. 
    * Customers of a video store can rent movies at any of its shops. 
    * Each shop has a set of copies of each movie for rent, and a clerk who attends the store. 
    * Customers ask the clerks for the movie they wish to rent and, if a copy of the movie is available at that shop, they are granted a 3 or 5 day loan depending on whether the customer is a regular or VIP customer. 
    * The loans must be returned to the same shop where they were rented. 
    * Regular customers can have a maximum of three active loans from the same video store, regardless of the shop they where rented from. VIP customers have no such limit.  
    * No clerk may be a customer of the shop where he/she works. 
    * If a customer returns a copy of a movie late, he/she is blacklisted in that video store and cannot rent any more movies in any of its shops.
    """

    batch_results = handle_all_in_one_for_all_providers_and_models(
        original_srs_text=original_srs,
        temperature=1.0
    )

    # Now do something with batch_results
    # for item in batch_results:
    #     prov = item["provider"]
    #     mdl = item["model"]
    #     res = item["result"]
    #     print(f"\n--- Results for {prov} / {mdl} ---\n")
    #     print(res)  # or handle them further


def main_batch_run_from_db(temperature=0.5, srs_ids=None):
    """
    1. Initializes DB (ensuring tables exist).
    2. Selects all SRS from srs_data table.
    3. For each SRS, calls handle_all_in_one_for_all_providers_and_models(original_srs_text, temperature).
    4. Prints or processes results for each SRS.

    Because usage_logs is written inside handle_all_in_one..., 
    each iteration's results are also saved automatically in the DB.
    """
    app_utils_db.initialize_db()  # ensure srs_data and usage_logs exist

    # Decide which rows to fetch
    if srs_ids is None or len(srs_ids) == 0:
        # Grab all SRS records
        srs_list = app_utils_db.select_all_srs()
    else:
        # Grab only the specified IDs
        srs_list = app_utils_db.select_srs_by_ids(srs_ids)

    for (srs_id, srs_title, srs_content) in srs_list:
        print(f"\n==========================")
        print(f"PROCESSING SRS ID: {srs_id}  TITLE: {srs_title}")
        print("==========================\n")

        #Variable to concatenate the srs_id and srs_title without blank spaces
        srs_title_no_blank = srs_title.replace(" ", "") 
        srs_id_and_title = f"{srs_id}_{srs_title_no_blank}"

        # Call your existing function that enumerates all providers & models
        batch_results = handle_all_in_one_for_all_providers_and_models(
            original_srs_text=srs_content,
            srs_id=srs_id_and_title,
            temperature=temperature
        )

        # Now we can do something with batch_results, e.g., print them:
        for item in batch_results:
            provider = item["provider"]
            model = item["model"]
            run_result = item["result"]
            # run_result is the tuple from handle_all_in_one_original_and_distilled
            print(f"Provider: {provider} | Model: {model}")
            print(f"Original UML script: {run_result[2]}")
            print(f"Original similarity: {run_result[4]}")
            print(f"Distilled UML script: {run_result[7]}")
            print(f"Distilled similarity: {run_result[9]}")
            print("-----------------------------------")

        print(f"Finished processing SRS ID: {srs_id}\n")


# Uncomment the function you want to run:

# manual_srs_input_batch_run()

main_batch_run_from_db(srs_ids=[19,20], temperature=0.5)

