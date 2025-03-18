# db_utils.py

import sqlite3
import os

# Path to your SQLite DB file
DB_PATH = "nl2sm-app-prototype/nl2sm_oltp.sqlite3"

def initialize_db():
    """Ensure the database file and 'usage_logs' table exist."""
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
                distilled_srs_text              TEXT,
                similarity_score_orig_srs       REAL,
                uml_image_data_orig_srs         BLOB,
                plantuml_script_orig_srs        TEXT,
                extracted_uml_descr_orig_srs    TEXT,
                similarity_descr_orig_srs       TEXT,
                similarity_score_dist_srs       REAL,
                uml_image_data_dist_srs         BLOB,
                plantuml_script_dist_srs        TEXT,
                extracted_uml_descr_dist_srs    TEXT,
                similarity_descr_dist_srs       TEXT,
                created_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create a new table to store SRS data.
        db_curs.execute("""
            CREATE TABLE IF NOT EXISTS srs_data (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                srs_origin  TEXT,
                srs_title   TEXT,
                srs_content TEXT
            )
        """)


def store_usage_log(
    event_type: str,
    llm_provider: str = None,
    llm_model: str = None,
    temperature: float = None,
    srs_id: str = None,
    original_srs_text: str = None,
    distilled_srs_text: str = None,
    plantuml_script_orig_srs: str = None,
    plantuml_script_dist_srs: str = None,
    uml_image_data_orig_srs: bytes = None,
    uml_image_data_dist_srs: bytes = None,
    extracted_uml_descr_orig_srs: str = None,
    extracted_uml_descr_dist_srs: str = None,
    similarity_descr_orig_srs: str = None,
    similarity_score_orig_srs: float = None,
    similarity_descr_dist_srs: str = None,
    similarity_score_dist_srs: float = None
):
    """
    Insert a row into the usage_logs table. 
    """
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()
        db_curs.execute("""
            INSERT INTO usage_logs (
                event_type,
                llm_provider,
                llm_model,
                temperature,
                srs_id,
                original_srs_text,
                distilled_srs_text,
                similarity_score_orig_srs,
                uml_image_data_orig_srs,
                plantuml_script_orig_srs,
                extracted_uml_descr_orig_srs,
                similarity_descr_orig_srs,
                similarity_score_dist_srs,
                uml_image_data_dist_srs,
                plantuml_script_dist_srs,
                extracted_uml_descr_dist_srs,
                similarity_descr_dist_srs
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            event_type,
            llm_provider,
            llm_model,
            temperature,
            srs_id,
            original_srs_text,
            distilled_srs_text,
            similarity_score_orig_srs,
            uml_image_data_orig_srs,
            plantuml_script_orig_srs,
            extracted_uml_descr_orig_srs,
            similarity_descr_orig_srs,
            similarity_score_dist_srs,
            uml_image_data_dist_srs,
            plantuml_script_dist_srs,
            extracted_uml_descr_dist_srs,
            similarity_descr_dist_srs
        ])


def retrieve_usage_logs(srs_id: str, provider: str, model: str, temperature: float):
    """
    Using the SRS_ID, Provider, Model, and Temperature, retrieve from the latest log entry:
      - Original SRS text
      - Distilled SRS text
      - PlantUML script for original SRS
      - PlantUML script for distilled SRS
      - Similarity analysis for original SRS
      - Similarity analysis for distilled SRS
    Also returns the total number of matching records as an integer.
    Returns a tuple of seven elements. If no matching record is found, returns (None, None, None, None, None, None, 0).
    """
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()
        
        # First, count matching records.
        count_query = """
            SELECT COUNT(*) FROM usage_logs
            WHERE srs_id = ? AND llm_provider = ? AND llm_model = ? AND temperature = ?
        """
        db_curs.execute(count_query, (srs_id, provider, model, temperature))
        count_result = db_curs.fetchone()
        record_count = count_result[0] if count_result else 0

        # Then, retrieve the latest record.
        select_query = """
            SELECT 
                original_srs_text,
                distilled_srs_text,
                plantuml_script_orig_srs,
                plantuml_script_dist_srs,
                similarity_descr_orig_srs,
                similarity_descr_dist_srs
            FROM usage_logs
            WHERE srs_id = ? AND llm_provider = ? AND llm_model = ? AND temperature = ?
            ORDER BY created_at DESC
            LIMIT 1
        """
        db_curs.execute(select_query, (srs_id, provider, model, temperature))
        row = db_curs.fetchone()
        if row is None:
            return (None, None, None, None, None, None, 0)
        # Ensure the returned tuple is 7 elements by appending record_count.
        return (*row, record_count)


# (A) Helpers for storing & retrieving SRS
def insert_srs_data(srs_origin: str, srs_title: str, srs_content: str):
    """
    Insert a single SRS record into the srs_data table, 
    but only if the same row (same origin, title, and content) doesn't already exist.
    """
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()

        # 1) Check if an identical row already exists
        db_curs.execute("""
            SELECT COUNT(*)
            FROM srs_data
            WHERE srs_origin = ?
              AND srs_title  = ?
              AND srs_content = ?
        """, (srs_origin, srs_title, srs_content))
        (count,) = db_curs.fetchone()

        if count > 0:
            print(f"Skipped insertion because an identical SRS row already exists: "
                  f"origin='{srs_origin}', title='{srs_title}'")
            return

        # 2) If not present, insert the new row
        db_curs.execute("""
            INSERT INTO srs_data (srs_origin, srs_title, srs_content)
            VALUES (?, ?, ?)
        """, (srs_origin, srs_title, srs_content))
        db_conn.commit()
        print(f"Inserted row with origin='{srs_origin}', title='{srs_title}' into srs_data.")


def select_all_srs():
    """
    Return a list of tuples: (id, srs_title, srs_content)
    """
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()
        db_curs.execute("SELECT id, srs_origin, srs_title, srs_content FROM srs_data")
        rows = db_curs.fetchall()
    return rows


def select_srs_by_ids(srs_ids: list[int]):
    """
    Returns rows from srs_data that match the given list of srs_ids.
    Each row is a tuple: (id, srs_title, srs_content, srs_origin)
    """
    if not srs_ids:
        return []  # empty list if no IDs

    placeholders = ",".join("?" for _ in srs_ids)
    query = f"SELECT id, srs_title, srs_content FROM srs_data WHERE id IN ({placeholders})"
    with sqlite3.connect(DB_PATH) as db_conn:
        db_curs = db_conn.cursor()
        db_curs.execute(query, srs_ids)
        return db_curs.fetchall()


def import_srs_descriptions(srs_dict: dict):
    """
    1) Accepts a dictionary of SRS entries, each containing 'title', 'content' and 'origin'.
    2) Calls insert_srs_data(...) for each entry, storing them in the srs_data table.
    """
    # Ensure DB and tables exist
    initialize_db()

    # Insert each SRS into srs_data
    for title, srs_info in srs_dict.items():
        content = srs_info["content"]
        origin = srs_info["origin"]
        insert_srs_data(origin, title, content)

manual_input_srs_dict = {
        # Title": {
        #     "content": """
        #          SRS content goes here...
        #     """,
        #     "origin": "Bibtex Key or URL"
        # },
        #### [11] CamaraEtAl2023AssessmentGenerativeAI
        "The Banks": {
            "content": """
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
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Book management": {
            "content": """
            We want to model a computerized system for book management, whose structure is described below. 

            * Each book has a title and one or more authors (e.g., Don Quixote, Miguel de Cervantes). 
            * In addition to the authors' names, their date of birth and death (if they are deceased) are also stored. 
            * Different editions of each book may exist, including the original one. 
            * An edition may be of one type (paperback, hardcover or deluxe), has a number of pages and is published in one year, by one publisher and in one language. 
            * Each edition may have a set of illustrators, who are in charge of the drawings of that edition (if any), as well as a set of translators, if the book has been translated from its original language. 
            * Publishers print a number of copies of each edition. Each copy has an owner, which can be the publisher itself (if no one has bought it yet -- this is the default option), a bookstore, a library, or an individual. 

            The behavior of the system allows books to be bought and borrowed. 
            * Libraries can lend books to their registered users or to other libraries. 
            * Individuals can only buy books through bookstores or from other individuals. 
            * However, libraries and stores can buy them from publishers directly, from other stores or from individuals. 
            * Publishers cannot buy books. 
            * Finally, individuals may sell books that they own, but not books that they have borrowed from a library. 
            * Libraries cannot sell books, only lend them, only if they own them and they are not borrowed.  
            * Stores may not borrow books from any library.  
            * Likewise, books may not be returned to a library that have not been previously borrowed. 
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Distribution companies": {
            "content": """
            We want to model the structure and behavior of a set of distribution companies. 

            * Each company has a number of employees, including a director, a manager, and at least one base worker. 
            * All employees have a salary. Of course, a person can work in several companies (though never in more than three), and therefore have different salaries. 
            * Within the same company, the director has to be paid more than the manager, and the manager more than the base workers. 
            * In addition, a person cannot hold two positions in the same company, i.e., a director cannot be a manager or a base worker, a manager cannot be a director or a base worker, and base workers cannot be managers or directors of that company (although they can be in other companies). 
            * Each company sells a series of products (e.g., screws, long bits, short bits, nails, hammers, etc.), each with its own price. Of course, the price of items of the same product must be the same within the same company, but it may vary between different companies selling the same product. 
            * People who place orders with a company become its customers. 
            * Each order includes a number of items of the products sold by the company (for example, an order may be for 10 nails and 2 hammers, or for three chairs and two tables).  
            * Each order must exceed a minimum value, otherwise it is not profitable for the company. Each company defines the minimum value of its orders. 
            * Each company has two types of customers: normal customers, which are those who have placed at least one order, and VIP customers, which are those who have placed orders totaling more than 1,000 Euros. As soon as a customer becomes a VIP, he gets a 10% discount on all new orders. Let us further assume that the employees of a company are considered VIPs as soon as they start working for the company. 
            * Companies have a set of items in stock in their warehouse at any given time. No one can place an order for items that are not in the company's warehouse. 
            * Once an order is placed, the item disappears from the company's warehouse and becomes the property of the person who placed the order.
            * Finally, the same person cannot have items of more than 10 different product types, regardless of the company where they were purchased. 

            The behavior of the system is determined by a set of actions that allow: 

            * Purchasing items by placing orders, 
            * Replenishing the companies' warehouses, and
            * Hiring and firing companies' workers.    
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Movie Theaters": {
            "content": """
            We want to model a system that represents chains of movie theaters in a given country, with the following restrictions. 

            * Each chain owns several movie theaters in different cities of the country. 
            * The laws of that country prevent more than two movie theaters of the same chain in the same city. 
            * Each theater has a number of rooms where movies are shown. 
            * Each film has a duration and a director, and the chains have to pay a fee to the producer of the film each time they show it. 
            * Each room has a maximum number of seats to accommodate the audience. 
            * Each room may screen a maximum of five movies on the same day, whether they are the same film or not. 
            * At least 20 minutes must elapse between the end of one screening and the beginning of the next in the same room. 
            * Another rule in the country prevents films by a single director from being shown in the same cinema on the same day. 
            * Customers buy their tickets centrally at each cinema chain, indicating the movie they want to see, the cinema of the chain where they want to see it, the day, the room, the start time, and the number of tickets they want. 
            * Ticket prices for each movie are set by the chain. The price of a movie is the same for all theaters of the same chain. 
            * Customers cannot buy more tickets than are currently available in the desired room. 
            * A customer who has purchased a series of tickets may also return them if he/she does so before the movie starts. 
            * Each chain has a loyalty card, and offers discounts at its theaters for customers who have it. Discounts are decided by each chain, but cannot exceed 50% of the ticket value, nor be less than 20%.         
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Online shopping": {
            "content": """
            The company nozamA sells different products to its customers. 
            To purchase a product, the customer indicates the company how many units of the product he wants to buy. 
            The company generates an order if there are enough units of that product in stock. 
            The customer, after receiving the Order, passes it to his favorite transport company to send the items to his postal address. 
            As a result of the shipment, a delivery note is generated, which is associated with the order, and which includes the total amount to be paid by the customer.  
            After shipment, the corresponding items become the property of the customer.    
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Processes, threads and resources": {
            "content": """
            In a concurrent system there may be multiple processes running concurrently. We want to develop a UML of the system, according to the following requirements:

            * Each process is composed of one or more threads. Both processes and threads can access shared resources, which can be of two types: global, which are owned by the system and can be shared by any thread of any process, and local, which are owned by a particular process and can only be shared by threads of that process.

            * The system is limited by the number of processes, threads and resources it can manage: The maximum number of processes is the system is 128, and each process can have a maximum of 32 threads. However, the total number of threads in the system can never exceed 64, and the total number of resources (sum of local and global) cannot exceed 128.

            * If a thread of an active process wants to acquire a shared resource that is free, it acquires it directly. The acquisition implies that the shared resource becomes occupied and that its owner at that moment is the thread that has acquired it. However, if a thread tries to acquire a resource that is being used by another thread, the requesting thread is blocked waiting for its turn. Each resource maintains a queue of waiting threads, so that if a thread requests an occupied resource the thread is inserted into the queue. A thread can use several resources at the same time, but if it is blocked when trying to acquire one it cannot request others and therefore can only be blocked by one resource at most.

            * The thread that owns the shared resource can release it when it has finished using it. When a thread releases a resource, the first of the threads in the queue for that resource gets to use it. In case the queue is empty the resource will be unoccupied.

            * Processes can be in three different states: running, sleeping or available. In the system there is always only one process running and the rest are asleep or available. None of the threads of the running process or the available processes can be locked on a global resource. On the other hand, all sleeping processes must have some thread locked on some resource.
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "The Templeman Library at the University of Kent": {
            "content": """
            The University of Kent Library has the following rules to regulate the processes of borrowing items from the library.

            * Only faculty members (i.e., professors) and undergraduate and graduate students may borrow items.
            * Items that can be borrowed are books and journals.
            * The Chief Librarian may temporarily suspend the loan of any item, or cancel them when they are no longer in loanable condition.
            * Loans must be requested from one of the librarians. 
            * There are limits on the number of items each type of user may have on loan at any given time, and on the duration each item may be borrowed. The Chief Librarian is responsible for setting these limits. By default, these are as follows:
                * Undergraduate students may have up to 3 books on loan at any one time. They may not borrow journals. Books may be borrowed for up to 7 days.
                * Graduate students may borrow up to 8 items (books or journals) simultaneously. They may have the journals for 3 days and the books for two weeks. 
                * Professors may borrow up to 16 items (books or journals) simultaneously. They may have the journals for 2 weeks and the books for 8 weeks. 
            * Borrowed items must be returned before their deadlines. Users who fail to return an item on time will be charged a daily fine according to the established rates. 
            * Borrowers must return items to librarians on duty, along with the corresponding fine in case of late return of the loan.
            * Any librarian on duty can withdraw borrowing rights from any user if the user does not pay the fine in full at the time of returning the loan.    
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Theater plays": {
            "content": """
            We want to model a system with theaters that represent plays.

            * In a city there is a group of theaters, which throughout the season represent different plays. 
            * Each play has a title, an author and a cast, which is nothing more than the set of characters that appear in it. 
            * Each play is performed several times during a season, on specific days and in afternoon and evening sessions. 
            * Actors are people who play the characters of a play in a specific performance. 
            * In the same performance, a character can only be played by one actor, although the same actor can play several characters. 
            * An actor can act in different performances on the same day (of the same or different plays), as long as they do not coincide in the same session (afternoon or evening). 
            """,    
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Underground system": {
            "content": """
            We want to model an underground system, composed of several elements: stations, lines, tracks, track sections, and trains. 

            * An underground network is composed of lines, which are sequences of stations connected by tracks. In a line, each track is connected to the next and previous tracks of the sequence, except for the first and last, which are connected to the origin and destination stations of the line (which may be the same), respectively. 
            * Stations represent points in the network where trains regularly stop so that passengers can get on or off. 
            * Several trains can be in the same station at the same time. 
            * Each track connects two consecutive stations and has two sections, one for each directions of travel. 
            * A station may belong to more than one line, and therefore be connected to more than two tracks.
            * Trains are the objects that move through the network. At any given time, each train must be located either at a station or on a section of a track. 
            * For safety, at any given time, there must be at most one train on each section of track. 
            * Each train services one line only, and therefore can only move through the tracks of that line.
            * A train can be moving, if it is on a section of track, or stopped at a station.
            * Trains go from the initial station of the line to the final station, in one direction, and then return, in the opposite direction, unless the line is circular, i.e., the initial and final stations are the same. In this case, trains always follow the same direction. 
            * All lines have trains moving in both directions.
            * All the main elements of a railway system have a unique name (a string of characters).     
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "University Degrees": {
            "content": """
            We want to model the educational system of universities offering different undergraduate and graduate degrees (masters), according to the following specifications. 

            *	Each degree is composed of a set of subjects, each with several ECTS credits, which are taught every academic year. 
            *	The same subject can only be taught once each academic year. 
            *	Each subject only belongs to one University, although it can be part of several of its degrees (both bachelor's and master's).
            *	Each subject is taught by a single lecturer, which may vary from year to year. 
            *	Each lecturer may not teach more than 4 subjects each year (regardless of their number of credits).
            *	Each year the lecturers evaluate the students enrolled in the subjects they teach, assigning them a grade between 0 and 10 (we assume that there are no "no-shows", but that all enrolled students obtain a grade between 0 and 10). A grade of 5 or higher is considered a passing grade. 
            *	The same student can only register up to 3 times for the same subject, in different years. 
            *	A student may only pass a subject once and cannot register for subjects already passed. 
            *	For the sake of simplicity, we will assume that each subject has only one exam per academic year. 
            *	All subjects must be evaluated before the end of the academic year they are taught. 
            *	Teachers cannot enroll in the subjects they teach in that course, although they can enroll in other subjects. 
            *	To obtain a diploma at a university, a student needs to pass at least 240 credits of the subjects in case of undergraduate degrees or 60 for graduate degrees. 
            *	The diploma includes the name of the university and the name of the degree, as well as the year in which the student passed the last subject that completed the necessary credits. 
            *	A person can obtain as many diplomas as degrees a university offers if he or she completes the corresponding courses. 
            *	Finally, to be able to enroll in a subject that is taught only at the postgraduate level, the student must hold an undergraduate diploma (from that university or from another one).    
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        "Video Stores": {
            "content": """
            We want to model a system composed of video stores, which are companies that have shops dedicated to renting movies. 

            * Each video store has a name and a set of associated shops, each with a different address. 
            * Customers of a video store can rent movies at any of its shops. 
            * Each shop has a set of copies of each movie for rent, and a clerk who attends the store. 
            * Customers ask the clerks for the movie they wish to rent and, if a copy of the movie is available at that shop, they are granted a 3 or 5 day loan depending on whether the customer is a regular or VIP customer. 
            * The loans must be returned to the same shop where they were rented. 
            * Regular customers can have a maximum of three active loans from the same video store, regardless of the shop they where rented from. VIP customers have no such limit.  
            * No clerk may be a customer of the shop where he/she works. 
            * If a customer returns a copy of a movie late, he/she is blacklisted in that video store and cannot rent any more movies in any of its shops.    
            """,
            "origin": "CamaraEtAl2023AssessmentGenerativeAI"
        },
        
        #### [20] DeBariEtAl2024EvaluatingLargeLanguage
        "Project Management System": {
            "content": """
            A project manager uses the project management system to manage a project. 
            The project manager leads a team to execute the project within the project's start and end dates. 
            Once a project is created in the project management system, a manager may initiate and later terminate the project due to its completion or for some other reason. 
            As input, a project uses requirements. 
            As output, a project produces a system (or part of a system). 
            The requirements and system are work products: things that are created, used, updated, and elaborated on throughout a project. 
            Every work product has a description, is of some percent complete throughout the effort, and may be validated. 
            However, validation is dependent on the type of work product. 
            For example, the requirements are validated with users in workshops, and the system is validated by being tested against the requirements. 
            Furthermore, requirements may be published using various types of media, including on an intranet or in paper form; and systems may be deployed onto specific platforms. 
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Hollywood Approach": {
            "content": """
            We are interested in building a software application to manage filmed scenes for realizing a movie, by following the so-called “Hollywood Approach”. 
            Every scene is identified by a code (a string) and it is described by a text in natural language. 
            Every scene is filmed from different positions (at least one), each of this is called a setup. 
            Every setup is characterized by a code (a string) and a text in natural language where the photographic parameters are noted (e.g., aperture, exposure, focal length, filters, etc.). 
            Note that a setup is related to a single scene. 
            For every setup, several takes may be filmed (at least one). 
            Every take is characterized by a (positive) natural number, a real number representing the number of meters of film that have been used for shooting the take, and the code (a string) of the reel where the film is stored. 
            Note that a take is associated to a single setup. 
            Scenes are divided into internals that are filmed in a theater, and externals that are filmed in a location and can either be “day scene” or “night scene”. 
            Locations are characterized by a code (a string) and the address of the location, and a text describing them in natural language.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Word Processor": {
            "content": """
            A user can open a new or existing document. Text is entered through a keyboard. 
            A document is made up of several pages and each page is made up of a header, body and footer. 
            Date, time and page number may be added to header or footer. 
            Document body is made up of sentences, which are themselves made up of words and punctuation characters. 
            Words are made up of letters, digits and/or special characters. 
            Pictures and tables may be inserted into the document body. 
            Tables are made up of rows and columns and every cell in a table can contain both text and pictures. 
            Users can save or print documents.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Patient Record and Scheduling System": {
            "content": """
            A patient record and scheduling system in a doctor’s office is used by the receptionists, nurses, and doctors. 
            The receptionists use the system to enter new patient information when first-time patients visit the doctor.
            They also schedule all appointments. 
            The nurses use the system to keep track of the results of each visit including diagnosis and medications. 
            For each visit, free form text fields are used captures information on diagnosis and treatment. 
            Multiple medications may be prescribed during each visit. 
            The nurses can also access the information to print out a history of patient visits. 
            The doctors primarily use the system to view patient history. 
            The doctors may enter some patient treatment information and prescriptions occasionally, but most frequently they let the nurses enter this information. 
            Each patient is assigned to a family. 
            The head of family is responsible for the person with the primary medical coverage. 
            Information about doctors is maintained since a family has a primary care physician, but different doctors may be the ones seeing the patient during the visit.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Movie-Shop": {
            "content": """
            Design a system for a movie-shop, in order to handle ordering of movies and browsing of the catalogue of the store, and user subscriptions with rechargeable cards. 
            Only subscribers are allowed hiring movies with their own card. 
            Credit is updated on the card during rent operations. 
            Both users and subscribers can buy a movie and their data are saved in the related order. 
            When a movie is not available it is ordered .    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Flights": {
            "content": """
            We want to model a system for management of flights and pilots. An airline operates flights. 
            Each airline has an ID. 
            Each flight has an ID a departure airport and an arrival airport: an airport as a unique identifier. 
            Each flight has a pilot and a co-pilot, and it uses an aircraft of a certain type; a flight has also a departure time and an arrival time. 
            An airline owns a set of aircrafts of different types. 
            An aircraft can be in a working state or it can be under repair. 
            In a particular moment an aircraft can be landed or airborne. 
            A company has a set of pilots: each pilot has an experience level: 1 is minimum, 3 is maximum. 
            A type of aeroplane may need a particular number of pilots, with a different role (e.g.: captain, co-pilot, navigator): there must be at least one captain and one co-pilot, and a captain must have a level 3.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Bank System": {
            "content": """
            A bank system contains data on customers (identified by name and address) and their accounts. 
            Each account has a balance and there are 2 type of accounts: one for savings which offers an interest rate, the other for investments, used to buy stocks. 
            Stocks are bought at a certain quantity for a certain price (ticker) and the bank applies commission on stock orders.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Veterinary Clinic": {
            "content": """
            The owner of a veterinary clinic wants to create a database to store information about all veterinary services performed. 
            After some research he came up with the following requirements:
                - For each admitted animal, its name, breed (if any) and owner must be stored. Each animal should be given an unique numeric identifier.
                - For each owner, its name, address and phone number should be stored. An unique numeric identifier should also be generated for each one of them.
                - An animal might be owner-less. This happens frequently as the clinic often rescues abandoned dogs from the streets in order to treat them and get them new owners.
                - It should be possible to store information about a specific breed even if no animals of that breed have been treated at the clinic.
                - Each appointement always has a responsible physician. All appointements start at a certain date and time; and are attended by an animal (and of course its owner).
                - For each physician, his name, address and phone number should be stored. An unique numeric identifier should also be generated for each one of them.
                - In an appointement, several medical conditions might be detected. Each condition has a common name and a scientific name. No two conditions have the same scientific name.
                - It should be possible to store information about the most common conditions for each different breed in the database.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Auto Repair": {
            "content": """
            An auto repair shop, that sells and mounts parts and accessories for all kinds of vehicles, wants a new information system to manage their clients, parts, accessories and assembly services:  
            - There are several employees. Each one of them has an unique identifying number, a name and an address.
            - In this shop, assembly services, where parts and accessories are installed in a vehicle, are executed. For each one these services the following data must be stored: In which car the service was executed, how many kms had the car at the time, who was the responsible employee, which parts and accessories were fitted, how many work hours did it take and the admission and finish dates.
            - Parts and accessories are only sold together with an assembly service.
            - Each part/accessory only fits in some car models. Therefore, it is important to store that information.
            - Each part/accessory has a category (radio, tyre,…), a serial number and a price.
            - Each car has a license plate, a make, a model, a color and an owner. Each owner has a name, identifying number, address and a phone.
            - One person can own more than one car but one car only has one owner.

            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Restaurant": {
            "content": """
            The owner of a small restaurant wants a new information system to store data for all meals consumed there and also to keep a record of ingredients kept in stock. 
            After some research he reached the following requirements list:
                - Each ingredient has a name, a measuring unit (e.g. olive oil is measured in liters, while eggs are unit based) and a quantity in stock. There are no two ingredients with the same name.
                - Each dish is composed of several ingredients in a certain quantity. An ingredient can, of course, be used in different dishes.
                - A dish has an unique name and a numeric identifier.
                - There are several tables at the restaurant. Each one of them has an unique numeric identifier and a maximum ammount of people that can be seated there.
                - In each meal, several dishes are consumed at a certain table. The same dish can be eaten more than once in the same meal.
                - A meal takes place in a certain date and has a start and end time. Each meal has a responsible waiter.
                - A waiter has an unique numerical identifier, a name, an address and a phone number.
                - In some cases it is important to store information about the client that consumed the meal. A client has a tax identification number, a name and an address.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Deliveries": {
            "content": """
            The owner of a small delivery company plans to have an information system that allows him to save data about his customers and deliveries. 
            After some time studying the problem, he reached the following requirements:
                - Each customer has a VAT number, a name, a phone number and an address. There are no two clients with the same VAT number.
                - When a customer wants to send a package to another customer, he just has to login to the company website, select the customer he wants to send the package to, enter the package's weight and if the delivery is normal or urgent. He then receives an unique identifier code that he writes on the package.
                - The package is then delivered by the customer at the delivery center of his choosing. A delivery center has a unique name and an address.
                - Each client has an associated delivery center. This delivery center is chosen by the company and it is normally the one closest to the customer's house.
                - The package is them routed through an internal system until it reaches the delivery center of the recipient.
                - The package is then delivered by hand from that delivery center to the recipient by a courier.
                - Couriers have a single VAT number, a name and a phone number. Each courier works in a single delivery center.
                - A courier is assigned to a packet as soon as the packet is introduced in the system.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Furniture": {
            "content": """
            The known furniture factory Hi-Key-Ah, intends to implement an information system to store all data on the different types of furniture and components it produces:  
                - The factory produces several lines of furniture, each with a different name and consisting of several pieces of furniture of different types (beds, tables, chairs,…).
                - All furniture pieces have a type, a single reference (eg CC6578) and a selling price.
                - The major competitive advantage of this innovative plant is the fact that each component produced can be used in more than one piece of furniture.
                - Each piece of furniture is thus composed of several components. The same component can be used more than once in the same piece.
                - Every type of component produced is assigned a unique numerical code, a manufacturing price and a type (screw, hinge, shelf…).
                - The furniture is then sold in various stores throughout the world. Each store has a different address and a fax number.
                - To make the manufacturing process more efficient, stores have to place orders everytime they need to replenish their stock. These orders must also be stored in the database.
                - Each order has a order number, a date, the store that placed the order as well as a list of all the ordered furniture and their quantities.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Factory": {
            "content": """
            Create a database for a factory with the following requirements. Don't forget to add unique identifiers for each one of the entities if needed.   
                - A factory has several machines. Each one of them is operated by several workers. 
                - A worker might work in more than one machine.
                - In this factory, several products of different types, are produced. Each different type of product is produced in a single machine. But, the same machine can produce more than one type of product.
                - Products from the same type are all produced from the same single material and have the same weigth.
                - Clients can issue purchase orders. Each order has a list of the desired products and their quantity.
                - For each worker, the following data should be stored in the database: name (first and last), birth date, address and a list of his skills.
                - For each machine, the following data should be stored: serial number, make, model and purchase date.
                - For each client, the followig data should be stored: name, address, phone number and name of the contact person (if any).
                - For each purchase order, the following date should be stored: order number, date it has been made, expected and actual delivery date.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Bycicle Rental": {
            "content": """
            A bicycle renting company wants to create an information system that allows it to store the data regarding all their reservations and rentals. 
            The system should follow these requirements:
                - It should be possible to store the national id number (NIN), tax identification number (TIN), name and address for every client. The NIN and TIN must be different for every client and all clients should have at least a TIN and a name.
                - The database should also contain information about the bicycle models that can be rented- Each model has an unique name, a type (that can only be road, mountain, bmx or hybrid) and the number of gears.
                - Each bicycle has a unique identifying number and a model.
                - The company has several different stores where bicycles can be picked up and returned. Each one of these stores is identified by an unique name and has an address (both mandatory).
                - When a reservation is made, the following data must be known: which client did the reservation, when will he pick up the bike (day), which bike model he wants and where will he pick up the bike (store). 
                - When a bike is picked up, the actual bike that was picked up must be stored in the database.
                - When a bike is returned, the return date should be stored in the database.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Saturn Int. Management": {
            "content": """
            Saturn Int. management wants to improve their security measures, both for their building and on site. 
            They would like to prevent people who are not part of the company to use their car park.
            Saturn Int. has decided to issue identity cards to all employees. 
            Each card records the name, department and number of a company staff, and give them access to the company car park.
            Employees are asked to wear the cards while on the site.
            There is a barrier and a card reader placed at the entrance to the car park. 
            When a driver drives his car into the car park, he/she inserts his or her identity card into the card reader. 
            The card reader then verify the card number to see if it is known to the system. 
            If the number is recognized, the reader sends a signal to trigger the barrier to rise. 
            The driver can then drive his/her car into the car park.
            There is another barrier at the exit of the car park, which is automatically raised when a car wishes to leave the car park.
            A sign at the entrance display “Full” when there are no spaces in the car park. 
            It is only switched off when a car leaves.
            There is another type of card for guests, which also permits access to the car park. 
            The card records a number and the current date. 
            Such cards may be sent out in advance, or collected from reception. 
            All guest cards must be returned to reception when the visitor leaves Saturn Int.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "OOBank": {
            "content": """
            This system provides the basic services to manage bank accounts at a bank called OOBank.
            OOBank has many branches, each of which has an address and branch number. 
            A client opens accounts at a branch. Each account is uniquely identified by an account number; it has a balance and a credit or overdraft limit. 
            There are many types of accounts, including: a mortgage account (which has a property as collateral), a checking account, and a credit card account (which has an expiry date and can have secondary cards attached to it). 
            It is possible to have a joint account (e.g. for a husband and wife). 
            Each type of account has a particular interest rate, a monthly fee and a specific set of privileges (e.g. ability to write checks, insurance for purchases etc.). 
            OOBank is divided into divisions and subdivisions (such as Planning, Investments and Consumer); the branches are considered subdivisions of the Consumer Division. 
            Each division has a manager and a set of other employees. Each customer is assigned a particular employee as his or her 'personal banker'.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Prepaid Cell Phone": {
            "content": """
            The contract of a prepaid cell phone should be modelled and implemented. 
            A basic contract has a contract number (of type int) and a balance (of type double), but no monthly charges. 
            The contract number is not automatically generated, but is to be set as a parameter by the constructor as well as the initial balance. 
            The balance has a getter and a setter. 
            The following options can be added to a contract (if needed also several times):  
                - 100 MB of data (monthly charge 1.00€)
                - 50 SMS (monthly charge 0.50€)
                - 50 minutes (monthly charge 1.50€)
                - Double Transfer Rate (monthly charge 2.00€) implement this requirement with the help of the decorator pattern. All contract elements should be able to understand the methods getCharges():double, getBalance():double and setBalance(double).  
            
            The method getCharges() should provide the monthly charge of a contract with all its options selected. 
            The methods getBalance() and setBalance() should be passed through and access the basic contract.
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Library System": {
            "content": """
            The exercise is to design a class structure for a library system. 
            It should fulfil those requirements:
                - There are two type of users - under-aged and adults.
                - Under-aged users are identified with usage of their full name and student card.
                - Adult users are identified with usage of their full name and ID card.
                - The library contains books.
                - There is basic information about every book (title, author, etc).
                - The user can borrow at most 4 books at the same time.
                - There is a history of previously borrowed books for every user (along with all the dates).    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "MyDoctor": {
            "content": """
            The MyDoctor application aims to be a management tool for the appointments of a doctor.
            A hospital has multiple offices. 
            The users of the application can be doctors and patients. 
            The doctors can apply to practice in offices and create a schedule for an office. 
            The schedules in different offices can't overlay.
            
            Example:
            Doctor Ana is available in Office 4 on the 4th of September during 1 PM - 5PM.
            Doctor Ana can't practice in Office 5 on the 4th of September during 3PM - 8 PM, but she can
            practice in Office 5 on the 4th of September during 5:30PM - 8 PM.    

            The patients can see the existing doctors in the system, the schedule of the offices and can book appointments for specific doctors and for specific schedules. 
            The appointments can be of 3 types:
                - Blood Test - 15 mins
                - Consultation - 30 mins
                - Surgery - 60 mins

            The booking of an appointment will not be possible if another appointment is already booked at the same time frame. 
            An email is sent to the patient with the confirmation of the appointment.

            Example:
            Action 1: User Mike will create a blood test booking for Doctor Ana for the 4th of September starting with 15:30 PM → Possible
            Action 2: User Mike will create an intervention booking for Doctor Ana for the 4th of September starting with 15:00 PM → Not Possible
            Action 3: User Mike will create an intervention booking for Doctor Ana for the 4th of September starting with 16:00 PM → Possible

            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },
        "Online Shopping": {
            "content": """
            Each customer has unique id and is linked to exactly one account. 
            Account owns shopping cart and orders. 
            Customer could register as a web user to be able to buy items online. 
            Customer is not required to be a web user because purchases could also be made by phone or by ordering from catalogues. 
            Web user has login name which also serves as unique id. 
            Web user could be in several states - new, active, temporary blocked, or banned, and be linked to a shopping cart. 
            Shopping cart belongs to account.
            Account owns customer orders. 
            Customer may have no orders. 
            Customer orders are sorted and unique. 
            Each order could refer to several payments, possibly none. 
            Every payment has unique id and is related to exactly one account.
            Each order has current order status. 
            Both order and shopping cart have line items linked to a specific product. 
            Each line item is related to exactly one product. 
            A product could be associated to many line items or no item at all.    
            """,
            "origin": "DeBariEtAl2024EvaluatingLargeLanguage"
        },

        #### [1] WangEtAl2024HowLLMsAid
        "Order Processing System": {
            "content": """
            Consider the following problem description: A mail-order company wants to automate its order processing. 
            The initial version of the order processing system should be accessible to customers via the web. 
            Customers can also call the company by phone and interact with the system via a customer representative. 
            It is highly likely that the company will enhance this system in upcoming years with new features. 
            The system allows customers to place orders, check the status of their orders, cancel an existing order and request a catalog. 
            Customers may also return a product but this is only possible through the phone, not available on the web. 
            When placing an order, the customer identifies himself by means of customer number (only for existing registered customers) or by means of his name and address. 
            He then selects a number of products by giving the product number or by selecting products from the online catalogue. 
            For each product, information such as price, a description and a picture (only on demand as they are usually high-resolution images of large size) are presented to the customer. 
            Also, the availability of the product is obtained from the inventory. 
            The customer indicates whether he wants to buy the product and in what quantity. 
            When all desired products have been selected, the customer provides a shipping address and a credit card number and a billing address (if different from the shipping address). 
            Then an overview of the ordered products and the total cost are presented. 
            If the customer approves, the order is submitted. 
            Credit card number, billing address and a specification of the cost of the order are used on the invoice, which is forwarded to the accounting system (an existing software module). 
            Orders are forwarded to the shipping company, where they are filled and shipped.
            Customers who spent over a certain amount within the past year are promoted to be gold customers. 
            Gold customers have additional rights such as being able to return products in an extended time period as well as earning more bonus points with each purchase. 
            In addition, in cases where a product is on back order, gold customers have the option to sign up for an email notification for when the particular product becomes available.

            Consider the following use case scenario (for use case “place order”):
            Ali is an existing customer of the order processing company described earlier, registered with their web site. 
            Also assume that having browsed the printed catalogue he has, he already identified the two items (including their prices) he likes to buy from the company's website using their product numbers (i.e. #2 and #9). 
            First, he tries to buy one of product #2, but it is listed as unavailable in the inventory. 
            Then, he adds two quantities of product #9, which turns out to be available, to his basket. 
            He is then asked to confirm his registered shipping and billing addresses and credit card information from the customer database. 
            He completes the order by clicking the Submit button. 
            You may ignore processing of customer authentication.
            """,
            "origin": "WangEtAl2024HowLLMsAid"
        }

        #### [22] FerrariEtAl2024ModelGenerationLLMs
        # "": {
        #     "content": """
        #     """,
        #     "origin": "FerrariEtAl2024ModelGenerationLLMs"
        # },
    }

# import_srs_descriptions(manual_input_srs_dict)