import sqlite3

def recover_uml_image(db_path: str, record_id: int):
    """
    Recovers a UML image stored as BLOB from the 'usage_logs' table,
    using the specified record_id, and writes it out to a PNG file
    named 'recovered_uml_<id>.png'.
    """
    # Connect to or create a DuckDB database file
    with sqlite3.connect(db_path) as db_conn:
        # Use parameterized query to avoid SQL injection
        blob_row = db_conn.execute(
            "SELECT uml_image_data_ori_srs FROM usage_logs WHERE id = ?",
            (record_id,)  # Parameter tuple
        ).fetchone()

        if not blob_row:
            print(f"\n\n### ERROR ===========> No record found for id = {record_id}\n\n")
            exit()

        image_blob = blob_row[0]  # This will be a bytes object

        if image_blob is None:
            print(f"\n\n### ERROR ===========> Record found for id = {record_id}, but uml_image_data_ori_srs is NULL.\n\n")
            exit()

        # Write the recovered bytes out to a PNG file
        output_filename = f"recovered_uml_{record_id}.png"
        with open(output_filename, 'wb') as f:
                f.write(image_blob)
        
        print(f"UML image for id = {record_id} written to {output_filename}")



# Call the function with the database file path and record ID
recover_uml_image("nl2sm-app-prototype/nl2sm_oltp.sqlite3", 2)