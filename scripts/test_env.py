from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

load_dotenv()
INCIDENT_DB=os.getenv('DB_CONNECTION')
print(INCIDENT_DB)