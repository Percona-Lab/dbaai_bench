"""Print every model id the configured DigitalOcean key can reach, one per line.

`dba.py --list-models` is the friendlier view - grouped, priced, chat models
only. This is the raw list, for grepping.
"""

from do_dba.inference.client import InferenceClient
from do_dba.inference.config import base_url, find_api_key, load_dotenv

load_dotenv()
client = InferenceClient(api_key=find_api_key(), base_url=base_url(), label="DigitalOcean")
for record in client.list_models():
    print(record["id"])
