import os
from dotenv import load_dotenv

load_dotenv()

DEV_GEMINI_API_KEY = os.getenv("DEV_GEMINI_API_KEY", "")
DEV_MODEL          = "gemini-2.5-flash"

ENABLE_HYDE    = os.getenv("ENABLE_HYDE", "false").lower() == "true"
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"