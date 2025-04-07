import logging
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", mode="w"),
    ],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

