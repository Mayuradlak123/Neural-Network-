import os
from langchain_groq import ChatGroq
from config.logger import logger


def setup_groq():
    """
    Sets up and returns a configured ChatGroq LLM instance.
    Ensures environment variables are loaded and validated.
    """
    logger.info("Setting up Groq LLM...")

    # Load environment variables
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Validate environment variables
    if not GROQ_MODEL:
        logger.error("GROQ_MODEL not set in environment or .env file")
        raise ValueError("GROQ_MODEL not set in environment or .env file")

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in environment or .env file")
        raise ValueError("GROQ_API_KEY not set in environment or .env file")

    # Initialize the Groq LLM
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
        logger.info(f"ChatGroq model '{GROQ_MODEL}' initialized successfully.")
        return llm
    except Exception as e:
        logger.exception("Failed to initialize ChatGroq:")
        raise e


def process_prompt(query: str, llm: ChatGroq):
    """
    Processes a prompt using the provided ChatGroq LLM instance.
    """
    logger.info(f"Processing prompt: {query[:80]}...")  # limit length for readability
    try:
        message = llm.invoke(query)
        logger.info("Prompt processed successfully.")
        logger.debug(f"Full response: {message}")
        return message.content
    except Exception as e:
        logger.exception(f"Error while processing prompt: {query}")
        raise e


# Example usage (only if running directly)
if __name__ == "__main__":
    llm = setup_groq()
    response = process_prompt("Write a short greeting email.", llm)
    print("LLM Response:", response)
