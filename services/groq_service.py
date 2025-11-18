import os
import json
from langchain_groq import ChatGroq
from config.logger import logger
from typing import Dict, Any
from config.chroma import insert_vector

class GroqService:
    """
    Service to handle Groq LLM operations for job extraction and email generation
    """
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Groq LLM instance"""
        logger.info("Setting up Groq LLM...")
        
        GROQ_MODEL = os.getenv("GROQ_MODEL")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        if not GROQ_MODEL or not GROQ_API_KEY:
            logger.error("GROQ_MODEL or GROQ_API_KEY not set in environment")
            raise ValueError("GROQ_MODEL or GROQ_API_KEY not set in environment")
        
        try:
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            logger.info(f"ChatGroq model '{GROQ_MODEL}' initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize ChatGroq")
            raise e
    
    def extract_job_info(self, job_content: str) -> Dict[str, Any]:
        """
        Extract structured information from job description and return JSON.
        
        Args:
            job_content (str): The raw job description text
            
        Returns:
            Dict containing extracted job information
        """
        try:
            logger.info("Extracting job information using Groq...")
            
            prompt = f"""
You are a job description analyzer. Extract the following information from the job description and return ONLY a valid JSON object with no additional text, markdown, or formatting.

Job Description:
{job_content}

Extract and return a JSON object

Return ONLY the JSON object, nothing else of this job_title, company_name,experience, location, employment_type, experience_level, salary_range, job_description, key_responsibilities, required_skills, preferred_skills, education, benefits, application_link, posted_date, application_deadline, contact_email
field.
Like 
"""
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            logger.debug(f"Raw LLM response: {content}")
            
            # Clean up response - remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON
            job_info = json.loads(content)
            
            logger.info("Successfully extracted job information")
            return job_info
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response content: {content}")
            raise ValueError(f"Failed to parse job information: {str(e)}")
        except Exception as e:
            logger.exception("Error extracting job information")
            raise e
    
    def generate_cold_email(self, job_info: Dict[str, Any]) -> str:
        """
        Generate a cold email based on extracted job information.
        
        Args:
            job_info (Dict): Structured job information from extract_job_info
            
        Returns:
            str: Generated cold email content
        """
        try:
            logger.info("Generating cold email using Groq...")
            
            prompt = f"""
You are a professional email writer. Write a compelling cold email for the following job opportunity.

Job Information:
{json.dumps(job_info, indent=2)}

Write a professional cold email that:
1. Has an engaging subject line
2. Is concise (150-200 words)
3. Highlights relevant skills matching the job requirements
4. Shows enthusiasm for the role
5. Includes a clear call to action
6. Is personalized based on the job details
7. Maintains a professional yet friendly tone

Format the email with:
Subject: [subject line]

[Email body]

Return ONLY the email content, nothing else.
"""
            
            response = self.llm.invoke(prompt)
            email_content = response.content.strip()
            
            logger.info("Successfully generated cold email")
            return email_content
            
        except Exception as e:
            logger.exception("Error generating cold email")
            raise e
    
    def process_prompt(self, query: str) -> str:
        """
        Process a general prompt using Groq LLM.
        
        Args:
            query (str): The prompt/query to process
            
        Returns:
            str: LLM response
        """
        try:
            logger.info(f"Processing prompt: {query[:80]}...")
            
            response = self.llm.invoke(query)
            
            logger.info("Prompt processed successfully")
            return response.content
            
        except Exception as e:
            logger.exception(f"Error processing prompt: {query[:80]}")
            raise e


# Singleton instance
groq_service = GroqService()


# Convenience functions
def extract_job_info(job_content: str) -> Dict[str, Any]:
    """Extract job information and return JSON"""
    return groq_service.extract_job_info(job_content)


def generate_cold_email(job_info: Dict[str, Any]) -> str:
    """Generate cold email from job information"""
    return groq_service.generate_cold_email(job_info)


def process_prompt(query: str) -> str:
    """Process a general prompt"""
    return groq_service.process_prompt(query)