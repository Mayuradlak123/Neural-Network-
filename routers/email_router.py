from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from services.job_scraper import JobScraperService
from services.groq_service import extract_job_info, generate_cold_email
from config.logger import logger
from typing import Dict, Any

email_router = APIRouter(tags=["email"])

# Initialize service
job_scraper = JobScraperService()


class JobLinkRequest(BaseModel):
    job_link: HttpUrl


class JobInfoRequest(BaseModel):
    job_info: Dict[str, Any]


@email_router.post("/scrape-job")
async def scrape_job_description(request: JobLinkRequest):
    """
    Scrape and return job description data from URL.
    """
    try:
        logger.info(f"Scraping job from: {request.job_link}")
        
        # Extract job description using the service
        job_data = job_scraper.extract_job_description(str(request.job_link))
        print(job_data)
        if not job_data:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract job description from URL"
            )
        
        return {
            "success": True,
            "data": job_data
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@email_router.post("/extract-job-info")
async def extract_job_information(request: JobLinkRequest):
    """
    Scrape job URL and extract structured information as JSON.
    """
    try:
        logger.info(f"Extracting job info from: {request.job_link}")
        
        # Step 1: Scrape the job description
        job_data = job_scraper.extract_job_description(str(request.job_link))
        print(job_data)
        if not job_data:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract job description from URL"
            )
        
        # Step 2: Extract structured information using Groq
        job_info = extract_job_info(job_data['cleaned_content'])
        
        return {
            "success": True,
            "job_info": job_info,
            "source_url": job_data['url']
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@email_router.post("/generate-email-from-info")
async def generate_email_from_job_info(request: JobInfoRequest):
    """
    Generate cold email from provided job information JSON.
    """
    try:
        logger.info("Generating email from provided job info")
        
        # Generate cold email using Groq
        email_content = generate_cold_email(request.job_info)
        
        return {
            "success": True,
            "email": email_content,
            "job_info": request.job_info
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@email_router.post("/generate-email")
async def generate_cold_email_complete(request: JobLinkRequest):
    """
    Complete workflow: Scrape job URL -> Extract info -> Generate email.
    """
    try:
        logger.info(f"Complete email generation for: {request.job_link}")
        
        # Step 1: Scrape job description
        job_data = job_scraper.extract_job_description(str(request.job_link))
        
        if not job_data:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract job description from URL"
            )
        
        # Step 2: Extract structured job information
        job_info = extract_job_info(job_data['cleaned_content'])
        
        # Step 3: Generate cold email
        email_content = generate_cold_email(job_info)
        
        return {
            "success": True,
            "email": email_content,
            "job_info": job_info,
            "source_url": job_data['url']
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))