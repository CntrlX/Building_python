"""
Vision Counter Service

This service uses OpenAI's Vision API to analyze floor plan images and count architectural elements.
"""

import os
import logging
import base64
from pathlib import Path
from typing import List, Dict, Any
import json
from openai import OpenAI
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionCounter:
    def __init__(self, api_key=None):
        """
        Initialize the vision counter service.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a floor plan image using OpenAI's Vision API.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results including counts of architectural elements
        """
        try:
            # Prepare the image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Encode image
            base64_image = self.encode_image(str(image_path))
            
            # Prepare the prompt for accurate counting
            prompt = """
            Please analyze this architectural floor plan image and provide an accurate count of the following elements:
            
            1. Doors (including all types: single, double, sliding, etc.)
            2. Security Cameras/CCTV (look for camera symbols, usually circular or dome-shaped)
            3. Furniture:
               - Tables (conference tables, desks, etc.)
               - Chairs (office chairs, meeting chairs, etc.)
               - Cabinets and Storage
               - Other furniture items
            
            For each category, please:
            1. Count the total number
            2. Describe their locations
            3. Note any special types or variations
            4. Indicate your confidence level in the count (high/medium/low)
            
            Format the response as a JSON object with these counts and details.
            """
            
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            # Parse the response
            try:
                # Extract the JSON from the response text
                response_text = response.choices[0].message.content
                # Find the JSON part in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    counts = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the entire response
                    counts = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response from the text
                counts = self._parse_unstructured_response(response_text)
            
            return counts
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
            
    def _parse_unstructured_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse unstructured response text into a structured format.
        
        Args:
            response_text (str): The response text from the API
            
        Returns:
            dict: Structured count data
        """
        # Default structure for counts
        counts = {
            "doors": {"count": 0, "confidence": "low", "details": ""},
            "security_cameras": {"count": 0, "confidence": "low", "details": ""},
            "furniture": {
                "tables": {"count": 0, "confidence": "low", "details": ""},
                "chairs": {"count": 0, "confidence": "low", "details": ""},
                "cabinets": {"count": 0, "confidence": "low", "details": ""},
                "other": {"count": 0, "confidence": "low", "details": ""}
            }
        }
        
        # Try to extract numbers and details from the text
        lines = response_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.lower().strip()
            if not line:
                continue
                
            # Check for category headers
            if "door" in line:
                current_category = "doors"
            elif "camera" in line or "cctv" in line:
                current_category = "security_cameras"
            elif "table" in line:
                current_category = "furniture.tables"
            elif "chair" in line:
                current_category = "furniture.chairs"
            elif "cabinet" in line or "storage" in line:
                current_category = "furniture.cabinets"
            
            # Extract numbers
            import re
            numbers = re.findall(r'\d+', line)
            if numbers and current_category:
                # Update the appropriate count
                if current_category.startswith("furniture."):
                    category_parts = current_category.split('.')
                    counts["furniture"][category_parts[1]]["count"] = int(numbers[0])
                    counts["furniture"][category_parts[1]]["details"] = line
                else:
                    counts[current_category]["count"] = int(numbers[0])
                    counts[current_category]["details"] = line
                
                # Set confidence based on language used
                confidence = "medium"
                if "approximately" in line or "about" in line:
                    confidence = "medium"
                elif "exactly" in line or "counted" in line:
                    confidence = "high"
                
                if current_category.startswith("furniture."):
                    counts["furniture"][category_parts[1]]["confidence"] = confidence
                else:
                    counts[current_category]["confidence"] = confidence
        
        return counts
        
    def process_multiple_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple floor plan images and combine the results.
        
        Args:
            image_paths (List[str]): List of paths to image files
            
        Returns:
            dict: Combined analysis results
        """
        combined_results = {
            "doors": {"count": 0, "confidence": "low", "details": []},
            "security_cameras": {"count": 0, "confidence": "low", "details": []},
            "furniture": {
                "tables": {"count": 0, "confidence": "low", "details": []},
                "chairs": {"count": 0, "confidence": "low", "details": []},
                "cabinets": {"count": 0, "confidence": "low", "details": []},
                "other": {"count": 0, "confidence": "low", "details": []}
            },
            "page_details": []
        }
        
        confidence_scores = {"high": 3, "medium": 2, "low": 1}
        
        for idx, image_path in enumerate(image_paths):
            try:
                # Analyze each image
                result = self.analyze_image(image_path)
                page_num = idx + 1
                
                # Combine counts and track confidence
                for category in ["doors", "security_cameras"]:
                    if category in result:
                        combined_results[category]["count"] += result[category]["count"]
                        combined_results[category]["details"].append(
                            f"Page {page_num}: {result[category]['details']}"
                        )
                        
                        # Update confidence based on highest confidence seen
                        current_confidence = combined_results[category]["confidence"]
                        new_confidence = result[category]["confidence"]
                        if confidence_scores.get(new_confidence, 0) > confidence_scores.get(current_confidence, 0):
                            combined_results[category]["confidence"] = new_confidence
                
                # Combine furniture counts
                if "furniture" in result:
                    for furniture_type in ["tables", "chairs", "cabinets", "other"]:
                        if furniture_type in result["furniture"]:
                            combined_results["furniture"][furniture_type]["count"] += \
                                result["furniture"][furniture_type]["count"]
                            combined_results["furniture"][furniture_type]["details"].append(
                                f"Page {page_num}: {result['furniture'][furniture_type]['details']}"
                            )
                            
                            # Update confidence
                            current_confidence = combined_results["furniture"][furniture_type]["confidence"]
                            new_confidence = result["furniture"][furniture_type]["confidence"]
                            if confidence_scores.get(new_confidence, 0) > confidence_scores.get(current_confidence, 0):
                                combined_results["furniture"][furniture_type]["confidence"] = new_confidence
                
                # Add page details
                combined_results["page_details"].append({
                    "page_number": page_num,
                    "image_path": image_path,
                    "analysis_success": True
                })
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                combined_results["page_details"].append({
                    "page_number": idx + 1,
                    "image_path": image_path,
                    "analysis_success": False,
                    "error": str(e)
                })
        
        return combined_results 

    def count_items_in_image(self, image_path, categories=None):
        """
        Count specific items in an image using OpenAI's Vision API.
        
        Args:
            image_path (str): Path to the image file
            categories (list, optional): List of item categories to count
            
        Returns:
            dict: Counts of items by category
        """
        try:
            if categories is None:
                categories = ['doors', 'windows', 'security_cameras', 'furniture']
            
            # Set up OpenAI client
            client = OpenAI(api_key=self.api_key)
            
            # Read the image
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return {"error": "Image file not found"}
                
            # Encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create the prompt
            prompt = self._create_prompt(categories)
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o", # Changed from gpt-4-vision to gpt-4o which has vision capabilities
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            # Parse the response
            return self._parse_response(response, categories)
            
        except Exception as e:
            logger.error(f"Error counting items in image: {e}")
            raise 