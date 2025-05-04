"""
PDF Processor Service

This service handles the extraction and processing of PDF documents for AI-powered counting.
"""

import os
import logging
import fitz  # PyMuPDF
from pathlib import Path
import tempfile
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
    def extract_images(self, output_dir=None):
        """
        Extract images from the PDF file.
        
        Args:
            output_dir (str, optional): Directory to save extracted images
            
        Returns:
            list: List of paths to extracted images
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        image_paths = []
        try:
            # Open the PDF
            pdf_document = fitz.open(self.pdf_path)
            
            # Process each page
            for page_num, page in enumerate(pdf_document):
                # Get page as image
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Save the image
                image_path = output_dir / f"page_{page_num + 1}.png"
                img.save(image_path, "PNG")
                image_paths.append(str(image_path))
                
                logger.info(f"Extracted image from page {page_num + 1}: {image_path}")
                
            return image_paths
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            raise
            
    def process_pdf(self, output_dir=None):
        """
        Process the PDF file and prepare it for AI analysis.
        
        Args:
            output_dir (str, optional): Directory to save processed files
            
        Returns:
            dict: Processing results including image paths and metadata
        """
        try:
            # Extract images from PDF
            image_paths = self.extract_images(output_dir)
            
            # Get PDF metadata
            pdf_document = fitz.open(self.pdf_path)
            metadata = {
                "page_count": len(pdf_document),
                "title": pdf_document.metadata.get("title", "Unknown"),
                "author": pdf_document.metadata.get("author", "Unknown"),
                "creation_date": pdf_document.metadata.get("creationDate", "Unknown"),
                "modification_date": pdf_document.metadata.get("modDate", "Unknown")
            }
            
            return {
                "image_paths": image_paths,
                "metadata": metadata,
                "original_pdf": str(self.pdf_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise 