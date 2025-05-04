"""
Test script for PDF processing functionality.
"""

import os
import sys
import logging
import json
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test PDF processing")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--output", type=str, default="test_output", help="Output directory")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--test-vision", action="store_true", help="Test Vision API integration")
    return parser.parse_args()

def test_pdf_processor(pdf_path, output_dir):
    """Test the PDFProcessor class."""
    try:
        from services.pdf_processor import PDFProcessor
        
        logger.info(f"Testing PDF processor with file: {pdf_path}")
        pdf_processor = PDFProcessor(pdf_path)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process the PDF
        result = pdf_processor.process_pdf(output_dir)
        
        logger.info(f"PDF processing successful. Extracted {len(result['image_paths'])} images.")
        logger.info(f"PDF metadata: {result['metadata']}")
        
        # Save result to JSON
        result_file = output_path / "pdf_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
        return result
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required packages are installed: pip install PyMuPDF Pillow")
        return None
    except Exception as e:
        logger.error(f"Error testing PDF processor: {e}")
        return None

def test_vision_counter(image_paths, api_key):
    """Test the VisionCounter class."""
    try:
        from services.vision_counter import VisionCounter
        
        logger.info(f"Testing Vision Counter with {len(image_paths)} images")
        
        # Initialize vision counter
        vision_counter = VisionCounter(api_key=api_key)
        
        # Process the first image only to save API calls during testing
        test_image = image_paths[0]
        logger.info(f"Testing with image: {test_image}")
        
        result = vision_counter.analyze_image(test_image)
        
        logger.info("Vision API analysis result:")
        logger.info(f"Doors: {result.get('doors', {}).get('count', 'N/A')}")
        logger.info(f"Security Cameras: {result.get('security_cameras', {}).get('count', 'N/A')}")
        
        if 'furniture' in result:
            logger.info("Furniture counts:")
            for furniture_type, data in result['furniture'].items():
                logger.info(f"  - {furniture_type}: {data.get('count', 'N/A')}")
        
        # Save result to JSON
        output_dir = os.path.dirname(image_paths[0])
        result_file = os.path.join(output_dir, "vision_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Vision results saved to {result_file}")
        return result
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required packages are installed: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Error testing Vision Counter: {e}")
        return None

def test_count_reconciler(dxf_counts, vision_counts, output_dir):
    """Test the CountReconciler class."""
    try:
        from services.count_reconciler import CountReconciler
        
        logger.info("Testing Count Reconciler")
        
        # Initialize count reconciler
        reconciler = CountReconciler()
        
        # Reconcile counts
        reconciled_counts = reconciler.reconcile_counts(dxf_counts, vision_counts)
        
        logger.info("Reconciled counts:")
        logger.info(f"Doors: {reconciled_counts.get('doors', {}).get('count', 'N/A')}")
        logger.info(f"Security Cameras: {reconciled_counts.get('security_cameras', {}).get('count', 'N/A')}")
        
        if 'furniture' in reconciled_counts:
            logger.info("Reconciled furniture counts:")
            for furniture_type, data in reconciled_counts['furniture'].items():
                logger.info(f"  - {furniture_type}: {data.get('count', 'N/A')}")
        
        # Save result to JSON
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "reconciled_counts.json"
        
        reconciler.export_reconciled_counts(reconciled_counts, str(result_file))
        logger.info(f"Reconciled counts saved to {result_file}")
        
        return reconciled_counts
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error testing Count Reconciler: {e}")
        return None

def main():
    args = parse_args()
    
    # Test PDF processor
    pdf_result = test_pdf_processor(args.pdf, args.output)
    
    if pdf_result and args.test_vision:
        if not args.openai_key:
            logger.warning("OpenAI API key not provided. Set it with --openai-key or as OPENAI_API_KEY environment variable.")
            return 1
        
        # Test Vision Counter
        vision_result = test_vision_counter(pdf_result['image_paths'], args.openai_key)
        
        if vision_result:
            # Create a mock DXF count for testing reconciliation
            mock_dxf_counts = {
                "doors": {"count": 45, "confidence": "medium"},
                "security_cameras": {"count": 18, "confidence": "low"},
                "furniture": {
                    "tables": {"count": 30, "confidence": "medium"},
                    "chairs": {"count": 120, "confidence": "medium"},
                    "cabinets": {"count": 15, "confidence": "low"},
                    "other": {"count": 10, "confidence": "low"}
                }
            }
            
            # Test Count Reconciler
            test_count_reconciler(mock_dxf_counts, vision_result, args.output)
    
    logger.info("Tests completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 