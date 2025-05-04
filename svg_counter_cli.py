#!/usr/bin/env python
"""
SVG Shape Counter CLI

Command-line interface for converting DWG/DXF files to SVG and counting shapes.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SVG Shape Counter CLI')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input DWG/DXF file path')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Create output directory
    output_dir = args.output
    # We'll let the SVGShapeCounter handle directory creation
    
    # Import the SVG shape counter
    try:
        from services.svg_shape_counter import SVGShapeCounter
    except ImportError as e:
        logger.error(f"Error importing SVG shape counter: {e}")
        logger.error("Make sure the 'svg_shape_counter.py' file is in the 'services' directory.")
        return 1
    
    try:
        # Initialize the shape counter
        counter = SVGShapeCounter()
        
        # Process the DWG file and count shapes
        logger.info(f"Processing file: {input_path}")
        results = counter.process_and_count(str(input_path), str(output_dir))
        
        # Display summary
        print("\n===== Shape Count Summary =====")
        print(f"Total shapes: {results['total_shapes']}")
        print(f"Unique shape types: {results['unique_shapes']}")
        print("\nClassifications:")
        for category, count in results['classifications'].items():
            if count > 0:
                print(f"  - {category.title()}: {count}")
        
        print(f"\nResults saved to: {results['results_path']}")
        print(f"SVG file saved to: {results['svg_path']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 