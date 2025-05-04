"""
CAD Material Estimator - Main Application

This application processes CAD files to identify construction materials 
and estimate construction costs based on the identified materials.
"""
import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import the modules
try:
    from material_identifier import identify_materials, categorize_materials
    from cost_estimator import estimate_costs
    from cad_processor import process_cad_file
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are installed.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cadme.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CAD Material Estimator')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CAD file path')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory for results')
    parser.add_argument('--location', '-l', type=str, default='Default',
                        help='Project location for regional cost factors')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'csv', 'xlsx'], default='json',
                        help='Output format for the estimate')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for advanced material identification')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip exporting results to files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--pdf', type=str,
                        help='Path to PDF file for AI-powered counting')
    parser.add_argument('--openai-key', type=str,
                        help='OpenAI API key for Vision analysis')
    parser.add_argument('--use-svg-counting', action='store_true',
                        help='Use SVG-based shape counting for more accurate quantification')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for the application."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Log starting information
    logger.info(f"Starting CAD Material Estimator")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Project location: {args.location}")
    logger.info(f"Using LLM: {'Yes' if args.use_llm else 'No'}")

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up the environment
    setup_environment(args)
    
    try:
        # Process the CAD file
        logger.info(f"Processing CAD file: {args.input}")
        cad_data = process_cad_file(args.input)
        
        # Export processed CAD data if requested
        if not args.no_export:
            cad_output_file = os.path.join(args.output, "processed_cad_data.json")
            with open(cad_output_file, 'w') as f:
                json.dump(cad_data, f, indent=2)
            logger.info(f"Processed CAD data exported to {cad_output_file}")
        
        # Identify materials using the comprehensive identify_materials function
        logger.info("Identifying materials from CAD data")
        materials_data = identify_materials(cad_data, use_llm=args.use_llm)
        
        # If SVG-based counting is enabled, use it for more accurate counting
        if args.use_svg_counting:
            try:
                from services.svg_shape_counter import SVGShapeCounter
                
                # Process the DWG file with SVG-based counting
                logger.info("Using SVG-based shape counting for more accurate quantification")
                svg_counter = SVGShapeCounter()
                svg_results = svg_counter.process_and_count(
                    args.input,
                    os.path.join(args.output, "svg_output")
                )
                
                # Update material quantities with SVG-based counts
                logger.info("Updating material quantities with SVG-based counts")
                
                # Get the classifications from SVG results
                svg_counts = svg_results.get('classifications', {})
                
                # Update the material quantities
                for category, count in svg_counts.items():
                    if count > 0:
                        # Map SVG categories to material names
                        material_name = category
                        if category == 'security_camera':
                            material_name = 'security'
                        elif category == 'fixture':
                            material_name = 'fixture'
                        
                        # Create or update the material in the materials_data
                        if material_name not in materials_data.get("material_quantities", {}):
                            materials_data.setdefault("material_quantities", {})[material_name] = {
                                "count": count,
                                "category": "architectural" if material_name in ["door", "window"] else "interior",
                                "subcategory": "openings" if material_name in ["door", "window"] else "furnishings",
                                "source": "svg_counting"
                            }
                        else:
                            materials_data["material_quantities"][material_name]["count"] = count
                            materials_data["material_quantities"][material_name]["source"] = "svg_counting"
                
                # Add source information
                materials_data.setdefault("count_sources", {}).update({
                    "svg": True
                })
                
                # Export SVG-based counts if requested
                if not args.no_export:
                    svg_output_file = os.path.join(args.output, "svg_counts.json")
                    with open(svg_output_file, 'w') as f:
                        json.dump(svg_results, f, indent=2)
                    logger.info(f"SVG-based counts exported to {svg_output_file}")
            
            except ImportError as e:
                logger.error(f"Error importing SVG shape counter: {e}")
                logger.warning("Proceeding without SVG-based counting")
            except Exception as e:
                logger.error(f"Error in SVG-based counting: {e}")
                logger.warning("Proceeding without SVG-based counting")
        
        # If PDF file is provided, use AI-powered counting
        if args.pdf:
            try:
                from services.pdf_processor import PDFProcessor
                from services.vision_counter import VisionCounter
                from services.count_reconciler import CountReconciler
                
                # Process PDF and extract images
                logger.info(f"Processing PDF file: {args.pdf}")
                pdf_processor = PDFProcessor(args.pdf)
                pdf_data = pdf_processor.process_pdf(os.path.join(args.output, "pdf_images"))
                
                # Analyze images with Vision API
                logger.info("Analyzing floor plan images with Vision API")
                vision_counter = VisionCounter(api_key=args.openai_key)
                vision_counts = vision_counter.process_multiple_images(pdf_data["image_paths"])
                
                # Reconcile counts
                logger.info("Reconciling counts from DXF and Vision analysis")
                reconciler = CountReconciler()
                reconciled_counts = reconciler.reconcile_counts(
                    materials_data.get("material_quantities", {}),
                    vision_counts
                )
                
                # Export reconciled counts
                if not args.no_export:
                    reconciled_output_file = os.path.join(args.output, "reconciled_counts.json")
                    reconciler.export_reconciled_counts(reconciled_counts, reconciled_output_file)
                    logger.info(f"Reconciled counts exported to {reconciled_output_file}")
                
                # Update materials_data with reconciled counts
                materials_data["material_quantities"].update(reconciled_counts)
                materials_data.setdefault("count_sources", {}).update({
                    "dxf": True,
                    "vision": True,
                    "reconciled": True
                })
                
            except ImportError as e:
                logger.error(f"Error importing AI services: {e}")
                logger.warning("Proceeding with DXF analysis only")
            except Exception as e:
                logger.error(f"Error in AI-powered counting: {e}")
                logger.warning("Proceeding with DXF analysis only")
        
        # Categorize materials
        logger.info("Categorizing materials")
        categorized_materials = categorize_materials(materials_data)
        
        # Export materials data if requested
        if not args.no_export:
            materials_output_file = os.path.join(args.output, "materials_data.json")
            with open(materials_output_file, 'w') as f:
                json.dump(categorized_materials, f, indent=2)
            logger.info(f"Materials data exported to {materials_output_file}")
            
            # Generate count.json with material counts
            logger.info("Generating material count data")
            material_counts = {
                "project_info": materials_data.get("project_info", {}),
                "material_counts": {}
            }
            
            # Extract count information from materials_data
            for material_name, material_info in materials_data.get("material_quantities", {}).items():
                material_counts["material_counts"][material_name] = {
                    "count": material_info.get("count", 0),
                    "category": material_info.get("category", "other"),
                    "subcategory": material_info.get("subcategory", "unknown")
                }
            
            # Save count.json
            count_output_file = os.path.join(args.output, "count.json")
            with open(count_output_file, 'w') as f:
                json.dump(material_counts, f, indent=2)
            logger.info(f"Material count data exported to {count_output_file}")
        
        # Estimate costs
        logger.info(f"Estimating costs for location: {args.location}")
        export_format = None if args.no_export else args.format
        cost_estimate = estimate_costs(
            cad_data, 
            categorized_materials,
            location=args.location,
            export_format=export_format
        )
        
        # Print summary to console
        print("\n===== Cost Estimate Summary =====")
        print(f"Project: {cost_estimate.get('project_info', {}).get('name', 'Unknown')}")
        print(f"Location: {args.location} (Factor: {cost_estimate.get('location_factor', 1.0)})")
        print(f"Estimation Date: {datetime.now().strftime('%Y-%m-%d')}")
        print("\nMaterial Categories:")
        
        for category, data in cost_estimate.get('categories', {}).items():
            print(f"  - {category.title()}: {data.get('total_cost', 0):.2f} {cost_estimate.get('currency', 'USD')}")
        
        print(f"\nMaterials Subtotal: {cost_estimate.get('summary', {}).get('materials_subtotal', 0):.2f} {cost_estimate.get('currency', 'USD')}")
        print(f"Labor Subtotal: {cost_estimate.get('summary', {}).get('labor_subtotal', 0):.2f} {cost_estimate.get('currency', 'USD')}")
        print(f"Total Estimate: {cost_estimate.get('summary', {}).get('total', 0):.2f} {cost_estimate.get('currency', 'USD')}")
        
        if not args.no_export:
            print(f"\nDetailed results exported to: {args.output}")
        
        logger.info("CAD Material Estimator completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error in CAD Material Estimator: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 