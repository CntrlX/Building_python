"""
Cost Estimator Module - Calculates construction costs based on identified materials and dimensions
"""
import os
import logging
import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from pricing_manager import PricingManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CostEstimator:
    """Handles the estimation of construction costs based on CAD data and material identification."""
    
    def __init__(self, data_dir="data/cost_data"):
        """
        Initialize the cost estimator.
        
        Args:
            data_dir (str): Directory containing cost data files
        """
        self.data_dir = data_dir
        self.pricing_manager = PricingManager(data_dir)
        
    def estimate_material_costs(self, materials_data, location="Default"):
        """
        Estimate costs for the identified materials.
        
        Args:
            materials_data (dict): Categorized materials from material identification
            location (str): Project location for applying location factor
            
        Returns:
            dict: Estimated costs by category
        """
        logger.info(f"Estimating costs for materials in {location}")
        
        # Get location factor
        location_factor = 1.0  # Default factor
        location_data = self.pricing_manager.get_location_factor(location)
        if location_data:
            location_factor = location_data['factor']
        else:
            # Try to find the default location factor
            default_loc = self.pricing_manager.get_location_factor("Default")
            if default_loc:
                location_factor = default_loc['factor']
            logger.warning(f"Location '{location}' not found in database. Using factor: {location_factor}")
        
        # Initialize results
        estimate_results = {
            "total_cost": 0.0,
            "material_costs": {},
            "labor_costs": {},
            "categories": {},
            "summary": {},
            "location_factor": location_factor,
            "currency": "USD",  # Default currency
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Check if materials_data is properly structured
        if not isinstance(materials_data, dict):
            logger.error(f"Invalid materials_data structure: {type(materials_data)}")
            return estimate_results
        
        # Process each material category
        for category_name, category_data in materials_data.items():
            # Skip metadata keys like "_totals"
            if category_name.startswith("_"):
                continue
                
            category_total = 0.0
            category_items = []
            
            for item in items:
                material_name = item.get('material', '').lower().replace(' ', '_')
                item_type = item.get('type', 'other')
                unit = item.get('unit', 'm²')
                
                # Get quantity - different calculation based on type
                quantity = 0.0
                if item_type == 'wall':
                    # For walls, we calculate area based on length * height
                    length = float(item.get('length', 0))
                    height = float(item.get('height', 3.0))  # Default wall height of 3m if not specified
                    quantity = length * height
                elif item_type == 'floor':
                    # For floors, we directly use the area
                    quantity = float(item.get('area', 0))
                else:
                    # For other items, use quantity if provided
                    quantity = float(item.get('quantity', 1.0))
                
                # Get material cost from pricing manager
                material_data = self.pricing_manager.get_material_cost(material_name, category)
                
                if material_data:
                    unit_cost = material_data['unit_cost']
                    db_unit = material_data['unit']
                    currency = material_data['currency']
                    
                    # Unit conversion if needed (simplified - would need more sophisticated conversion in real app)
                    conversion_factor = 1.0
                    if unit != db_unit:
                        logger.warning(f"Unit mismatch for {material_name}: {unit} vs {db_unit}. Using conversion factor: {conversion_factor}")
                    
                    # Calculate material cost
                    material_cost = unit_cost * quantity * conversion_factor * location_factor
                    
                    # Add to category total
                    category_total += material_cost
                    
                    # Store individual item cost
                    item_result = {
                        "material_name": material_name,
                        "quantity": quantity,
                        "unit": unit,
                        "unit_cost": unit_cost,
                        "location_factor": location_factor,
                        "total_cost": material_cost,
                        "currency": currency
                    }
                    category_items.append(item_result)
                    
                    # Add to material costs dictionary
                    if material_name not in estimate_results["material_costs"]:
                        estimate_results["material_costs"][material_name] = material_cost
                    else:
                        estimate_results["material_costs"][material_name] += material_cost
                else:
                    # Material not found in database
                    logger.warning(f"Material '{material_name}' not found in cost database. Skipping cost calculation.")
                    item_result = {
                        "material_name": material_name,
                        "quantity": quantity,
                        "unit": unit,
                        "unit_cost": "unknown",
                        "location_factor": location_factor,
                        "total_cost": 0.0,
                        "currency": "USD",
                        "note": "Material not found in database"
                    }
                    category_items.append(item_result)
            
            # Store category results
            estimate_results["categories"][category_name] = {
                "total_cost": category_total,
                "items": category_items
            }
            
            # Add to total cost
            estimate_results["total_cost"] += category_total
        
        # Calculate labor costs (simplified - in reality would be more detailed)
        # Assuming labor is roughly 40% of material costs
        labor_cost = estimate_results["total_cost"] * 0.4
        estimate_results["labor_costs"]["total"] = labor_cost
        estimate_results["total_cost"] += labor_cost
        
        # Add summary
        estimate_results["summary"] = {
            "materials_subtotal": estimate_results["total_cost"] - labor_cost,
            "labor_subtotal": labor_cost,
            "total": estimate_results["total_cost"],
            "currency": estimate_results["currency"]
        }
        
        logger.info(f"Cost estimation complete. Total estimated cost: {estimate_results['total_cost']} {estimate_results['currency']}")
        return estimate_results
    
    def export_estimate(self, estimate_data, output_format="json", output_file=None):
        """
        Export the cost estimate to file.
        
        Args:
            estimate_data (dict): The estimated cost data
            output_format (str): Format to export (json, csv, xlsx)
            output_file (str): Path to output file
            
        Returns:
            str: Path to the exported file
        """
        if output_file is None:
            # Generate default filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output/estimates"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"estimate_{timestamp}.{output_format}")
        
        try:
            if output_format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump(estimate_data, f, indent=2)
                logger.info(f"Estimate exported as JSON to {output_file}")
            
            elif output_format.lower() == "csv":
                # Convert to flattened format for CSV
                csv_data = []
                for category, cat_data in estimate_data["categories"].items():
                    for item in cat_data["items"]:
                        row = {
                            "category": category,
                            "material_name": item["material_name"],
                            "quantity": item["quantity"],
                            "unit": item["unit"],
                            "unit_cost": item["unit_cost"],
                            "total_cost": item["total_cost"],
                            "currency": item["currency"]
                        }
                        csv_data.append(row)
                
                pd.DataFrame(csv_data).to_csv(output_file, index=False)
                logger.info(f"Estimate exported as CSV to {output_file}")
            
            elif output_format.lower() == "xlsx":
                # Create Excel workbook with multiple sheets
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = {
                        "Category": ["Materials Subtotal", "Labor Subtotal", "Total"],
                        "Cost": [
                            estimate_data["summary"]["materials_subtotal"],
                            estimate_data["summary"]["labor_subtotal"],
                            estimate_data["summary"]["total"]
                        ],
                        "Currency": [estimate_data["currency"]] * 3
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                    
                    # Category breakdown sheet
                    cat_data = []
                    for category, data in estimate_data["categories"].items():
                        cat_data.append({
                            "Category": category,
                            "Total Cost": data["total_cost"],
                            "Currency": estimate_data["currency"]
                        })
                    pd.DataFrame(cat_data).to_excel(writer, sheet_name="Categories", index=False)
                    
                    # Detailed items sheet
                    detailed_data = []
                    for category, cat_data in estimate_data["categories"].items():
                        for item in cat_data["items"]:
                            row = {
                                "Category": category,
                                "Material": item["material_name"],
                                "Quantity": item["quantity"],
                                "Unit": item["unit"],
                                "Unit Cost": item["unit_cost"],
                                "Total Cost": item["total_cost"],
                                "Currency": item["currency"]
                            }
                            detailed_data.append(row)
                    pd.DataFrame(detailed_data).to_excel(writer, sheet_name="Details", index=False)
                
                logger.info(f"Estimate exported as Excel to {output_file}")
            
            else:
                logger.warning(f"Unsupported output format: {output_format}")
                output_file = None
        
        except Exception as e:
            logger.error(f"Error exporting estimate: {e}")
            output_file = None
        
        return output_file

def estimate_costs(cad_data, materials_data, location="Default", export_format="json"):
    """
    Main function to estimate costs from CAD and material data.
    
    Args:
        cad_data (dict): Processed CAD data
        materials_data (dict): Categorized materials data
        location (str): Project location for applying regional factors
        export_format (str): Format to export the estimate (json, csv, xlsx)
        
    Returns:
        dict: Complete cost estimate
    """
    try:
        # Initialize cost estimator
        estimator = CostEstimator()
        
        # Calculate the estimate
        estimate = estimator.estimate_material_costs(materials_data, location)
        
        # Add project metadata
        estimate["project_info"] = {
            "name": cad_data.get("file_info", {}).get("file_name", "Unknown"),
            "location": location,
            "drawing_units": cad_data.get("metadata", {}).get("units", "Unknown") if isinstance(cad_data.get("metadata"), dict) else "Unknown",
            "drawing_scale": cad_data.get("metadata", {}).get("dimension_style", "1:1") if isinstance(cad_data.get("metadata"), dict) else "1:1",
            "estimation_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Export if requested
        if export_format:
            estimator.export_estimate(estimate, output_format=export_format)
        
        logger.info("Cost estimation completed successfully")
        return estimate
        
    except Exception as e:
        logger.error(f"Error in cost estimation: {e}")
        return {"error": str(e)} 