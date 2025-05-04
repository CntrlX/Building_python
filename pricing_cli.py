"""
Pricing CLI Module

This module provides a command-line interface for managing construction pricing data.
"""
import argparse
import logging
from pricing_manager import PricingManager
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the pricing CLI."""
    parser = argparse.ArgumentParser(description="Manage construction pricing data")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # View pricing data
    view_parser = subparsers.add_parser("view", help="View pricing data")
    view_parser.add_argument("--type", choices=["materials", "labor", "locations", "all"], default="all",
                            help="Type of pricing data to view")
    
    # Update material cost
    update_material_parser = subparsers.add_parser("update-material", help="Update material cost")
    update_material_parser.add_argument("--name", required=True, help="Material name")
    update_material_parser.add_argument("--category", required=True, help="Material category")
    update_material_parser.add_argument("--cost", type=float, required=True, help="New unit cost")
    update_material_parser.add_argument("--unit", required=True, help="Unit of measurement")
    update_material_parser.add_argument("--currency", default="USD", help="Currency code")
    
    # Update labor rate
    update_labor_parser = subparsers.add_parser("update-labor", help="Update labor rate")
    update_labor_parser.add_argument("--trade", required=True, help="Trade name")
    update_labor_parser.add_argument("--rate", type=float, required=True, help="New hourly rate")
    update_labor_parser.add_argument("--unit", default="hour", help="Unit of measurement")
    update_labor_parser.add_argument("--currency", default="USD", help="Currency code")
    
    # Update location factor
    update_location_parser = subparsers.add_parser("update-location", help="Update location factor")
    update_location_parser.add_argument("--location", required=True, help="Location name")
    update_location_parser.add_argument("--factor", type=float, required=True, help="New cost factor")
    update_location_parser.add_argument("--currency", default="USD", help="Currency code")
    
    # Export pricing data
    export_parser = subparsers.add_parser("export", help="Export pricing data")
    export_parser.add_argument("--output-dir", help="Output directory for exported files")
    
    # Import pricing data
    import_parser = subparsers.add_parser("import", help="Import pricing data")
    import_parser.add_argument("--input-dir", help="Input directory containing pricing files")
    
    args = parser.parse_args()
    
    try:
        # Initialize pricing manager
        pricing_manager = PricingManager()
        
        if args.command == "view":
            if args.type in ["materials", "all"]:
                print("\nMaterial Costs:")
                for _, row in pricing_manager.materials_df.iterrows():
                    print(f"  {row['material_name']} ({row['material_category']}): {row['unit_cost']} {row['currency']}/{row['unit']}")
            
            if args.type in ["labor", "all"]:
                print("\nLabor Rates:")
                for _, row in pricing_manager.labor_df.iterrows():
                    print(f"  {row['trade']}: {row['rate']} {row['currency']}/{row['unit']}")
            
            if args.type in ["locations", "all"]:
                print("\nLocation Factors:")
                for _, row in pricing_manager.location_df.iterrows():
                    print(f"  {row['location']}: {row['factor']} {row['currency']}")
        
        elif args.command == "update-material":
            success = pricing_manager.update_material_cost(
                args.name, args.category, args.cost, args.unit, args.currency
            )
            if success:
                print(f"Successfully updated material cost for {args.name}")
            else:
                print(f"Failed to update material cost for {args.name}")
        
        elif args.command == "update-labor":
            success = pricing_manager.update_labor_rate(
                args.trade, args.rate, args.unit, args.currency
            )
            if success:
                print(f"Successfully updated labor rate for {args.trade}")
            else:
                print(f"Failed to update labor rate for {args.trade}")
        
        elif args.command == "update-location":
            success = pricing_manager.update_location_factor(
                args.location, args.factor, args.currency
            )
            if success:
                print(f"Successfully updated location factor for {args.location}")
            else:
                print(f"Failed to update location factor for {args.location}")
        
        elif args.command == "export":
            success = pricing_manager.export_pricing_data(args.output_dir)
            if success:
                print(f"Successfully exported pricing data to {args.output_dir or 'default directory'}")
            else:
                print("Failed to export pricing data")
        
        elif args.command == "import":
            success = pricing_manager.import_pricing_data(args.input_dir)
            if success:
                print(f"Successfully imported pricing data from {args.input_dir or 'default directory'}")
            else:
                print("Failed to import pricing data")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error in pricing CLI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 