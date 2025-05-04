"""
Pricing Manager Module

This module handles the management of construction material costs, labor rates, and location factors.
It provides functionality to read, update, and validate pricing data.
"""
import os
import logging
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PricingManager:
    """Manages construction pricing data including material costs, labor rates, and location factors."""
    
    def __init__(self, data_dir: str = "data/cost_data"):
        """
        Initialize the pricing manager.
        
        Args:
            data_dir (str): Directory containing pricing data files
        """
        self.data_dir = data_dir
        self.materials_df = None
        self.labor_df = None
        self.location_df = None
        self.load_pricing_data()
    
    def load_pricing_data(self) -> None:
        """Load pricing data from CSV files."""
        try:
            # Ensure data directory exists
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
            
            # Define paths to pricing data files
            materials_file = os.path.join(self.data_dir, "materials_costs.csv")
            labor_file = os.path.join(self.data_dir, "labor_rates.csv")
            location_file = os.path.join(self.data_dir, "location_factors.csv")
            
            # Check if files exist, otherwise create with default data
            self._ensure_pricing_files(materials_file, labor_file, location_file)
            
            # Load pricing data
            self.materials_df = pd.read_csv(materials_file)
            self.labor_df = pd.read_csv(labor_file)
            self.location_df = pd.read_csv(location_file)
            
            logger.info("Pricing data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading pricing data: {e}")
            raise
    
    def _ensure_pricing_files(self, materials_file: str, labor_file: str, location_file: str) -> None:
        """Create default pricing data files if they don't exist."""
        # Default material costs
        if not os.path.exists(materials_file):
            logger.warning(f"Materials cost file not found. Creating default at {materials_file}")
            default_materials = [
                {"material_category": "structural", "material_name": "concrete", "unit": "m³", "unit_cost": 120.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "structural", "material_name": "steel_rebar", "unit": "ton", "unit_cost": 1200.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "structural", "material_name": "steel_structural", "unit": "ton", "unit_cost": 2000.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "enclosure", "material_name": "brick", "unit": "m²", "unit_cost": 85.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "enclosure", "material_name": "curtain_wall", "unit": "m²", "unit_cost": 750.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "enclosure", "material_name": "window", "unit": "m²", "unit_cost": 450.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "enclosure", "material_name": "door", "unit": "each", "unit_cost": 350.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "enclosure", "material_name": "roofing_membrane", "unit": "m²", "unit_cost": 65.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "enclosure", "material_name": "insulation", "unit": "m²", "unit_cost": 25.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "finishes", "material_name": "drywall", "unit": "m²", "unit_cost": 35.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "finishes", "material_name": "paint", "unit": "m²", "unit_cost": 15.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "finishes", "material_name": "flooring_tile", "unit": "m²", "unit_cost": 95.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "finishes", "material_name": "flooring_carpet", "unit": "m²", "unit_cost": 45.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "finishes", "material_name": "ceiling_suspended", "unit": "m²", "unit_cost": 55.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "mechanical", "material_name": "ductwork", "unit": "m", "unit_cost": 120.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "mechanical", "material_name": "hvac_equipment", "unit": "each", "unit_cost": 5000.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "electrical", "material_name": "wiring", "unit": "m", "unit_cost": 25.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "electrical", "material_name": "lighting_fixture", "unit": "each", "unit_cost": 250.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "plumbing", "material_name": "pipe_copper", "unit": "m", "unit_cost": 45.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "plumbing", "material_name": "pipe_pvc", "unit": "m", "unit_cost": 20.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "plumbing", "material_name": "fixture_sink", "unit": "each", "unit_cost": 450.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"material_category": "plumbing", "material_name": "fixture_toilet", "unit": "each", "unit_cost": 350.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")}
            ]
            pd.DataFrame(default_materials).to_csv(materials_file, index=False)
        
        # Default labor rates
        if not os.path.exists(labor_file):
            logger.warning(f"Labor rates file not found. Creating default at {labor_file}")
            default_labor = [
                {"trade": "general_labor", "unit": "hour", "rate": 25.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "carpenter", "unit": "hour", "rate": 35.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "electrician", "unit": "hour", "rate": 45.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "plumber", "unit": "hour", "rate": 50.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "hvac_technician", "unit": "hour", "rate": 48.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "concrete_worker", "unit": "hour", "rate": 32.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "mason", "unit": "hour", "rate": 38.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "roofer", "unit": "hour", "rate": 34.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "painter", "unit": "hour", "rate": 30.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "steel_worker", "unit": "hour", "rate": 42.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "glazier", "unit": "hour", "rate": 36.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "tile_setter", "unit": "hour", "rate": 35.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"trade": "drywall_installer", "unit": "hour", "rate": 32.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")}
            ]
            pd.DataFrame(default_labor).to_csv(labor_file, index=False)
        
        # Default location factors
        if not os.path.exists(location_file):
            logger.warning(f"Location factors file not found. Creating default at {location_file}")
            default_locations = [
                {"location": "New York City, NY", "factor": 1.35, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Chicago, IL", "factor": 1.15, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Los Angeles, CA", "factor": 1.25, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Houston, TX", "factor": 0.95, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Miami, FL", "factor": 1.05, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Boston, MA", "factor": 1.30, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "San Francisco, CA", "factor": 1.40, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Seattle, WA", "factor": 1.20, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Denver, CO", "factor": 1.10, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Phoenix, AZ", "factor": 0.90, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Rural Areas", "factor": 0.85, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")},
                {"location": "Default", "factor": 1.00, "currency": "USD", "date_updated": datetime.now().strftime("%Y-%m-%d")}
            ]
            pd.DataFrame(default_locations).to_csv(location_file, index=False)
    
    def update_material_cost(self, material_name: str, category: str, unit_cost: float, unit: str, currency: str = "USD") -> bool:
        """
        Update the cost of a material.
        
        Args:
            material_name (str): Name of the material
            category (str): Material category
            unit_cost (float): New unit cost
            unit (str): Unit of measurement
            currency (str): Currency code (default: USD)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Validate input
            if not all([material_name, category, unit_cost, unit]):
                logger.error("Missing required parameters for material cost update")
                return False
            
            if unit_cost <= 0:
                logger.error("Unit cost must be positive")
                return False
            
            # Update or add material cost
            mask = (self.materials_df['material_name'] == material_name) & (self.materials_df['material_category'] == category)
            if mask.any():
                # Update existing material
                self.materials_df.loc[mask, ['unit_cost', 'unit', 'currency', 'date_updated']] = [
                    unit_cost, unit, currency, datetime.now().strftime("%Y-%m-%d")
                ]
            else:
                # Add new material
                new_row = pd.DataFrame([{
                    'material_category': category,
                    'material_name': material_name,
                    'unit': unit,
                    'unit_cost': unit_cost,
                    'currency': currency,
                    'date_updated': datetime.now().strftime("%Y-%m-%d")
                }])
                self.materials_df = pd.concat([self.materials_df, new_row], ignore_index=True)
            
            # Save changes
            self.materials_df.to_csv(os.path.join(self.data_dir, "materials_costs.csv"), index=False)
            logger.info(f"Updated material cost for {material_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating material cost: {e}")
            return False
    
    def update_labor_rate(self, trade: str, rate: float, unit: str, currency: str = "USD") -> bool:
        """
        Update the labor rate for a trade.
        
        Args:
            trade (str): Trade name
            rate (float): New hourly rate
            unit (str): Unit of measurement (e.g., "hour")
            currency (str): Currency code (default: USD)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Validate input
            if not all([trade, rate, unit]):
                logger.error("Missing required parameters for labor rate update")
                return False
            
            if rate <= 0:
                logger.error("Labor rate must be positive")
                return False
            
            # Update or add labor rate
            mask = self.labor_df['trade'] == trade
            if mask.any():
                # Update existing trade
                self.labor_df.loc[mask, ['rate', 'unit', 'currency', 'date_updated']] = [
                    rate, unit, currency, datetime.now().strftime("%Y-%m-%d")
                ]
            else:
                # Add new trade
                new_row = pd.DataFrame([{
                    'trade': trade,
                    'unit': unit,
                    'rate': rate,
                    'currency': currency,
                    'date_updated': datetime.now().strftime("%Y-%m-%d")
                }])
                self.labor_df = pd.concat([self.labor_df, new_row], ignore_index=True)
            
            # Save changes
            self.labor_df.to_csv(os.path.join(self.data_dir, "labor_rates.csv"), index=False)
            logger.info(f"Updated labor rate for {trade}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating labor rate: {e}")
            return False
    
    def update_location_factor(self, location: str, factor: float, currency: str = "USD") -> bool:
        """
        Update the cost factor for a location.
        
        Args:
            location (str): Location name
            factor (float): New cost factor
            currency (str): Currency code (default: USD)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Validate input
            if not all([location, factor]):
                logger.error("Missing required parameters for location factor update")
                return False
            
            if factor <= 0:
                logger.error("Location factor must be positive")
                return False
            
            # Update or add location factor
            mask = self.location_df['location'] == location
            if mask.any():
                # Update existing location
                self.location_df.loc[mask, ['factor', 'currency', 'date_updated']] = [
                    factor, currency, datetime.now().strftime("%Y-%m-%d")
                ]
            else:
                # Add new location
                new_row = pd.DataFrame([{
                    'location': location,
                    'factor': factor,
                    'currency': currency,
                    'date_updated': datetime.now().strftime("%Y-%m-%d")
                }])
                self.location_df = pd.concat([self.location_df, new_row], ignore_index=True)
            
            # Save changes
            self.location_df.to_csv(os.path.join(self.data_dir, "location_factors.csv"), index=False)
            logger.info(f"Updated location factor for {location}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating location factor: {e}")
            return False
    
    def get_material_cost(self, material_name: str, category: str = None) -> Optional[Dict]:
        """
        Get the cost information for a material.
        
        Args:
            material_name (str): Name of the material
            category (str, optional): Material category to filter by
            
        Returns:
            Optional[Dict]: Material cost information or None if not found
        """
        try:
            mask = self.materials_df['material_name'] == material_name
            if category:
                mask &= self.materials_df['material_category'] == category
            
            if mask.any():
                return self.materials_df[mask].iloc[0].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting material cost: {e}")
            return None
    
    def get_labor_rate(self, trade: str) -> Optional[Dict]:
        """
        Get the labor rate information for a trade.
        
        Args:
            trade (str): Trade name
            
        Returns:
            Optional[Dict]: Labor rate information or None if not found
        """
        try:
            mask = self.labor_df['trade'] == trade
            if mask.any():
                return self.labor_df[mask].iloc[0].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting labor rate: {e}")
            return None
    
    def get_location_factor(self, location: str) -> Optional[Dict]:
        """
        Get the cost factor information for a location.
        
        Args:
            location (str): Location name
            
        Returns:
            Optional[Dict]: Location factor information or None if not found
        """
        try:
            mask = self.location_df['location'] == location
            if mask.any():
                return self.location_df[mask].iloc[0].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting location factor: {e}")
            return None
    
    def export_pricing_data(self, output_dir: str = None) -> bool:
        """
        Export pricing data to JSON files.
        
        Args:
            output_dir (str, optional): Directory to save exported files
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            if output_dir is None:
                output_dir = self.data_dir
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Export materials data
            materials_file = os.path.join(output_dir, "materials_costs.json")
            self.materials_df.to_json(materials_file, orient='records', indent=2)
            
            # Export labor data
            labor_file = os.path.join(output_dir, "labor_rates.json")
            self.labor_df.to_json(labor_file, orient='records', indent=2)
            
            # Export location data
            location_file = os.path.join(output_dir, "location_factors.json")
            self.location_df.to_json(location_file, orient='records', indent=2)
            
            logger.info(f"Pricing data exported to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting pricing data: {e}")
            return False
    
    def import_pricing_data(self, input_dir: str = None) -> bool:
        """
        Import pricing data from JSON files.
        
        Args:
            input_dir (str, optional): Directory containing pricing files
            
        Returns:
            bool: True if import was successful, False otherwise
        """
        try:
            if input_dir is None:
                input_dir = self.data_dir
            
            # Import materials data
            materials_file = os.path.join(input_dir, "materials_costs.json")
            if os.path.exists(materials_file):
                self.materials_df = pd.read_json(materials_file)
                self.materials_df.to_csv(os.path.join(self.data_dir, "materials_costs.csv"), index=False)
            
            # Import labor data
            labor_file = os.path.join(input_dir, "labor_rates.json")
            if os.path.exists(labor_file):
                self.labor_df = pd.read_json(labor_file)
                self.labor_df.to_csv(os.path.join(self.data_dir, "labor_rates.csv"), index=False)
            
            # Import location data
            location_file = os.path.join(input_dir, "location_factors.json")
            if os.path.exists(location_file):
                self.location_df = pd.read_json(location_file)
                self.location_df.to_csv(os.path.join(self.data_dir, "location_factors.csv"), index=False)
            
            logger.info(f"Pricing data imported from {input_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing pricing data: {e}")
            return False 