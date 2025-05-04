"""
SVG Shape Counter Service

This service provides functionality to convert DWG/DXF files to SVG format
and count identical shapes within the SVG for more accurate material quantification.
"""

import os
import logging
import tempfile
from pathlib import Path
import subprocess
import json
import collections
from typing import Dict, List, Any, Tuple

import ezdxf
try:
    from svgelements import *
    SVGELEMENTS_AVAILABLE = True
except ImportError:
    SVGELEMENTS_AVAILABLE = False

try:
    from dxf2svg import Dxf2Svg
    DXF2SVG_AVAILABLE = True
except ImportError:
    DXF2SVG_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVGShapeCounter:
    """
    Service for counting identical shapes in SVG files converted from DWG/DXF files.
    """
    
    def __init__(self):
        """
        Initialize the SVG shape counter service.
        """
        if not SVGELEMENTS_AVAILABLE:
            logger.warning("svgelements library not found. Install with 'pip install svgelements'")
        
        if not DXF2SVG_AVAILABLE:
            logger.warning("dxf2svg library not found. Install with 'pip install dxf2svg'")
    
    def convert_dwg_to_svg(self, dwg_path: str, output_path: str = None) -> str:
        """
        Convert a DWG file to SVG format.
        
        Args:
            dwg_path (str): Path to the DWG file
            output_path (str, optional): Path where the SVG should be saved
            
        Returns:
            str: Path to the generated SVG file
        """
        # Create a temporary folder for the SVG if output_path is not provided
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            dwg_file = os.path.basename(dwg_path)
            dwg_name = os.path.splitext(dwg_file)[0]
            output_path = os.path.join(temp_dir, f"{dwg_name}.svg")
        
        logger.info(f"Converting DWG to SVG: {dwg_path} -> {output_path}")
        
        # First convert DWG to DXF if needed
        if dwg_path.lower().endswith('.dwg'):
            dxf_path = self._convert_dwg_to_dxf(dwg_path)
        else:
            dxf_path = dwg_path
        
        # Convert DXF to SVG
        success = self._convert_dxf_to_svg(dxf_path, output_path)
        
        if not success:
            raise ValueError(f"Failed to convert {dwg_path} to SVG")
        
        return output_path
    
    def _convert_dwg_to_dxf(self, dwg_path: str) -> str:
        """
        Convert a DWG file to DXF format using ODA File Converter.
        
        Args:
            dwg_path (str): Path to the DWG file
            
        Returns:
            str: Path to the generated DXF file
        """
        temp_dir = tempfile.mkdtemp()
        dwg_file = os.path.basename(dwg_path)
        dwg_name = os.path.splitext(dwg_file)[0]
        dxf_path = os.path.join(temp_dir, f"{dwg_name}.dxf")
        
        # Try to convert using ODA File Converter
        # Note: This requires ODA File Converter to be installed on the system
        try:
            from cad_processor import convert_dwg_to_dxf
            success = convert_dwg_to_dxf(dwg_path, dxf_path)
            if success:
                logger.info(f"Successfully converted DWG to DXF: {dxf_path}")
                return dxf_path
        except Exception as e:
            logger.error(f"Error converting DWG to DXF: {e}")
        
        # If that fails, try another approach or fallback
        logger.warning("Failed to convert DWG to DXF, assuming it's already a DXF file")
        return dwg_path
    
    def _convert_dxf_to_svg(self, dxf_path: str, output_path: str) -> bool:
        """
        Convert a DXF file to SVG format.
        
        Args:
            dxf_path (str): Path to the DXF file
            output_path (str): Path where the SVG should be saved
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            if DXF2SVG_AVAILABLE:
                # Use dxf2svg library for conversion
                converter = Dxf2Svg(dxf_path)
                converter.save(output_path)
                logger.info(f"Successfully converted DXF to SVG using dxf2svg: {output_path}")
                return True
            else:
                # Fallback to manual conversion using ezdxf
                return self._manual_dxf_to_svg_conversion(dxf_path, output_path)
        except Exception as e:
            logger.error(f"Error converting DXF to SVG: {e}")
            return False
    
    def _manual_dxf_to_svg_conversion(self, dxf_path: str, output_path: str) -> bool:
        """
        Manually convert DXF to SVG using ezdxf.
        
        Args:
            dxf_path (str): Path to the DXF file
            output_path (str): Path where the SVG should be saved
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # Load DXF file
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Create a basic SVG header
            svg_content = [
                '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
            ]
            
            # Process all entities in the DXF and convert to SVG elements
            for entity in msp:
                entity_type = entity.dxftype()
                
                if entity_type == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    svg_content.append(f'<line x1="{start.x}" y1="{start.y}" x2="{end.x}" y2="{end.y}" stroke="black" />')
                
                elif entity_type == 'CIRCLE':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    svg_content.append(f'<circle cx="{center.x}" cy="{center.y}" r="{radius}" stroke="black" fill="none" />')
                
                elif entity_type == 'ARC':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle
                    
                    # Converting angles to points
                    import math
                    start_x = center.x + radius * math.cos(math.radians(start_angle))
                    start_y = center.y + radius * math.sin(math.radians(start_angle))
                    end_x = center.x + radius * math.cos(math.radians(end_angle))
                    end_y = center.y + radius * math.sin(math.radians(end_angle))
                    
                    # Check if arc is more than 180 degrees
                    large_arc = 1 if (end_angle - start_angle) % 360 > 180 else 0
                    
                    svg_content.append(
                        f'<path d="M {start_x},{start_y} A {radius},{radius} 0 {large_arc},1 {end_x},{end_y}" stroke="black" fill="none" />'
                    )
                
                elif entity_type == 'LWPOLYLINE':
                    points = entity.get_points()
                    if len(points) > 1:
                        path_data = f"M {points[0][0]},{points[0][1]}"
                        for x, y, *_ in points[1:]:
                            path_data += f" L {x},{y}"
                        if entity.closed:
                            path_data += " Z"
                        svg_content.append(f'<path d="{path_data}" stroke="black" fill="none" />')
            
            # Close SVG
            svg_content.append('</svg>')
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write('\n'.join(svg_content))
            
            logger.info(f"Successfully converted DXF to SVG manually: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error in manual DXF to SVG conversion: {e}")
            return False
    
    def count_shapes(self, svg_path: str) -> Dict[str, Any]:
        """
        Count identical shapes in an SVG file.
        
        Args:
            svg_path (str): Path to the SVG file
            
        Returns:
            dict: Count of identical shapes by shape hash
        """
        if not SVGELEMENTS_AVAILABLE:
            raise ImportError("svgelements library is required for counting shapes")
        
        logger.info(f"Counting shapes in SVG: {svg_path}")
        
        # Parse SVG file
        svg = SVG.parse(svg_path)
        
        # Dictionary to store shape information by hash
        shape_counts = collections.defaultdict(int)
        shape_info = {}
        
        # Process all elements and count them
        for element in svg.elements():
            # Skip non-shape elements
            if not isinstance(element, (Path, Shape)):
                continue
            
            # Convert to Path if it's a Shape
            if isinstance(element, Shape) and not isinstance(element, Path):
                path = Path(element)
            else:
                path = element
            
            # Generate a normalized representation for comparison
            shape_hash = self._generate_shape_hash(path)
            
            shape_counts[shape_hash] += 1
            
            # Store info about this shape type if we haven't seen it before
            if shape_hash not in shape_info:
                shape_info[shape_hash] = {
                    'sample_path': str(path),
                    'bbox': path.bbox(),
                    'color': str(path.fill) if hasattr(path, 'fill') else 'none'
                }
        
        # Create result dictionary
        result = {
            'total_shapes': sum(shape_counts.values()),
            'unique_shapes': len(shape_counts),
            'shape_counts': {},
            'shape_details': {}
        }
        
        # Add each shape type to the result
        for shape_hash, count in shape_counts.items():
            result['shape_counts'][shape_hash] = count
            result['shape_details'][shape_hash] = shape_info[shape_hash]
        
        return result
    
    def _generate_shape_hash(self, path: Path) -> str:
        """
        Generate a normalized hash for a path to identify identical shapes.
        
        Args:
            path (Path): The path to generate a hash for
            
        Returns:
            str: A hash string that represents the shape
        """
        # Get a normalized version of the path
        # We need to account for different starting points and directions
        path_copy = abs(path)  # Get a copy with all transformations applied
        
        # Normalize the path by taking its bbox and path data
        bbox = path_copy.bbox()
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Generate a simple hash based on the normalized path data
        # This is a basic implementation - could be improved for better shape matching
        segments = len(path_copy)
        
        # Check if path has isclosedac method
        if hasattr(path_copy, 'isclosedac'):
            path_type = "closed" if path_copy.isclosedac() else "open"
        elif hasattr(path_copy, 'isclosed'):
            path_type = "closed" if path_copy.isclosed() else "open"
        else:
            # Fallback if neither method is available
            path_type = "unknown"
        
        # Use aspect ratio and segment count as a simple hash
        # This can be enhanced with more sophisticated geometry analysis
        aspect_ratio = round(width / height, 3) if height != 0 else 0
        
        hash_parts = [
            f"type={path_type}",
            f"segments={segments}",
            f"ratio={aspect_ratio}",
            f"width={round(width, 2)}",
            f"height={round(height, 2)}"
        ]
        
        return ":".join(hash_parts)
    
    def classify_shapes(self, svg_path: str) -> Dict[str, Any]:
        """
        Count and classify shapes in an SVG file into categories like doors, windows, etc.
        
        Args:
            svg_path (str): Path to the SVG file
            
        Returns:
            dict: Classification and count of shapes by category
        """
        # Get raw shape counts
        shape_data = self.count_shapes(svg_path)
        
        # Initialize classification results
        classification = {
            'door': 0,
            'window': 0,
            'furniture': 0,
            'fixture': 0,
            'wall': 0,
            'column': 0,
            'security_camera': 0,
            'other': 0
        }
        
        # Simple classification rules based on shape properties
        for shape_hash, details in shape_data['shape_details'].items():
            count = shape_data['shape_counts'][shape_hash]
            bbox = details['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Get the shape's path string
            path_data = details['sample_path']
            
            # Basic classification based on shape properties
            # This is a simple example and would need to be refined based on actual drawings
            if self._is_likely_door(width, height, path_data):
                classification['door'] += count
            elif self._is_likely_window(width, height, path_data):
                classification['window'] += count
            elif self._is_likely_furniture(width, height, path_data):
                classification['furniture'] += count
            elif self._is_likely_security_camera(width, height, path_data):
                classification['security_camera'] += count
            elif self._is_likely_wall(width, height, path_data):
                classification['wall'] += count
            elif self._is_likely_column(width, height, path_data):
                classification['column'] += count
            elif self._is_likely_fixture(width, height, path_data):
                classification['fixture'] += count
            else:
                classification['other'] += count
        
        result = {
            'total_shapes': shape_data['total_shapes'],
            'unique_shapes': shape_data['unique_shapes'],
            'classifications': classification,
            'shape_details': shape_data['shape_details'],
            'shape_counts': shape_data['shape_counts']
        }
        
        return result
    
    def _is_likely_door(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely a door based on its properties."""
        # Typical doors are rectangular with an arc for the swing
        # Simple heuristic: check for rectangular shape with aspect ratio around 0.5-1.5
        # and presence of arc in the path
        door_ratio = width / height if height != 0 else 0
        is_door_ratio = 0.3 < door_ratio < 2.0
        has_arc = 'A' in path_data
        
        return is_door_ratio and (has_arc or 'Z' in path_data)
    
    def _is_likely_window(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely a window based on its properties."""
        # Windows are usually rectangular with specific aspect ratios
        window_ratio = width / height if height != 0 else 0
        is_window_ratio = 0.5 < window_ratio < 3.0
        
        # Windows typically have parallel lines and don't have arcs
        has_no_arc = 'A' not in path_data
        is_closed = 'Z' in path_data
        
        return is_window_ratio and has_no_arc and is_closed
    
    def _is_likely_security_camera(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely a security camera based on its properties."""
        # Security cameras are often small circular or dome-shaped
        is_small = width < 50 and height < 50
        is_circular = 0.8 < (width / height if height != 0 else 0) < 1.2
        
        # Cameras often have circle elements
        has_circle = ('circle' in path_data.lower()) or ('C' in path_data and is_circular)
        
        return is_small and (is_circular or has_circle)
    
    def _is_likely_furniture(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely furniture based on its properties."""
        # Furniture varies widely, but often has specific size ranges
        is_furniture_size = (50 < width < 300) and (50 < height < 300)
        
        # Many furniture items have complex shapes
        is_complex = path_data.count('L') > 4 or path_data.count('C') > 2
        
        return is_furniture_size and is_complex
    
    def _is_likely_wall(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely a wall based on its properties."""
        # Walls are typically long and thin
        wall_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0
        is_wall_ratio = wall_ratio > 5.0
        
        # Walls are usually simple straight lines
        is_simple = path_data.count('L') < 3 and 'C' not in path_data and 'A' not in path_data
        
        return is_wall_ratio and is_simple
    
    def _is_likely_column(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely a column based on its properties."""
        # Columns are usually small and square/circular
        is_column_size = width < 100 and height < 100
        is_square_ish = 0.8 < (width / height if height != 0 else 0) < 1.2
        
        # Columns are often represented as circles or simple closed shapes
        is_circular = 'A' in path_data or 'C' in path_data
        is_closed = 'Z' in path_data
        
        return is_column_size and is_square_ish and (is_circular or is_closed)
    
    def _is_likely_fixture(self, width: float, height: float, path_data: str) -> bool:
        """Check if a shape is likely a fixture based on its properties."""
        # Fixtures like lights, switches, etc. are usually small
        is_small = width < 50 and height < 50
        
        # Often represented as simple shapes
        is_simple_shape = path_data.count('L') < 4 and path_data.count('C') < 2
        
        return is_small and is_simple_shape
    
    def process_and_count(self, dwg_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process a DWG file and count the shapes in it.
        
        Args:
            dwg_path (str): Path to the DWG file
            output_dir (str, optional): Directory to save intermediate files
            
        Returns:
            dict: Count and classification of shapes
        """
        # Set up output directory
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get file name without extension
        dwg_file = os.path.basename(dwg_path)
        dwg_name = os.path.splitext(dwg_file)[0]
        
        # Convert DWG to SVG
        svg_path = self.convert_dwg_to_svg(
            dwg_path, 
            output_path=os.path.join(output_dir, f"{dwg_name}.svg")
        )
        
        # Count and classify shapes in the SVG
        results = self.classify_shapes(svg_path)
        
        # Save results to JSON file
        results_path = os.path.join(output_dir, f"{dwg_name}_shape_counts.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['svg_path'] = svg_path
        results['results_path'] = results_path
        
        return results 