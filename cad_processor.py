"""
CAD Processor Module

This module handles the processing of CAD files to extract relevant information
about geometry, layers, and other properties needed for material identification and cost estimation.
"""
import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path

# We'll use ezdxf for DXF file processing, but we'll make it optional
try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

logger = logging.getLogger(__name__)

def process_cad_file(file_path):
    """
    Process a CAD file and extract relevant information for material identification.
    
    Args:
        file_path (str): Path to the CAD file
        
    Returns:
        dict: Structured data extracted from the CAD file
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    logger.info(f"Processing CAD file: {file_path}")
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"CAD file not found: {file_path}")
    
    # Process based on file extension
    if file_extension == '.dxf':
        return process_dxf_file(file_path)
    elif file_extension == '.dwg':
        return process_dwg_file(file_path)
    elif file_extension == '.ifc':
        return process_ifc_file(file_path)
    elif file_extension in ['.json', '.cadme']:
        return process_json_file(file_path)
    else:
        raise ValueError(f"Unsupported CAD file format: {file_extension}")

def process_dxf_file(file_path):
    """
    Process a DXF file using ezdxf.
    
    Args:
        file_path (str): Path to the DXF file
        
    Returns:
        dict: Structured data extracted from the DXF file
    """
    if not EZDXF_AVAILABLE:
        logger.error("ezdxf library is required for processing DXF files")
        raise ImportError("ezdxf library is required for processing DXF files. Install with 'pip install ezdxf'")
    
    logger.info(f"Processing DXF file: {file_path}")
    
    try:
        # Open the DXF file
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        # Extract basic document information
        cad_data = {
            "file_info": {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": "dxf",
                "version": doc.dxfversion,
                "encoding": doc.encoding,
            },
            "entities": [],
            "layers": {},
            "blocks": {},
            "metadata": {
                "units": get_dxf_units(doc),
                "dimension_style": "metric" if get_dxf_units(doc) in ["mm", "cm", "m"] else "imperial"
            }
        }
        
        # Extract layer information
        for layer in doc.layers:
            layer_name = layer.dxf.name
            cad_data["layers"][layer_name] = {
                "name": layer_name,
                "color": layer.dxf.color,
                "linetype": layer.dxf.linetype,
                "is_on": layer.is_on,
                "is_locked": layer.is_locked,
                "entities_count": 0,
                "area": 0.0,
                "volume": 0.0,
                "length": 0.0,
                "material_hint": extract_material_hint_from_layer_name(layer_name)
            }
        
        # Process entities in modelspace
        for entity in msp:
            entity_data = extract_entity_data(entity)
            if entity_data:
                layer_name = entity_data.get("layer", "0")
                if layer_name in cad_data["layers"]:
                    cad_data["layers"][layer_name]["entities_count"] += 1
                    
                    # Accumulate length, area, volume
                    if "length" in entity_data:
                        cad_data["layers"][layer_name]["length"] += entity_data["length"]
                    if "area" in entity_data:
                        cad_data["layers"][layer_name]["area"] += entity_data["area"]
                    if "volume" in entity_data:
                        cad_data["layers"][layer_name]["volume"] += entity_data["volume"]
                        
                cad_data["entities"].append(entity_data)
        
        # Extract block information
        for block in doc.blocks:
            # Check if it's a paperspace block
            is_paperspace = getattr(block, "is_any_paperspace", False)
            # Instead of using is_xref attribute, check if the block name starts with '*' (xref indicator)
            is_xref = block.name.startswith('*')
            
            if not is_paperspace and not is_xref:
                block_name = block.name
                cad_data["blocks"][block_name] = {
                    "name": block_name,
                    "entities_count": len(list(block)),
                    "material_hint": extract_material_hint_from_layer_name(block_name)
                }
        
        logger.info(f"DXF processing complete. Found {len(cad_data['entities'])} entities across {len(cad_data['layers'])} layers.")
        return ensure_json_serializable(cad_data)
    
    except ezdxf.DXFError as e:
        logger.error(f"Error reading DXF file: {e}")
        raise ValueError(f"Invalid DXF file: {e}")
    except Exception as e:
        logger.error(f"Error processing DXF file: {e}", exc_info=True)
        raise

def process_dwg_file(file_path):
    """
    Process a DWG file by first converting it to DXF and then processing the DXF.
    
    Args:
        file_path (str): Path to the DWG file
        
    Returns:
        dict: Structured data extracted from the DWG file
    """
    file_path = Path(file_path)
    logger.info(f"Processing DWG file: {file_path}")
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"DWG file not found: {file_path}")
    
    # Create a temporary folder for the DXF conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dxf_path = temp_dir_path / f"{file_path.stem}.dxf"
        
        # Try to convert DWG to DXF using ODA File Converter
        try:
            result = convert_dwg_to_dxf(file_path, output_dxf_path)
            if result and output_dxf_path.exists():
                # If conversion is successful, process the DXF file
                logger.info(f"Successfully converted DWG to DXF: {output_dxf_path}")
                cad_data = process_dxf_file(output_dxf_path)
                
                # Update file info to show it came from a DWG
                cad_data["file_info"]["original_file_name"] = file_path.name
                cad_data["file_info"]["original_file_path"] = str(file_path)
                cad_data["file_info"]["original_file_type"] = "dwg"
                
                return cad_data
            else:
                logger.warning("DWG to DXF conversion failed, using placeholder implementation")
        except Exception as e:
            logger.error(f"Error converting DWG to DXF: {e}", exc_info=True)
            logger.warning("DWG to DXF conversion failed, using placeholder implementation")
    
    # If conversion fails, return placeholder data
    placeholder_data = {
        "file_info": {
            "file_name": Path(file_path).name,
            "file_path": str(file_path),
            "file_type": "dwg",
            "version": "unknown",
        },
        "entities": [],
        "layers": {},
        "blocks": {},
        "metadata": {
            "note": "DWG processing is limited. Consider converting to DXF for full feature support."
        }
    }
    
    return ensure_json_serializable(placeholder_data)

def convert_dwg_to_dxf(input_path, output_path):
    """
    Convert a DWG file to DXF format using ODA File Converter.
    
    Args:
        input_path (Path): Path to the DWG file
        output_path (Path): Path where the DXF file will be saved
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    logger.info(f"Converting DWG to DXF: {input_path} -> {output_path}")
    
    # Try to find ODA File Converter
    oda_converter_paths = [
        "C:/Program Files/ODA/ODAFileConverter/ODAFileConverter.exe",
        "C:/Program Files (x86)/ODA/ODAFileConverter/ODAFileConverter.exe",
        "C:/Program Files/Open Design Alliance/ODAFileConverter/ODAFileConverter.exe",
        "C:/Program Files (x86)/Open Design Alliance/ODAFileConverter/ODAFileConverter.exe"
    ]
    
    oda_converter = None
    for path in oda_converter_paths:
        if os.path.exists(path):
            oda_converter = path
            break
    
    if not oda_converter:
        logger.error("ODA File Converter not found. Please install it and add to PATH.")
        return False
    
    # Create input and output directories
    input_dir = str(input_path.parent.absolute())
    output_dir = str(output_path.parent.absolute())
    
    # Prepare command arguments
    # ODAFileConverter "input_folder" "output_folder" version type recurse audit
    cmd = [
        oda_converter,
        input_dir,              # Input folder
        output_dir,             # Output folder
        "ACAD2018",             # Output version
        "DXF",                  # Output format
        "0",                    # Recursive (0=no, 1=yes)
        "1"                     # Audit (0=no, 1=yes)
    ]
    
    try:
        # Run the conversion process
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        # Call ODA File Converter
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        logger.debug(f"Conversion output: {result.stdout}")
        
        # Check if output file exists after conversion
        if output_path.exists():
            logger.info(f"DWG to DXF conversion successful: {output_path}")
            return True
        else:
            logger.warning(f"Output DXF file not found after conversion: {output_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"ODA File Converter returned an error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error in DWG to DXF conversion: {e}", exc_info=True)
        return False

def process_ifc_file(file_path):
    """
    Process an IFC file.
    
    Args:
        file_path (str): Path to the IFC file
        
    Returns:
        dict: Structured data extracted from the IFC file
    """
    logger.warning("IFC processing not fully implemented. Using placeholder implementation.")
    
    # This is a placeholder - IFC processing would require additional libraries like ifcopenshell
    
    placeholder_data = {
        "file_info": {
            "file_name": Path(file_path).name,
            "file_path": str(file_path),
            "file_type": "ifc",
            "version": "unknown",
        },
        "entities": [],
        "layers": {},
        "elements": {},
        "metadata": {
            "note": "IFC processing is limited. Full IFC support requires the ifcopenshell library."
        }
    }
    
    return ensure_json_serializable(placeholder_data)

def process_json_file(file_path):
    """
    Process a JSON file that contains pre-processed CAD data in our format.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Structured data from the JSON file
    """
    logger.info(f"Processing JSON CAD data file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate basic structure
        required_keys = ["file_info", "entities", "layers"]
        for key in required_keys:
            if key not in data:
                logger.warning(f"JSON CAD data missing required key: {key}")
                data[key] = {}
        
        return ensure_json_serializable(data)
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        raise ValueError(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"Error processing JSON file: {e}", exc_info=True)
        raise

def extract_entity_data(entity):
    """
    Extract relevant data from a DXF entity.
    
    Args:
        entity: A DXF entity object from ezdxf
        
    Returns:
        dict: Structured data for the entity
    """
    try:
        entity_type = entity.dxftype()
        entity_data = {
            "id": getattr(entity, "handle", "unknown"),
            "type": entity_type,
            "layer": entity.dxf.layer,
            "color_index": getattr(entity.dxf, "color", 7),  # 7 is white/black
            "linetype": getattr(entity.dxf, "linetype", "CONTINUOUS"),
        }
        
        # Extract geometry information based on entity type
        if entity_type == "LINE":
            entity_data["start_point"] = tuple(entity.dxf.start)
            entity_data["end_point"] = tuple(entity.dxf.end)
            # Calculate length
            import numpy as np
            start = np.array(entity.dxf.start)
            end = np.array(entity.dxf.end)
            entity_data["length"] = float(np.linalg.norm(end - start))
            
        elif entity_type == "CIRCLE":
            entity_data["center"] = tuple(entity.dxf.center)
            entity_data["radius"] = entity.dxf.radius
            # Calculate area
            import math
            entity_data["area"] = math.pi * entity.dxf.radius ** 2
            entity_data["length"] = 2 * math.pi * entity.dxf.radius  # Circumference
            
        elif entity_type == "ARC":
            entity_data["center"] = tuple(entity.dxf.center)
            entity_data["radius"] = entity.dxf.radius
            entity_data["start_angle"] = entity.dxf.start_angle
            entity_data["end_angle"] = entity.dxf.end_angle
            # Calculate arc length
            import math
            angle_diff = abs(entity.dxf.end_angle - entity.dxf.start_angle)
            if angle_diff > 360:
                angle_diff = angle_diff % 360
            entity_data["length"] = math.radians(angle_diff) * entity.dxf.radius
            
        elif entity_type == "LWPOLYLINE":
            entity_data["is_closed"] = entity.is_closed
            entity_data["points"] = [tuple(point) for point in entity.get_points()]
            
            # Calculate length and area if closed
            if entity.has_width:
                # This is an approximation - accurate calculation requires more work
                entity_data["has_width"] = True
                entity_data["width"] = entity.dxf.const_width if hasattr(entity.dxf, "const_width") else 0
            
            # Calculate length manually
            points = list(entity.get_points())
            if len(points) > 1:
                length = 0
                import numpy as np
                for i in range(len(points)-1):
                    p1 = np.array(points[i][:2])  # Take only x, y coordinates
                    p2 = np.array(points[i+1][:2])
                    length += np.linalg.norm(p2 - p1)
                
                # Add closing segment if closed
                if entity.is_closed and len(points) > 2:
                    p1 = np.array(points[-1][:2])
                    p2 = np.array(points[0][:2])
                    length += np.linalg.norm(p2 - p1)
                
                entity_data["length"] = float(length)
            else:
                entity_data["length"] = 0.0
            
            # Calculate area if closed
            if entity.is_closed and len(points) > 2:
                # Simple approximation for area using shoelace formula
                try:
                    import numpy as np
                    x = np.array([p[0] for p in points])
                    y = np.array([p[1] for p in points])
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    entity_data["area"] = float(area)
                except Exception as e:
                    logger.warning(f"Error calculating area for polyline: {e}")
                    pass
            
        elif entity_type == "TEXT" or entity_type == "MTEXT":
            entity_data["text"] = entity.dxf.text if hasattr(entity.dxf, "text") else "?"
            entity_data["position"] = tuple(entity.dxf.insert) if hasattr(entity.dxf, "insert") else (0, 0, 0)
            entity_data["height"] = entity.dxf.height if hasattr(entity.dxf, "height") else 1.0
            
        elif entity_type == "INSERT":
            entity_data["block_name"] = entity.dxf.name
            entity_data["position"] = tuple(entity.dxf.insert)
            entity_data["scale"] = (
                getattr(entity.dxf, "xscale", 1.0),
                getattr(entity.dxf, "yscale", 1.0),
                getattr(entity.dxf, "zscale", 1.0)
            )
            entity_data["rotation"] = getattr(entity.dxf, "rotation", 0.0)
            
        # Add more entity types as needed
        
        return entity_data
    
    except Exception as e:
        logger.warning(f"Error extracting data from entity: {e}")
        return None

def get_dxf_units(doc):
    """
    Get the units used in the DXF document.
    
    Args:
        doc: DXF document
        
    Returns:
        str: Unit type (e.g., 'mm', 'inches')
    """
    try:
        # Get units from header variable $INSUNITS
        units_code = doc.header.get('$INSUNITS', 0)
        
        # Map units code to string representation
        units_map = {
            0: "unitless",
            1: "inches",
            2: "feet",
            3: "miles",
            4: "mm",
            5: "cm",
            6: "m",
            7: "km",
            8: "microinches",
            9: "mils",
            10: "yards",
            11: "angstroms",
            12: "nanometers",
            13: "microns",
            14: "decimeters",
            15: "decameters",
            16: "hectometers",
            17: "gigameters",
            18: "au",
            19: "light_years",
            20: "parsecs"
        }
        
        return units_map.get(units_code, "unknown")
    
    except Exception as e:
        logger.warning(f"Error getting DXF units: {e}")
        return "unknown"

def extract_material_hint_from_layer_name(layer_name):
    """
    Extract material hints from layer names based on common naming conventions.
    
    Args:
        layer_name (str): Name of the layer
        
    Returns:
        dict: Material hint information
    """
    layer_name = layer_name.lower()
    
    # Define common layer name patterns for materials
    material_patterns = {
        "concrete": ["conc", "concrete", "foundation", "footing", "slab", "column"],
        "steel": ["steel", "metal", "stl", "beam", "column", "framing", "struct"],
        "wood": ["wood", "timber", "framing", "stud", "joist", "rafter", "wdn"],
        "masonry": ["brick", "block", "cmu", "masonry", "stone"],
        "glass": ["glass", "glazing", "window", "curtainwall"],
        "drywall": ["drywall", "gypsum", "gyp", "gwb", "partition"],
        "insulation": ["insul", "insulation", "thermal"],
        "finish": ["finish", "paint", "coating", "flooring", "ceiling", "tile"],
        "roof": ["roof", "roofing", "shingle", "membrane"],
        "door": ["door", "entry"],
        "plumbing": ["plumb", "plumbing", "pipe", "water", "sewer", "drain"],
        "hvac": ["hvac", "mech", "mechanical", "duct", "equipment"],
        "electrical": ["elec", "electrical", "power", "lighting", "conduit"],
        "site": ["site", "grading", "landscape", "paving", "asphalt", "concrete"],
        "furniture": ["furn", "furniture", "fixture", "cabinet", "casework"],
    }
    
    # Check if layer name contains any material patterns
    matched_materials = []
    for material, patterns in material_patterns.items():
        for pattern in patterns:
            if pattern in layer_name:
                matched_materials.append(material)
                break
    
    # If multiple matches found, prioritize based on specificity
    # This is a simple approach and can be refined
    if len(matched_materials) > 1:
        # Check if material name is directly in the layer name
        for material in matched_materials:
            if material in layer_name:
                return {
                    "material": material,
                    "confidence": "medium",
                    "source": "layer_name_direct_match"
                }
        
        # If no direct match, return first match with low confidence
        return {
            "material": matched_materials[0],
            "confidence": "low",
            "source": "layer_name_partial_match"
        }
    
    elif len(matched_materials) == 1:
        return {
            "material": matched_materials[0],
            "confidence": "medium",
            "source": "layer_name_match"
        }
    
    # Check for dimension hints in layer name
    dimension_patterns = {
        "2d": ["2d", "plan", "elevation", "section", "detail"],
        "3d": ["3d", "model", "isometric", "axon"]
    }
    
    for dim, patterns in dimension_patterns.items():
        for pattern in patterns:
            if pattern in layer_name:
                return {
                    "dimension": dim,
                    "confidence": "medium",
                    "source": "layer_name_dimension_hint"
                }
    
    # No material hint found
    return {
        "material": "unknown",
        "confidence": "none",
        "source": "no_match"
    }

def ensure_json_serializable(obj):
    """
    Recursively convert an object to ensure it's JSON serializable.
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON serializable object
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return list(ensure_json_serializable(item) for item in obj)
    elif callable(obj):
        return str(obj)  # Convert callable (method, function) to string
    elif hasattr(obj, '__dict__'):
        return str(obj)  # Convert custom objects to string
    else:
        return obj 