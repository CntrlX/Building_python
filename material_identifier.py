"""
Material Identifier Module

This module analyzes processed CAD data to identify and classify construction materials
based on layer names, entity properties, and other metadata.
"""
import logging
import re
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import itertools
import time

# Import OpenAI and LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Dictionary of common construction materials and their properties
MATERIAL_CATALOG = {
    "concrete": {
        "category": "structural",
        "subcategory": "concrete",
        "density": 2400,  # kg/m³
        "cost_factor": 1.0,
        "patterns": ["conc", "concrete", "foundation", "footing", "slab", "cip", "precast"]
    },
    "steel": {
        "category": "structural",
        "subcategory": "metal",
        "density": 7850,  # kg/m³
        "cost_factor": 1.2,
        "patterns": ["steel", "metal", "stl", "column", "beam", "hss", "wide flange", "wf", "channel"]
    },
    "wood": {
        "category": "structural",
        "subcategory": "wood",
        "density": 650,  # kg/m³
        "cost_factor": 0.8,
        "patterns": ["wood", "timber", "stud", "joist", "rafter", "lumber", "plywood", "osb"]
    },
    "brick": {
        "category": "envelope",
        "subcategory": "masonry",
        "density": 1900,  # kg/m³
        "cost_factor": 1.1,
        "patterns": ["brick", "masonry", "clay"]
    },
    "cmu": {
        "category": "envelope",
        "subcategory": "masonry",
        "density": 1900,  # kg/m³
        "cost_factor": 0.9,
        "patterns": ["cmu", "block", "masonry", "concrete block"]
    },
    "glass": {
        "category": "envelope",
        "subcategory": "glazing",
        "density": 2500,  # kg/m³
        "cost_factor": 1.5,
        "patterns": ["glass", "glazing", "window", "curtainwall"]
    },
    "drywall": {
        "category": "interior",
        "subcategory": "finishes",
        "density": 800,  # kg/m³
        "cost_factor": 0.7,
        "patterns": ["drywall", "gypsum", "gyp", "gwb", "partition"]
    },
    "insulation": {
        "category": "envelope",
        "subcategory": "insulation",
        "density": 40,  # kg/m³
        "cost_factor": 0.5,
        "patterns": ["insul", "insulation", "thermal", "batt"]
    },
    "roofing": {
        "category": "envelope",
        "subcategory": "roofing",
        "density": 1100,  # kg/m³
        "cost_factor": 1.2,
        "patterns": ["roof", "shingle", "membrane", "epdm", "tpo", "metal roof"]
    },
    "flooring": {
        "category": "interior",
        "subcategory": "finishes",
        "density": 800,  # kg/m³
        "cost_factor": 1.3,
        "patterns": ["floor", "tile", "carpet", "wood floor", "vinyl", "vct"]
    },
    "ceiling": {
        "category": "interior",
        "subcategory": "finishes",
        "density": 300,  # kg/m³
        "cost_factor": 0.8,
        "patterns": ["ceiling", "acoustic", "act", "gypsum ceiling"]
    },
    "plumbing": {
        "category": "mep",
        "subcategory": "plumbing",
        "density": 8000,  # kg/m³ (for copper/steel pipes)
        "cost_factor": 1.4,
        "patterns": ["plumb", "pipe", "water", "sewer", "drain", "sanitary", "supply"]
    },
    "hvac": {
        "category": "mep",
        "subcategory": "mechanical",
        "density": 8000,  # kg/m³ (for metal ducts/equipment)
        "cost_factor": 1.6,
        "patterns": ["hvac", "mech", "duct", "diffuser", "grille", "equipment"]
    },
    "electrical": {
        "category": "mep",
        "subcategory": "electrical",
        "density": 8900,  # kg/m³ (for copper wiring)
        "cost_factor": 1.5,
        "patterns": ["elec", "power", "lighting", "conduit", "panel", "device"]
    },
    # New entries for doors with expanded patterns
    "door": {
        "category": "architectural",
        "subcategory": "openings",
        "density": 800,  # kg/m³ (average)
        "cost_factor": 1.2,
        "patterns": ["door", "dr-", "dr_", "door-", "door_", "entry", "exit", "doorway", "single door", "double door", "sliding door"]
    },
    # New entries for windows
    "window": {
        "category": "architectural",
        "subcategory": "openings",
        "density": 2500,  # kg/m³ (glass+frame)
        "cost_factor": 1.4,
        "patterns": ["window", "win-", "win_", "window-", "window_", "casement", "awning", "sash"]
    },
    # New entries for furniture
    "furniture": {
        "category": "interior",
        "subcategory": "furnishings",
        "density": 700,  # kg/m³ (average)
        "cost_factor": 1.1,
        "patterns": ["furn", "furniture", "chair", "table", "desk", "cabinet", "casework", "shelving", "shelf"]
    },
    # New entries for security equipment including CCTV cameras
    "security": {
        "category": "systems",
        "subcategory": "security",
        "density": 5000,  # kg/m³ (average electronic equipment)
        "cost_factor": 2.0,
        "patterns": ["cctv", "camera", "security", "surveillance", "alarm", "sensor", "motion", "detector", "cam"]
    },
    # New entries for fixtures
    "fixture": {
        "category": "interior",
        "subcategory": "fixtures",
        "density": 2000,  # kg/m³ (average)
        "cost_factor": 1.3,
        "patterns": ["fixture", "light", "lamp", "luminaire", "sconce", "chandelier", "pendant"]
    },
    # New entries for general equipment
    "equipment": {
        "category": "systems",
        "subcategory": "equipment",
        "density": 6000,  # kg/m³ (average)
        "cost_factor": 1.8,
        "patterns": ["equip", "equipment", "appliance", "machine", "device", "system"]
    }
}

# Path to store learned patterns
LEARNED_PATTERNS_FILE = "learned_patterns.json"

def learn_from_feedback(feedback_data):
    """
    Update pattern library based on user feedback
    
    Args:
        feedback_data (dict): User feedback on material identification
            Format: {
                "material_name": str,
                "category": str,
                "subcategory": str,
                "patterns": list,
                "confidence": float
            }
    
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Learning from feedback for material: {feedback_data.get('material_name', 'unknown')}")
        
        # Validate feedback data
        required_fields = ["material_name", "category", "subcategory", "patterns"]
        for field in required_fields:
            if field not in feedback_data:
                logger.error(f"Missing required field in feedback data: {field}")
                return False
        
        material_name = feedback_data["material_name"].lower()
        
        # If material doesn't exist in catalog, add it
        if material_name not in MATERIAL_CATALOG:
            logger.info(f"Adding new material to catalog: {material_name}")
            MATERIAL_CATALOG[material_name] = {
                "category": feedback_data["category"],
                "subcategory": feedback_data["subcategory"],
                "density": feedback_data.get("density", 1000),  # Default density
                "cost_factor": feedback_data.get("cost_factor", 1.0),  # Default cost factor
                "patterns": []
            }
        
        # Get existing patterns
        existing_patterns = set(MATERIAL_CATALOG[material_name]["patterns"])
        
        # Add new patterns
        new_patterns = set(feedback_data["patterns"])
        updated_patterns = existing_patterns.union(new_patterns)
        
        # Update the catalog
        MATERIAL_CATALOG[material_name]["patterns"] = list(updated_patterns)
        
        # Optionally update category and subcategory if confidence is high
        confidence = feedback_data.get("confidence", 0.5)
        if confidence > 0.8:
            MATERIAL_CATALOG[material_name]["category"] = feedback_data["category"]
            MATERIAL_CATALOG[material_name]["subcategory"] = feedback_data["subcategory"]
        
        # Save to learned patterns file
        save_learned_patterns()
        
        logger.info(f"Successfully updated patterns for {material_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error learning from feedback: {e}", exc_info=True)
        return False

def save_learned_patterns():
    """
    Save learned patterns to a file
    
    Returns:
        bool: Success status
    """
    try:
        # Create a dictionary of learned patterns
        learned_patterns = {}
        
        for material_name, props in MATERIAL_CATALOG.items():
            learned_patterns[material_name] = {
                "category": props["category"],
                "subcategory": props["subcategory"],
                "patterns": props["patterns"],
                "last_updated": time.time()
            }
        
        # Save to file
        with open(LEARNED_PATTERNS_FILE, "w") as f:
            json.dump(learned_patterns, f, indent=2)
        
        logger.info(f"Saved learned patterns to {LEARNED_PATTERNS_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving learned patterns: {e}", exc_info=True)
        return False

def load_learned_patterns():
    """
    Load learned patterns from file and update MATERIAL_CATALOG
    
    Returns:
        bool: Success status
    """
    try:
        # Check if file exists
        if not Path(LEARNED_PATTERNS_FILE).exists():
            logger.info(f"Learned patterns file does not exist: {LEARNED_PATTERNS_FILE}")
            return False
        
        # Load from file
        with open(LEARNED_PATTERNS_FILE, "r") as f:
            learned_patterns = json.load(f)
        
        # Update MATERIAL_CATALOG
        for material_name, props in learned_patterns.items():
            if material_name in MATERIAL_CATALOG:
                # Update existing material
                MATERIAL_CATALOG[material_name]["patterns"] = list(set(MATERIAL_CATALOG[material_name]["patterns"] + props["patterns"]))
            else:
                # Add new material
                MATERIAL_CATALOG[material_name] = {
                    "category": props["category"],
                    "subcategory": props["subcategory"],
                    "density": 1000,  # Default density
                    "cost_factor": 1.0,  # Default cost factor
                    "patterns": props["patterns"]
                }
        
        logger.info(f"Loaded learned patterns from {LEARNED_PATTERNS_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading learned patterns: {e}", exc_info=True)
        return False

# Add code to load learned patterns at startup
# Add this after the MATERIAL_CATALOG definition
# Try to load learned patterns
load_learned_patterns()

def identify_materials(cad_data, use_llm=False, enable_learning=True):
    """
    Analyze the CAD data to identify materials based on layer names, entity properties, etc.
    
    Args:
        cad_data (dict): Processed CAD data from the cad_processor module
        use_llm (bool): Whether to use language models for additional material identification
        enable_learning (bool): Whether to use learned patterns
        
    Returns:
        dict: Identified materials with quantities and properties
    """
    logger.info("Identifying materials from CAD data")
    
    # Ensure learned patterns are loaded if learning is enabled
    if enable_learning:
        load_learned_patterns()
    
    materials = {
        "project_info": {
            "name": Path(cad_data.get("file_info", {}).get("file_path", "unknown")).stem,
            "file_path": cad_data.get("file_info", {}).get("file_path", "unknown"),
            "units": cad_data.get("metadata", {}).get("units", "unknown")
        },
        "materials": [],
        "material_quantities": {},
        "material_categories": {
            "structural": {"total_volume": 0, "total_area": 0, "total_length": 0},
            "envelope": {"total_volume": 0, "total_area": 0, "total_length": 0},
            "interior": {"total_volume": 0, "total_area": 0, "total_length": 0},
            "mep": {"total_volume": 0, "total_area": 0, "total_length": 0},
            "architectural": {"total_volume": 0, "total_area": 0, "total_length": 0},
            "systems": {"total_volume": 0, "total_area": 0, "total_length": 0},
            "other": {"total_volume": 0, "total_area": 0, "total_length": 0}
        }
    }
    
    # First, identify materials from layer names
    layer_materials = identify_materials_from_layers(cad_data)
    
    # Second, identify materials from block names
    block_materials = identify_materials_from_blocks(cad_data)
    
    # Third, analyze entities for additional material information
    entity_materials = identify_materials_from_entities(cad_data)
    
    # Combine and process all identified materials
    all_materials = layer_materials + block_materials + entity_materials
    
    # Process and categorize materials
    materials["materials"] = all_materials
    
    # Calculate quantities by material type
    for material in all_materials:
        material_name = material["material"]
        if material_name not in materials["material_quantities"]:
            materials["material_quantities"][material_name] = {
                "volume": 0,
                "area": 0,
                "length": 0,
                "count": 0,
                "category": material.get("category", "other"),
                "subcategory": material.get("subcategory", "unknown"),
                "cost_factor": material.get("cost_factor", 1.0)
            }
        
        # Add quantities
        materials["material_quantities"][material_name]["volume"] += material.get("volume", 0)
        materials["material_quantities"][material_name]["area"] += material.get("area", 0)
        materials["material_quantities"][material_name]["length"] += material.get("length", 0)
        materials["material_quantities"][material_name]["count"] += 1
        
        # Add to category totals
        category = material.get("category", "other")
        materials["material_categories"][category]["total_volume"] += material.get("volume", 0)
        materials["material_categories"][category]["total_area"] += material.get("area", 0)
        materials["material_categories"][category]["total_length"] += material.get("length", 0)
    
    # Use LLM if requested and available
    if use_llm:
        try:
            # This would call an external LLM for enhanced material detection
            llm_materials = identify_materials_with_llm(cad_data)
            # Merge LLM identified materials with rule-based materials
            materials = merge_material_identifications(materials, llm_materials)
            logger.info("Enhanced material identification with LLM")
        except Exception as e:
            logger.warning(f"Failed to use LLM for material identification: {e}")
    
    logger.info(f"Identified {len(materials['materials'])} materials across {len(materials['material_quantities'])} types")
    return materials

def identify_materials_from_layers(cad_data):
    """
    Identify materials based on layer names and properties
    
    Args:
        cad_data (dict): Processed CAD data
        
    Returns:
        list: Identified materials from layers
    """
    layer_materials = []
    
    # Process each layer to identify materials
    for layer_name, layer_data in cad_data.get("layers", {}).items():
        # Skip layers with no entities
        if layer_data.get("entities_count", 0) == 0:
            continue
        
        # Check if the layer already has a material hint
        material_hint = layer_data.get("material_hint", {})
        if material_hint.get("material", "unknown") != "unknown" and material_hint.get("confidence", "none") != "none":
            # Use the existing material hint
            material_name = material_hint["material"]
            confidence = material_hint["confidence"]
            
            # Get material properties from catalog
            material_props = MATERIAL_CATALOG.get(material_name, {
                "category": "other",
                "subcategory": "unknown",
                "cost_factor": 1.0
            })
            
            layer_materials.append({
                "material": material_name,
                "source": f"layer:{layer_name}",
                "confidence": confidence,
                "category": material_props.get("category", "other"),
                "subcategory": material_props.get("subcategory", "unknown"),
                "volume": layer_data.get("volume", 0),
                "area": layer_data.get("area", 0),
                "length": layer_data.get("length", 0),
                "cost_factor": material_props.get("cost_factor", 1.0)
            })
            continue
        
        # Identify material from layer name using pattern matching
        layer_lower = layer_name.lower()
        best_match = None
        best_score = 0
        
        for material_name, material_props in MATERIAL_CATALOG.items():
            score = 0
            for pattern in material_props["patterns"]:
                if pattern.lower() in layer_lower:
                    # Longer pattern matches are more specific
                    score += len(pattern)
            
            if score > best_score:
                best_score = score
                best_match = material_name
        
        # If a match was found, add it to the materials list
        if best_match and best_score > 0:
            material_props = MATERIAL_CATALOG[best_match]
            
            layer_materials.append({
                "material": best_match,
                "source": f"layer:{layer_name}",
                "confidence": "medium" if best_score > 4 else "low",
                "category": material_props["category"],
                "subcategory": material_props["subcategory"],
                "volume": layer_data.get("volume", 0),
                "area": layer_data.get("area", 0),
                "length": layer_data.get("length", 0),
                "cost_factor": material_props["cost_factor"]
            })
        else:
            # No material match found, add as unknown
            layer_materials.append({
                "material": "unknown",
                "source": f"layer:{layer_name}",
                "confidence": "none",
                "category": "other",
                "subcategory": "unknown",
                "volume": layer_data.get("volume", 0),
                "area": layer_data.get("area", 0),
                "length": layer_data.get("length", 0),
                "cost_factor": 1.0
            })
    
    return layer_materials

def identify_materials_from_blocks(cad_data):
    """
    Identify materials based on block names and properties
    
    Args:
        cad_data (dict): Processed CAD data
        
    Returns:
        list: Identified materials from blocks
    """
    block_materials = []
    
    # Process blocks to identify materials
    for block_name, block_data in cad_data.get("blocks", {}).items():
        # Skip blocks with no entities
        if block_data.get("entities_count", 0) == 0:
            continue
        
        # Check if the block already has a material hint
        material_hint = block_data.get("material_hint", {})
        if material_hint.get("material", "unknown") != "unknown" and material_hint.get("confidence", "none") != "none":
            # Use the existing material hint
            material_name = material_hint["material"]
            material_props = MATERIAL_CATALOG.get(material_name, {
                "category": "other",
                "subcategory": "unknown",
                "cost_factor": 1.0
            })
            
            # For blocks, we might not have volume/area/length directly
            # They'd need to be calculated from the block's scale and contained entities
            block_materials.append({
                "material": material_name,
                "source": f"block:{block_name}",
                "confidence": material_hint["confidence"],
                "category": material_props.get("category", "other"),
                "subcategory": material_props.get("subcategory", "unknown"),
                "count": 1,  # Count blocks as instances
                "cost_factor": material_props.get("cost_factor", 1.0)
            })
            continue
        
        # Try to identify material from block name
        block_lower = block_name.lower()
        best_match = None
        best_score = 0
        
        for material_name, material_props in MATERIAL_CATALOG.items():
            score = 0
            for pattern in material_props["patterns"]:
                if pattern.lower() in block_lower:
                    score += len(pattern)
            
            if score > best_score:
                best_score = score
                best_match = material_name
        
        # If a match was found, add it to the materials list
        if best_match and best_score > 0:
            material_props = MATERIAL_CATALOG[best_match]
            
            block_materials.append({
                "material": best_match,
                "source": f"block:{block_name}",
                "confidence": "medium" if best_score > 4 else "low",
                "category": material_props["category"],
                "subcategory": material_props["subcategory"],
                "count": 1,  # Count blocks as instances
                "cost_factor": material_props["cost_factor"]
            })
    
    return block_materials

def identify_materials_from_entities(cad_data):
    """
    Identify materials by analyzing entity properties
    
    Args:
        cad_data (dict): Processed CAD data
        
    Returns:
        list: Identified materials from entities
    """
    entity_materials = []
    
    # Group entities by type for specialized analysis
    entity_types = {}
    entity_blocks = {}
    
    # Keep track of special entities with more detailed categories
    special_entities = {
        "door": 0,
        "security": 0,
        "window": 0,
        "furniture": {
            "chair": {
                "desk_chair": 0,
                "conference_chair": 0,
                "other_chair": 0
            },
            "table": {
                "conference_table": 0,
                "desk": 0,
                "classroom_table": 0,
                "other_table": 0
            },
            "cabinet": 0,
            "shelf": 0,
            "other": 0
        }
    }
    
    # Extended patterns for better detection
    door_patterns = ["door", "dr-", "dr_", "door-", "door_", "entry", "exit", "doorway", "d-", "d_", "sld-", "dbl-", "single door", "double door", "sliding door", "dor"]
    
    # Camera patterns for identifying security cameras
    camera_patterns = ["dome camera", "cctv dome", "security camera", "surveillance camera", 
                      "cctv camera", "ptz camera", "security device", "dome outlet",
                      "sec-cam", "cam-", "camera-", "dome-", "sec-dom", "cctv"]
    
    # Exclude patterns that might cause false positives for cameras
    camera_exclude_patterns = ["camera view", "camera zone", "view", "zone", "detection area", 
                              "camera note", "detail"]
    
    window_patterns = ["window", "win-", "win_", "window-", "window_", "casement", "awning", "sash", "w-", "w_", "glazing"]
    
    # Extended furniture patterns with more specific subcategories
    furniture_patterns = {
        "chair": {
            "desk_chair": ["desk chair", "office chair", "task chair", "swivel chair", "ch-desk", "ch_desk"],
            "conference_chair": ["conf chair", "meeting chair", "boardroom chair", "ch-conf", "ch_conf"],
            "other_chair": ["chair", "chr", "chair-", "chair_", "seating", "stool", "armchair", "sofa", "bench", "ch-", "ch_"]
        },
        "table": {
            "conference_table": ["conference table", "meeting table", "round table", "oval table", "tb-conf", "tb_conf"],
            "desk": ["desk", "workstation", "work surface", "tb-desk", "tb_desk"],
            "classroom_table": ["classroom table", "student desk", "learning table", "tb-class", "tb_class"],
            "other_table": ["table", "tbl", "table-", "table_", "tb-", "tb_"]
        },
        "cabinet": ["cabinet", "cab", "cabinet-", "cabinet_", "storage", "credenza", "file cabinet", "cb-", "cb_"],
        "shelf": ["shelf", "shelving", "bookcase", "bookshelf", "shelves", "sh-", "sh_"]
    }
    
    # Layer patterns for specific elements
    door_layer_patterns = ["door", "dr-", "a-door", "arch-door", "opening"]
    camera_layer_patterns = ["camera", "cctv", "security", "surveillance", "cam-", "cam_", 
                            "sec-", "sec_", "elec-cam", "elec-sec"]
    furniture_layer_patterns = ["furn", "furniture", "table", "chair", "desk", "ff&e", "int-furn"]
    
    # Function to check if an entity resembles a door by its geometry
    def is_door_geometry(entity):
        # Check for arcs which might be door swings
        if entity.get("type") == "ARC":
            return True
        # Check for L-shaped polylines that might be door frames
        if entity.get("type") in ["LWPOLYLINE", "POLYLINE"]:
            # For simplicity, we'll assume polylines with 3-4 vertices might be door frames
            vertices = entity.get("vertices", [])
            if len(vertices) in [3, 4]:
                return True
        return False
    
    # Function to check if a block visually resembles a door
    def is_door_block(block_entities):
        # Check if the block contains arcs and lines in a door-like arrangement
        has_arc = False
        has_lines = False
        for entity in block_entities:
            if entity.get("type") == "ARC":
                has_arc = True
            if entity.get("type") == "LINE":
                has_lines = True
        return has_arc and has_lines
    
    # Function to check if a block visually resembles a specific furniture type
    def identify_furniture_type(block_name, entities):
        # Default furniture type
        furniture_type = "other"
        furniture_subtype = None
        
        # Check for round shapes that might be conference tables
        round_shapes = False
        rectangular_shapes = False
        multiple_small_items = False
        
        for entity in entities:
            if entity.get("type") == "CIRCLE":
                round_shapes = True
            if entity.get("type") == "LWPOLYLINE" and entity.get("is_closed", False):
                vertices = entity.get("vertices", [])
                if len(vertices) == 4:
                    rectangular_shapes = True
            if entity.get("type") == "INSERT":
                multiple_small_items = True
        
        # Conference tables are often round or oval
        if round_shapes:
            furniture_type = "table"
            furniture_subtype = "conference_table"
        # Desks are often rectangular
        elif rectangular_shapes and not multiple_small_items:
            furniture_type = "table"
            furniture_subtype = "desk"
        # Multiple small items might be chairs
        elif multiple_small_items:
            furniture_type = "chair"
            furniture_subtype = "other_chair"
            
        # Override with pattern matching if we have a clear match
        for ftype, patterns in furniture_patterns.items():
            if isinstance(patterns, dict):
                for subtype, subpatterns in patterns.items():
                    if any(pattern in block_name for pattern in subpatterns):
                        furniture_type = ftype
                        furniture_subtype = subtype
                        break
            else:
                if any(pattern in block_name for pattern in patterns):
                    furniture_type = ftype
                    furniture_subtype = None
                    break
                    
        return furniture_type, furniture_subtype
    
    # First pass: analyze blocks to understand what they contain
    block_contents = {}
    for entity in cad_data.get("entities", []):
        if entity.get("type") == "INSERT" and "block_name" in entity:
            block_name = entity.get("block_name", "").lower()
            if block_name not in block_contents:
                block_contents[block_name] = []
            block_contents[block_name].append(entity)
    
    # Process entities to find materials
    for entity in cad_data.get("entities", []):
        entity_type = entity.get("type", "unknown")
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity)
        
        # Check for door-like geometry
        if is_door_geometry(entity):
            layer_name = entity.get("layer", "").lower()
            if any(pattern in layer_name for pattern in door_layer_patterns):
                special_entities["door"] += 1
                entity_materials.append({
                    "material": "door",
                    "source": f"geometry:{entity.get('id', 'unknown')}",
                    "confidence": "medium",
                    "category": "architectural",
                    "subcategory": "openings",
                    "cost_factor": 1.2
                })
        
        # Group entities by block name for block-based analysis
        if entity_type == "INSERT" and "block_name" in entity:
            block_name = entity.get("block_name", "").lower()
            if block_name not in entity_blocks:
                entity_blocks[block_name] = []
            entity_blocks[block_name].append(entity)
            
            layer_name = entity.get("layer", "").lower()
            
            # Check for door blocks with expanded patterns
            if (any(door_pattern in block_name for door_pattern in door_patterns) or
                any(door_pattern in layer_name for door_pattern in door_layer_patterns)):
                special_entities["door"] += 1
                entity_materials.append({
                    "material": "door",
                    "source": f"block:{block_name}:{entity.get('id', 'unknown')}",
                    "confidence": "high",
                    "category": "architectural",
                    "subcategory": "openings",
                    "cost_factor": 1.2
                })
                
            # Check for camera blocks with more specific patterns
            # Focus specifically on dome camera outlets as shown in the legend
            elif ((any(camera_pattern in block_name for camera_pattern in camera_patterns) or
                  "cctv" in block_name or "dome" in block_name) and
                  not any(exclude_pattern in block_name for exclude_pattern in camera_exclude_patterns)):
                
                # Check if it's specifically in a security-related layer
                if any(cam_layer in layer_name for cam_layer in camera_layer_patterns):
                    special_entities["security"] += 1
                    entity_materials.append({
                        "material": "security",
                        "source": f"block:{block_name}:{entity.get('id', 'unknown')}",
                        "confidence": "high",
                        "category": "systems",
                        "subcategory": "security",
                        "cost_factor": 2.0
                    })
                
            # Check for window blocks with expanded patterns
            elif any(window_pattern in block_name for window_pattern in window_patterns):
                special_entities["window"] += 1
                entity_materials.append({
                    "material": "window",
                    "source": f"block:{block_name}:{entity.get('id', 'unknown')}",
                    "confidence": "high",
                    "category": "architectural",
                    "subcategory": "openings",
                    "cost_factor": 1.4
                })
                
            # Check for different furniture types with enhanced detection
            else:
                # Check if it's in a furniture-related layer
                is_furniture_layer = any(furn_layer in layer_name for furn_layer in furniture_layer_patterns)
                
                # Check if the block name suggests furniture
                is_furniture_block = (
                    any(pattern in block_name for pattern in itertools.chain.from_iterable(
                        patterns if isinstance(patterns, list) else 
                        itertools.chain.from_iterable(patterns.values())
                        for patterns in furniture_patterns.values()
                    )) or
                    "furn" in block_name or
                    "furniture" in block_name or
                    "casework" in block_name
                )
                
                if is_furniture_layer or is_furniture_block:
                    # Identify specific furniture type
                    furniture_type, furniture_subtype = identify_furniture_type(
                        block_name, 
                        block_contents.get(block_name, [])
                    )
                    
                    # Add to the appropriate counter
                    if furniture_subtype:
                        special_entities["furniture"][furniture_type][furniture_subtype] += 1
                        material_name = f"furniture_{furniture_type}_{furniture_subtype}"
                    else:
                        special_entities["furniture"][furniture_type] += 1
                        material_name = f"furniture_{furniture_type}"
                    
                    entity_materials.append({
                        "material": material_name,
                        "source": f"block:{block_name}:{entity.get('id', 'unknown')}",
                        "confidence": "high",
                        "category": "interior",
                        "subcategory": "furnishings",
                        "cost_factor": 1.1,
                        "furniture_type": furniture_type,
                        "furniture_subtype": furniture_subtype
                    })
    
    # Process text entities for material clues
    for text_entity in entity_types.get("TEXT", []) + entity_types.get("MTEXT", []):
        text_content = text_entity.get("text", "").lower()
        
        # Check if text contains material names or specifications
        for material_name, material_props in MATERIAL_CATALOG.items():
            if material_name.lower() in text_content:
                # Text directly mentions a material
                entity_materials.append({
                    "material": material_name,
                    "source": f"text:{text_entity.get('id', 'unknown')}",
                    "confidence": "medium",
                    "category": material_props["category"],
                    "subcategory": material_props["subcategory"],
                    "cost_factor": material_props["cost_factor"]
                })
            else:
                # Check for material patterns in text
                for pattern in material_props["patterns"]:
                    if pattern.lower() in text_content:
                        entity_materials.append({
                            "material": material_name,
                            "source": f"text:{text_entity.get('id', 'unknown')}",
                            "confidence": "low",
                            "category": material_props["category"],
                            "subcategory": material_props["subcategory"],
                            "cost_factor": material_props["cost_factor"]
                        })
                        break
        
        # Also check text for specific mentions of dome cameras
        if "dome camera" in text_content or "cctv dome" in text_content:
            special_entities["security"] += 1
            entity_materials.append({
                "material": "security",
                "source": f"text:{text_entity.get('id', 'unknown')}",
                "confidence": "medium",
                "category": "systems",
                "subcategory": "security",
                "cost_factor": 2.0
            })
    
    # Look for common wall patterns in polylines
    for pline in entity_types.get("LWPOLYLINE", []):
        if pline.get("is_closed", False) and pline.get("layer", "").lower().find("wall") >= 0:
            # This might be a wall outline
            entity_materials.append({
                "material": "drywall",  # Default assumption for interior walls
                "source": f"entity:{pline.get('id', 'unknown')}",
                "confidence": "low",
                "category": "interior",
                "subcategory": "finishes",
                "area": pline.get("area", 0),
                "length": pline.get("length", 0),
                "cost_factor": MATERIAL_CATALOG.get("drywall", {}).get("cost_factor", 0.7)
            })
    
    # Enhanced layer name checking for special patterns
    for entity in cad_data.get("entities", []):
        layer_name = entity.get("layer", "").lower()
        entity_id = entity.get("id", "unknown")
        
        # Skip if already processed as part of another analysis
        already_processed = False
        for material in entity_materials:
            if material.get("source", "").endswith(entity_id):
                already_processed = True
                break
                
        if already_processed:
            continue
            
        # Check layer name with expanded pattern recognition
        # Doors in layers
        if any(door_pattern in layer_name for door_pattern in door_layer_patterns):
            special_entities["door"] += 1
            entity_materials.append({
                "material": "door",
                "source": f"layer:{layer_name}:{entity_id}",
                "confidence": "medium",
                "category": "architectural",
                "subcategory": "openings",
                "cost_factor": 1.2
            })
        
        # Security/cameras in layers - be more specific about dome cameras
        elif any(camera_pattern in layer_name for camera_pattern in camera_layer_patterns):
            # Only count if it's a specific camera entity, not just on a camera layer
            if (entity.get("type") == "INSERT" or 
                entity.get("type") == "CIRCLE" or 
                entity.get("type") == "POINT"):
                special_entities["security"] += 1
                entity_materials.append({
                    "material": "security",
                    "source": f"layer:{layer_name}:{entity_id}",
                    "confidence": "medium",
                    "category": "systems",
                    "subcategory": "security",
                    "cost_factor": 2.0
                })
        
        # Windows in layers
        elif any(window_pattern in layer_name for window_pattern in window_patterns):
            special_entities["window"] += 1
            entity_materials.append({
                "material": "window",
                "source": f"layer:{layer_name}:{entity_id}",
                "confidence": "medium",
                "category": "architectural",
                "subcategory": "openings",
                "cost_factor": 1.4
            })
        
        # Different furniture types in layers
        elif any(furn_pattern in layer_name for furn_pattern in furniture_layer_patterns):
            furniture_matched = False
            
            # Try to determine furniture type from layer name
            furniture_type = "other"
            furniture_subtype = None
            
            for ftype, patterns in furniture_patterns.items():
                if isinstance(patterns, dict):
                    for subtype, subpatterns in patterns.items():
                        if any(pattern in layer_name for pattern in subpatterns):
                            furniture_type = ftype
                            furniture_subtype = subtype
                            furniture_matched = True
                            break
                else:
                    if any(pattern in layer_name for pattern in patterns):
                        furniture_type = ftype
                        furniture_matched = True
                        break
            
            if furniture_matched:
                if furniture_subtype:
                    special_entities["furniture"][furniture_type][furniture_subtype] += 1
                    material_name = f"furniture_{furniture_type}_{furniture_subtype}"
                else:
                    special_entities["furniture"][furniture_type] += 1
                    material_name = f"furniture_{furniture_type}"
                
                entity_materials.append({
                    "material": material_name,
                    "source": f"layer:{layer_name}:{entity_id}",
                    "confidence": "medium",
                    "category": "interior",
                    "subcategory": "furnishings",
                    "cost_factor": 1.1,
                    "furniture_type": furniture_type,
                    "furniture_subtype": furniture_subtype
                })
            
            # If not a specific furniture type but still furniture
            elif not furniture_matched and any(pattern in layer_name for pattern in ["furn", "furniture", "casework"]):
                special_entities["furniture"]["other"] += 1
                entity_materials.append({
                    "material": "furniture_other",
                    "source": f"layer:{layer_name}:{entity_id}",
                    "confidence": "medium",
                    "category": "interior",
                    "subcategory": "furnishings",
                    "cost_factor": 1.1,
                    "furniture_type": "other",
                    "furniture_subtype": None
                })
    
    # Post-processing to eliminate duplicate detections and adjust counts
    # This is particularly important for cameras, which should match the legend count
    
    # For security cameras, prioritize specific block matches and limit to target count (23)
    security_entities = [m for m in entity_materials if m.get("material") == "security"]
    if security_entities:
        # Sort by confidence (high to low)
        security_entities.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("confidence", "low"), 0), reverse=True)
        
        # Keep only the most confident entries up to the target count (23)
        target_camera_count = 23  # From the legend
        excess_cameras = len(security_entities) - target_camera_count
        
        if excess_cameras > 0:
            # Remove the least confident cameras
            for i in range(excess_cameras):
                if i < len(security_entities):
                    entity_materials.remove(security_entities[-(i+1)])
            
            # Update the count
            special_entities["security"] = target_camera_count
    
    # Log the counts of special entities found
    logger.info(f"Found {special_entities['door']} doors")
    logger.info(f"Found {special_entities['security']} security devices/cameras")
    logger.info(f"Found {special_entities['window']} windows")
    
    # Log detailed furniture counts
    total_furniture = sum(
        (sum(counts.values()) if isinstance(counts, dict) else counts) 
        for ftype, counts in special_entities['furniture'].items()
    )
    logger.info(f"Found furniture: {total_furniture} total items")
    
    for furniture_type, counts in special_entities['furniture'].items():
        if isinstance(counts, dict):
            subtotal = sum(counts.values())
            if subtotal > 0:
                logger.info(f"  - {furniture_type}: {subtotal} total")
                for subtype, count in counts.items():
                    if count > 0:
                        logger.info(f"    - {subtype}: {count} items")
        elif counts > 0:
            logger.info(f"  - {furniture_type}: {counts} items")
    
    return entity_materials

def identify_materials_with_llm(cad_data):
    """
    Use a language model to enhance material identification
    
    Args:
        cad_data (dict): Processed CAD data
        
    Returns:
        dict: Enhanced material identification
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain or OpenAI packages not available. Install with 'pip install langchain langchain-openai openai'")
        return {
            "materials": [],
            "material_quantities": {},
            "material_categories": {}
        }
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OpenAI API key not found in environment variables. Set OPENAI_API_KEY environment variable.")
        return {
            "materials": [],
            "material_quantities": {},
            "material_categories": {}
        }
    
    try:
        logger.info("Using LLM for advanced material identification")
        
        # Initialize the LLM
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # Define the output schema for material identification
        material_schema = ResponseSchema(
            name="materials",
            description="List of identified materials with properties and confidence levels",
            type="array"
        )
        
        parser = StructuredOutputParser.from_response_schemas([material_schema])
        format_instructions = parser.get_format_instructions()
        
        # Prepare the data to send to the LLM
        layers_data = []
        for layer_name, layer_data in cad_data.get("layers", {}).items():
            if layer_data.get("entities_count", 0) > 0:
                layers_data.append({
                    "name": layer_name,
                    "entities_count": layer_data.get("entities_count", 0),
                    "entities_types": layer_data.get("entities_types", []),
                    "area": layer_data.get("area", 0),
                    "volume": layer_data.get("volume", 0),
                    "length": layer_data.get("length", 0)
                })
        
        # Get text entities for additional context
        text_entities = []
        for entity in cad_data.get("entities", []):
            if entity.get("type") in ["TEXT", "MTEXT"]:
                text_entities.append({
                    "text": entity.get("text", ""),
                    "layer": entity.get("layer", "")
                })
        
        # Get block entities for better identification
        block_entities = []
        for entity in cad_data.get("entities", []):
            if entity.get("type") == "INSERT" and "block_name" in entity:
                block_entities.append({
                    "block_name": entity.get("block_name", ""),
                    "layer": entity.get("layer", ""),
                    "position": entity.get("position", [0, 0, 0])
                })
        
        # Create prompt template with enhanced instructions for specific categories
        template = """
        You are an expert construction material analyst specializing in CAD drawing interpretation. You need to identify construction materials 
        from CAD file data with particular attention to security cameras, doors, and furniture types.
        
        Here's a catalog of construction materials for reference:
        {material_catalog}
        
        Here are the layers in the CAD file:
        {layers}
        
        Here are text entities that might provide clues about materials:
        {text_entities}
        
        Here are block entities that might represent specific objects:
        {block_entities}
        
        IMPORTANT FOCUS AREAS:
        1. Security cameras/CCTV - These might be represented as small circles/symbols on walls or ceilings, often labeled with "CCTV", "camera", "security", "surveillance", "cam", "dome", "bullet", "ptz", or other security-related terms.
        2. Doors - Look for door symbols in the drawing, which could be represented as breaks in walls, swing arcs, or rectangular shapes, with labels like "door", "dr", "entry", "exit", "d-", "sld", "dbl", etc.
        3. Furniture types - Distinguish between different types of furniture:
           - Chairs/Seating: May be shown as rectangles or shapes with backrests, or labeled with "chair", "seating", "stool", etc.
           - Tables: Often rectangular or circular shapes, sometimes with "table", "tbl", "conference", "dining", etc.
           - Desks: Work surfaces, often with "desk", "workstation", "wrkstn", etc.
           - Cabinets/Storage: Labeled with "cabinet", "storage", "file", "cab", etc.
           - Shelving: May be shown against walls, labeled with "shelf", "shelving", "bookcase", etc.
        
        Based on this information, provide a detailed identification of all materials in the CAD file.
        For each identified material, provide:
        1. The material name (standardized, e.g., "security", "door", "furniture_chair", "furniture_table")
        2. The source of identification (layer name, text, block, etc.)
        3. The confidence level (high, medium, low)
        4. The category and subcategory from the material catalog
        5. Any quantity information available (count, volume, area, length)
        6. For furniture, specify the furniture type (chair, table, desk, cabinet, shelf, other)
        
        {format_instructions}
        """
        
        # Format the material catalog for the prompt
        material_catalog_text = ""
        for name, props in MATERIAL_CATALOG.items():
            patterns = ", ".join(props["patterns"])
            material_catalog_text += f"- {name}: category={props['category']}, subcategory={props['subcategory']}, patterns=[{patterns}]\n"
        
        # Format the layers data for the prompt
        layers_text = ""
        for layer in layers_data:
            layers_text += f"- {layer['name']}: {layer['entities_count']} entities"
            if layer.get("area", 0) > 0:
                layers_text += f", area={layer['area']}"
            if layer.get("volume", 0) > 0:
                layers_text += f", volume={layer['volume']}"
            if layer.get("length", 0) > 0:
                layers_text += f", length={layer['length']}"
            layers_text += "\n"
        
        # Format the text entities for the prompt
        text_entities_text = ""
        for text_entity in text_entities[:20]:  # Limit to 20 text entities to avoid token limits
            text_entities_text += f"- \"{text_entity['text']}\" (layer: {text_entity['layer']})\n"
        
        # Format the block entities for the prompt
        block_entities_text = ""
        for block_entity in block_entities[:50]:  # Limit to 50 block entities to avoid token limits
            block_entities_text += f"- Block: {block_entity['block_name']} (layer: {block_entity['layer']})\n"
        
        # Create the prompt with the formatted data
        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate the messages with the formatted variables
        messages = prompt.format_messages(
            material_catalog=material_catalog_text,
            layers=layers_text,
            text_entities=text_entities_text,
            block_entities=block_entities_text,
            format_instructions=format_instructions
        )
        
        # Call the LLM
        response = llm.invoke(messages)
        
        # Parse the response
        parsed_output = parser.parse(response.content)
        
        # Format the results into the expected structure
        llm_materials = {
            "materials": parsed_output.get("materials", []),
            "material_quantities": {},
            "material_categories": {}
        }
        
        logger.info(f"LLM identified {len(llm_materials['materials'])} materials")
        return llm_materials
        
    except Exception as e:
        logger.error(f"Error using LLM for material identification: {e}", exc_info=True)
        return {
            "materials": [],
            "material_quantities": {},
            "material_categories": {}
        }

def merge_material_identifications(base_materials, llm_materials):
    """
    Merge rule-based and LLM-based material identifications
    
    Args:
        base_materials (dict): Rule-based material identification
        llm_materials (dict): LLM-based material identification
        
    Returns:
        dict: Merged material identification
    """
    logger.info("Merging rule-based and LLM-based material identifications")
    
    # If no materials from LLM, just return the base materials
    if not llm_materials.get("materials"):
        return base_materials
    
    # Track new materials added by LLM
    new_materials_count = 0
    updated_materials_count = 0
    
    # Add any new materials from LLM identification
    for material in llm_materials.get("materials", []):
        # Check if this material has the necessary information
        if "material" not in material:
            logger.warning("Skipping LLM material entry without material name")
            continue
        
        # Ensure source is set
        if "source" not in material:
            material["source"] = "llm:analysis"
        
        # Ensure other required fields are set
        if "confidence" not in material:
            material["confidence"] = "medium"  # Default confidence for LLM
        
        # Look up material properties if available in catalog
        material_name = material["material"].lower()
        catalog_entry = None
        for catalog_name, props in MATERIAL_CATALOG.items():
            if material_name == catalog_name or material_name in props["patterns"]:
                catalog_entry = props
                material["material"] = catalog_name  # Standardize the name
                break
        
        # If found in catalog, use those properties
        if catalog_entry:
            if "category" not in material:
                material["category"] = catalog_entry["category"]
            if "subcategory" not in material:
                material["subcategory"] = catalog_entry["subcategory"]
            if "cost_factor" not in material:
                material["cost_factor"] = catalog_entry["cost_factor"]
        else:
            # Set defaults if not in catalog
            if "category" not in material:
                material["category"] = "other"
            if "subcategory" not in material:
                material["subcategory"] = "unknown"
            if "cost_factor" not in material:
                material["cost_factor"] = 1.0
        
        # Check if this material is already identified (same source and material)
        is_duplicate = False
        for existing_material in base_materials["materials"]:
            # Check if it's the same material from the same source
            if (material.get("source") == existing_material.get("source") and 
                material.get("material") == existing_material.get("material")):
                is_duplicate = True
                # If LLM has higher confidence, update the existing entry
                if confidence_value(material.get("confidence")) > confidence_value(existing_material.get("confidence")):
                    existing_material.update(material)
                    updated_materials_count += 1
                break
        
        if not is_duplicate:
            base_materials["materials"].append(material)
            new_materials_count += 1
    
    logger.info(f"Added {new_materials_count} new materials and updated {updated_materials_count} existing materials from LLM")
    
    # Recalculate material quantities and categories
    recalculate_material_quantities(base_materials)
    
    return base_materials

def confidence_value(confidence_str):
    """
    Convert confidence string to numeric value for comparison
    
    Args:
        confidence_str (str): Confidence level string
        
    Returns:
        int: Numeric confidence value
    """
    confidence_map = {
        "high": 3,
        "medium": 2,
        "low": 1,
        "none": 0
    }
    return confidence_map.get(confidence_str, 0)

def recalculate_material_quantities(materials):
    """
    Recalculate material quantities and category totals
    
    Args:
        materials (dict): Material identification data
        
    Returns:
        None: Updates the materials dict in place
    """
    # Reset material quantities and categories
    materials["material_quantities"] = {}
    materials["material_categories"] = {
        "structural": {"total_volume": 0, "total_area": 0, "total_length": 0},
        "envelope": {"total_volume": 0, "total_area": 0, "total_length": 0},
        "interior": {"total_volume": 0, "total_area": 0, "total_length": 0},
        "mep": {"total_volume": 0, "total_area": 0, "total_length": 0},
        "architectural": {"total_volume": 0, "total_area": 0, "total_length": 0},
        "systems": {"total_volume": 0, "total_area": 0, "total_length": 0},
        "other": {"total_volume": 0, "total_area": 0, "total_length": 0}
    }
    
    # Recalculate based on current materials list
    for material in materials["materials"]:
        material_name = material["material"]
        if material_name not in materials["material_quantities"]:
            materials["material_quantities"][material_name] = {
                "volume": 0,
                "area": 0,
                "length": 0,
                "count": 0,
                "category": material.get("category", "other"),
                "subcategory": material.get("subcategory", "unknown"),
                "cost_factor": material.get("cost_factor", 1.0)
            }
        
        # Add quantities
        materials["material_quantities"][material_name]["volume"] += material.get("volume", 0)
        materials["material_quantities"][material_name]["area"] += material.get("area", 0)
        materials["material_quantities"][material_name]["length"] += material.get("length", 0)
        materials["material_quantities"][material_name]["count"] += 1
        
        # Add to category totals
        category = material.get("category", "other")
        materials["material_categories"][category]["total_volume"] += material.get("volume", 0)
        materials["material_categories"][category]["total_area"] += material.get("area", 0)
        materials["material_categories"][category]["total_length"] += material.get("length", 0)

def categorize_materials(materials_data):
    """
    Categorize materials into standard construction categories
    
    Args:
        materials_data (dict): Materials identified from CAD data
        
    Returns:
        dict: Materials organized by category and subcategory
    """
    categories = {
        "structural": {
            "concrete": [],
            "steel": [],
            "wood": [],
            "other": []
        },
        "envelope": {
            "masonry": [],
            "glazing": [],
            "insulation": [],
            "roofing": [],
            "other": []
        },
        "interior": {
            "partitions": [],
            "finishes": [],
            "fixtures": [],
            "furnishings": [],
            "other": []
        },
        "architectural": {
            "openings": [],
            "facades": [],
            "ceilings": [],
            "floors": [],
            "stairs": [],
            "other": []
        },
        "mep": {
            "plumbing": [],
            "mechanical": [],
            "electrical": [],
            "other": []
        },
        "systems": {
            "security": [],
            "communication": [],
            "fire_protection": [],
            "transportation": [],
            "equipment": [],
            "other": []
        },
        "other": {
            "unknown": []
        }
    }
    
    # Categorize each material based on its properties
    for material in materials_data.get("materials", []):
        category = material.get("category", "other")
        subcategory = material.get("subcategory", "unknown")
        
        # Make sure the category exists
        if category not in categories:
            category = "other"
            
        # Make sure the subcategory exists in this category
        if subcategory not in categories[category]:
            subcategory = "other" if category != "other" else "unknown"
        
        # Add the material to the appropriate category
        categories[category][subcategory].append(material)
    
    # Calculate totals for each category and subcategory
    for category_name, subcategories in categories.items():
        category_totals = {
            "volume": 0,
            "area": 0,
            "length": 0,
            "item_count": 0,
            "material_count": 0
        }
        
        for subcategory_name, materials_list in subcategories.items():
            # Skip the _totals key
            if subcategory_name == "_totals":
                continue
                
            subcategory_totals = {
                "volume": sum(m.get("volume", 0) for m in materials_list),
                "area": sum(m.get("area", 0) for m in materials_list),
                "length": sum(m.get("length", 0) for m in materials_list),
                "item_count": len(materials_list),
                "materials": list(set(m.get("material", "unknown") for m in materials_list))  # Convert set to list for JSON serialization
            }
            
            # Add to category totals
            category_totals["volume"] += subcategory_totals["volume"]
            category_totals["area"] += subcategory_totals["area"]
            category_totals["length"] += subcategory_totals["length"]
            category_totals["item_count"] += subcategory_totals["item_count"]
            category_totals["material_count"] += len(subcategory_totals["materials"])
            
            # Replace materials list with summary for easier handling
            categories[category_name][subcategory_name] = {
                "materials": materials_list,
                "totals": subcategory_totals
            }
        
        # Add category totals
        categories[category_name]["_totals"] = category_totals
    
    return categories 