#!/usr/bin/env python
"""
DXF Block Counter

This script counts all blocks in a DXF file and outputs the results in JSON format.
Includes all block definitions, references, and nested block hierarchies.
Excludes blocks with zero count from the main output file.
"""
import os
import json
import logging
import sys
import math
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
import ezdxf
from ezdxf.entities.dxfentity import DXFEntity
from ezdxf.math import Vec2, Vec3, BoundingBox2d

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_angle_between(angle, start_angle, end_angle):
    """
    Check if an angle is between the start and end angles.
    
    Args:
        angle (float): The angle to check, in radians
        start_angle (float): The start angle, in radians
        end_angle (float): The end angle, in radians
        
    Returns:
        bool: True if the angle is between start_angle and end_angle, False otherwise
    """
    # Normalize angles to 0-2π range
    angle = angle % (2 * math.pi)
    start_angle = start_angle % (2 * math.pi)
    end_angle = end_angle % (2 * math.pi)
    
    # Handle the case where end_angle is less than start_angle (arc crosses 0°)
    if end_angle < start_angle:
        return angle >= start_angle or angle <= end_angle
    else:
        return angle >= start_angle and angle <= end_angle

def analyze_blocks_in_dxf(dxf_path):
    """
    Analyze all blocks in a DXF file, including definitions, references, and nested blocks.
    
    Args:
        dxf_path (str): Path to the DXF file
        
    Returns:
        dict: Dictionary with comprehensive block information
    """
    try:
        logger.info(f"Processing DXF file: {dxf_path}")
        
        # Open the DXF file
        doc = ezdxf.readfile(dxf_path)
        
        # Dictionary to store results
        result = {
            "block_definitions": {},
            "block_references": defaultdict(lambda: {"total": 0, "modelspace": 0, "paperspace": 0}),
            "nested_blocks": defaultdict(list),
            "block_hierarchy": {},
            "nested_reference_counts": defaultdict(int),
            "summary": {
                "total_definitions": 0,
                "total_references": 0,
                "total_nested_references": 0,
                "blocks_with_references": 0,
                "blocks_without_references": 0,
                "blocks_with_nested_blocks": 0
            }
        }
        
        # First pass: collect all block definitions
        process_block_definitions(doc, result)
        
        # Second pass: find all block references in model space
        msp = doc.modelspace()
        process_block_references(msp, result["block_references"], "modelspace")
        
        # Try to process paper space layouts if available
        try:
            for layout in doc.layouts:
                if layout.name.upper() == 'MODEL':
                    continue  # Skip model space, already processed
                
                try:
                    # Different methods to access layout entities based on ezdxf version
                    if hasattr(layout, 'layout_space'):
                        layout_space = layout.layout_space()
                    elif hasattr(layout, 'block'):
                        layout_space = layout.block
                    elif hasattr(doc, 'blocks') and layout.dxf.block_record in doc.blocks:
                        layout_space = doc.blocks[layout.dxf.block_record]
                    else:
                        # As a last resort, try to get the block from the blocks collection
                        block_name = f"*Paper_Space{layout.dxf.layout_index if hasattr(layout.dxf, 'layout_index') else ''}"
                        if block_name in doc.blocks:
                            layout_space = doc.blocks[block_name]
                        else:
                            logger.warning(f"Could not access entities for layout {layout.name}")
                            continue
                        
                    process_block_references(layout_space, result["block_references"], "paperspace")
                except Exception as e:
                    logger.warning(f"Error processing layout {layout.name}: {e}")
        except Exception as e:
            logger.warning(f"Error processing paper space layouts: {e}")
            logger.warning("Continuing with model space blocks only.")
        
        # Find nested blocks (blocks referenced by other blocks)
        detect_nested_blocks(doc, result)
        
        # Build complete block hierarchy (including all nesting levels)
        build_block_hierarchy(doc, result)
        
        # Calculate totals and mark which blocks have references
        finalize_block_counts(result)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        result["block_references"] = dict(result["block_references"])
        result["nested_blocks"] = dict(result["nested_blocks"])
        result["nested_reference_counts"] = dict(result["nested_reference_counts"])
        
        logger.info(f"Found {result['summary']['total_definitions']} block definitions")
        logger.info(f"Found {result['summary']['total_references']} direct block references")
        logger.info(f"Found {result['summary']['total_nested_references']} nested block references")
        logger.info(f"Found {result['summary']['blocks_with_references']} blocks with direct references")
        logger.info(f"Found {result['summary']['blocks_with_nested_blocks']} blocks with nested blocks")
        logger.info(f"Found {result['summary']['blocks_without_references']} blocks without any references")
        
        return result
    
    except ezdxf.DXFError as e:
        logger.error(f"Error reading DXF file: {e}")
        raise ValueError(f"Invalid DXF file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def process_block_definitions(doc, result):
    """
    Process all block definitions in the document.
    
    Args:
        doc: ezdxf document
        result: Result dictionary to update
    """
    for block in doc.blocks:
        block_name = block.name
        
        # Skip temporary blocks used for dimension entities
        if block_name.startswith('*D'):
            continue
            
        # Count entities in block definition
        entity_count = sum(1 for _ in block)
        entity_types = {}
        
        # Analyze entities within the block
        for entity in block:
            entity_type = entity.dxftype()
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Analyze block geometry and generate SVG
        geometry = analyze_block_geometry(block)
        svg = generate_block_svg(block, geometry)
        
        # Add to results
        result["block_definitions"][block_name] = {
            "entity_count": entity_count,
            "entity_types": entity_types,
            "is_layout": block_name in ('*MODEL_SPACE', '*PAPER_SPACE', '*PAPER_SPACE0', '*PAPER_SPACE1', '*PAPER_SPACE2'),
            "is_xref": block_name.startswith('*') and not block_name.startswith('*U'),
            "is_anonymous": block_name.startswith('*U'),
            "referenced_count": 0,  # Will be updated later
            "has_nested_blocks": False,  # Will be updated later
            "shape": geometry,
            "svg": svg
        }
    
    result["summary"]["total_definitions"] = len(result["block_definitions"])

def process_block_references(block_container, reference_counts, space_type):
    """
    Process block references in a block container (model space or paper space layout).
    
    Args:
        block_container: ezdxf block container (modelspace or paperspace)
        reference_counts: Dictionary to store reference counts
        space_type: String identifying the space type ('modelspace' or 'paperspace')
    """
    for entity in block_container:
        if entity.dxftype() == "INSERT":
            block_name = entity.dxf.name
            reference_counts[block_name][space_type] += 1

def detect_nested_blocks(doc, result):
    """
    Detect blocks that contain references to other blocks.
    
    Args:
        doc: ezdxf document
        result: Result dictionary to update
    """
    for block_name, block_def in result["block_definitions"].items():
        if block_def["is_layout"]:
            continue
            
        # Get the block
        block = doc.blocks.get(block_name)
        if not block:
            continue
            
        # Look for INSERT entities within the block
        for entity in block:
            if entity.dxftype() == "INSERT":
                nested_block_name = entity.dxf.name
                
                # Record the nested relationship
                if nested_block_name != block_name:  # Avoid self-reference
                    result["nested_blocks"][block_name].append(nested_block_name)
                    result["block_definitions"][block_name]["has_nested_blocks"] = True
                    
                    # Count this as a nested reference
                    result["nested_reference_counts"][nested_block_name] += 1

def build_block_hierarchy(doc, result):
    """
    Build a complete hierarchy of block nesting, showing all levels.
    
    Args:
        doc: ezdxf document
        result: Result dictionary to update
    """
    # Initialize hierarchy with empty children lists for all blocks
    hierarchy = {}
    for block_name in result["block_definitions"]:
        hierarchy[block_name] = {
            "children": [],
            "parents": [],
            "level": 0,
            "path": [block_name],
            "nested_count": 0,
            "instances_in_model": result["block_references"].get(block_name, {"total": 0})["total"]
        }
    
    # Fill in direct children from nested_blocks
    for parent, children in result["nested_blocks"].items():
        hierarchy[parent]["children"] = children
        for child in children:
            if child in hierarchy:
                hierarchy[child]["parents"].append(parent)
    
    # Calculate nesting levels (how deep in the hierarchy)
    blocks_without_parents = [name for name, data in hierarchy.items() if not data["parents"]]
    
    # Start with blocks that have no parents (level 0)
    visited = set()
    to_process = [(name, 0, [name]) for name in blocks_without_parents]
    
    while to_process:
        block_name, level, path = to_process.pop(0)
        
        # Skip if we've already processed this block with a shorter path
        if block_name in visited and hierarchy[block_name]["level"] <= level:
            continue
            
        visited.add(block_name)
        hierarchy[block_name]["level"] = level
        hierarchy[block_name]["path"] = path
        
        # Process children
        for child in hierarchy[block_name]["children"]:
            if child in hierarchy:  # Safety check
                # Add to processing queue with incremented level and extended path
                new_path = path + [child]
                to_process.append((child, level + 1, new_path))
    
    # Calculate total nested instances
    for block_name, data in hierarchy.items():
        # If this block appears in the model directly
        direct_instances = data["instances_in_model"]
        
        # Start with direct instances from the model
        total_instances = direct_instances
        
        # For each parent, add instances based on how many times the parent appears
        for parent in data["parents"]:
            parent_instances = hierarchy[parent]["instances_in_model"]
            if parent_instances > 0:
                # This block appears nested inside its parent, and the parent appears in the model
                total_instances += parent_instances
        
        hierarchy[block_name]["nested_count"] = total_instances
    
    # Store the hierarchy
    result["block_hierarchy"] = hierarchy
    
    # Count blocks with nested blocks
    blocks_with_nested = sum(1 for data in hierarchy.values() if data["children"])
    result["summary"]["blocks_with_nested_blocks"] = blocks_with_nested

def finalize_block_counts(result):
    """
    Calculate final counts and statistics.
    
    Args:
        result: Result dictionary to update
    """
    total_references = 0
    total_nested_references = 0
    blocks_with_references = 0
    blocks_without_references = 0
    
    # Update reference counts for each block
    for block_name, counts in result["block_references"].items():
        counts["total"] = counts["modelspace"] + counts["paperspace"]
        total_references += counts["total"]
        
        # Update the block definition if it exists
        if block_name in result["block_definitions"]:
            result["block_definitions"][block_name]["referenced_count"] = counts["total"]
    
    # Count nested references
    for block_name, count in result["nested_reference_counts"].items():
        total_nested_references += count
    
    # Count blocks with and without references
    for block_name, block_def in result["block_definitions"].items():
        referenced_count = block_def["referenced_count"]
        nested_count = result["nested_reference_counts"].get(block_name, 0)
        
        if referenced_count > 0 or nested_count > 0:
            blocks_with_references += 1
        else:
            blocks_without_references += 1
    
    # Update summary
    result["summary"]["total_references"] = total_references
    result["summary"]["total_nested_references"] = total_nested_references
    result["summary"]["blocks_with_references"] = blocks_with_references
    result["summary"]["blocks_without_references"] = blocks_without_references

def create_simplified_count(block_data):
    """
    Create a simplified block count from detailed block data.
    Only include blocks with non-zero counts.
    
    Args:
        block_data (dict): Detailed block information
        
    Returns:
        dict: Dictionary with block names as keys and counts as values
    """
    simplified = {}
    
    # Add block definitions with non-zero reference counts
    for block_name, block_def in block_data["block_definitions"].items():
        # Skip temporary dimension blocks and layout blocks
        if block_name.startswith('*D') or block_def["is_layout"]:
            continue
            
        # Only include blocks with a count > 0
        if block_def["referenced_count"] > 0:
            simplified[block_name] = block_def["referenced_count"]
    
    return simplified

def create_referenced_blocks_only(block_data):
    """
    Create a dictionary with only blocks that have references.
    
    Args:
        block_data (dict): Detailed block information
        
    Returns:
        dict: Dictionary with only blocks that have references
    """
    referenced = {}
    
    # Add only blocks with references
    for block_name, counts in block_data["block_references"].items():
        if counts["total"] > 0:
            referenced[block_name] = counts["total"]
    
    return referenced

def create_total_reference_count(block_data):
    """
    Create a dictionary with blocks and their total reference count (direct + nested).
    Only includes blocks with non-zero total counts.
    
    Args:
        block_data (dict): Detailed block information
        
    Returns:
        dict: Dictionary with block names and their total reference counts
    """
    total_counts = {}
    
    # Process all blocks
    for block_name in block_data["block_definitions"]:
        # Get direct references (from model/paper space)
        direct_count = block_data["block_references"].get(block_name, {"total": 0})["total"]
        
        # Get nested references (from within other blocks)
        nested_count = block_data["nested_reference_counts"].get(block_name, 0)
        
        # Calculate total and include only if greater than zero
        total_count = direct_count + nested_count
        if total_count > 0:
            total_counts[block_name] = {
                "direct_references": direct_count,
                "nested_references": nested_count,
                "total_references": total_count
            }
    
    return total_counts

def create_nested_hierarchy_output(block_data):
    """
    Create a structured output that shows the nested block hierarchy.
    
    Args:
        block_data (dict): Detailed block information
        
    Returns:
        dict: Structured hierarchy data for output
    """
    hierarchy_output = {
        "blocks_with_nested_content": {},
        "all_blocks_with_hierarchy": {},
        "summary": {
            "total_blocks_with_nested_content": 0,
            "total_nested_levels": 0,
            "deepest_nesting": 0
        }
    }
    
    # Extract hierarchy info from block_hierarchy
    max_level = 0
    blocks_with_children = 0
    
    for block_name, data in block_data["block_hierarchy"].items():
        # Skip blocks without children or those that never appear in the model
        instances_in_model = data["instances_in_model"]
        has_children = len(data["children"]) > 0
        
        if has_children:
            blocks_with_children += 1
        
        # Track max nesting level
        level = data["level"]
        if level > max_level:
            max_level = level
        
        # Create a structured record for this block
        hierarchy_record = {
            "level": level,
            "direct_references": instances_in_model,
            "nested_references": data["nested_count"] - instances_in_model if data["nested_count"] > instances_in_model else 0,
            "total_references": data["nested_count"],
            "children": data["children"],
            "parents": data["parents"],
            "path": data["path"]
        }
        
        # Add to full hierarchy output
        hierarchy_output["all_blocks_with_hierarchy"][block_name] = hierarchy_record
        
        # If this block has children, add it to the blocks_with_nested_content section
        if has_children:
            hierarchy_output["blocks_with_nested_content"][block_name] = hierarchy_record
    
    # Update summary
    hierarchy_output["summary"]["total_blocks_with_nested_content"] = blocks_with_children
    hierarchy_output["summary"]["total_nested_levels"] = max_level + 1  # Levels are 0-indexed
    hierarchy_output["summary"]["deepest_nesting"] = max_level
    
    return hierarchy_output

def create_hierarchical_tree(block_data):
    """
    Create a simplified hierarchical tree of blocks with counts > 0,
    showing each block's total count and its children (also with counts > 0).
    
    The output is focused on blocks that:
    1. Have a count > 0 (direct or nested references)
    2. Have children blocks (are parent blocks)
    3. The blocks are sorted by total count in descending order
    
    Each block entry includes:
    - count: Total number of references (direct + nested)
    - direct: Number of direct references in model/paper space
    - nested: Number of nested references (inside other blocks)
    - children: Dictionary of child blocks and their counts
    - shape: Geometric information about the block
    
    Args:
        block_data (dict): Detailed block information
        
    Returns:
        dict: Tree structure with block names, counts, and children
    """
    # Start with all blocks and their total references
    all_blocks = {}
    
    # Get all blocks with count > 0
    for block_name in block_data["block_definitions"]:
        # Get direct references (from model/paper space)
        direct_count = block_data["block_references"].get(block_name, {"total": 0})["total"]
        
        # Get nested references (from within other blocks)
        nested_count = block_data["nested_reference_counts"].get(block_name, 0)
        
        # Calculate total count
        total_count = direct_count + nested_count
        
        if total_count > 0:
            block_def = block_data["block_definitions"][block_name]
            
            all_blocks[block_name] = {
                "count": total_count,
                "direct_count": direct_count,
                "nested_count": nested_count,
                "children": {},
                # Include shape information only for blocks with count > 0
                "shape": block_def.get("shape", {}),
                "svg": block_def.get("svg", "")
            }
    
    # Now build the parent-child relationships
    for parent_name, children in block_data["nested_blocks"].items():
        # Skip parents that have no references
        if parent_name not in all_blocks:
            continue
            
        # Add children with count > 0
        for child_name in children:
            if child_name in all_blocks:
                # Store child reference with its count
                all_blocks[parent_name]["children"][child_name] = all_blocks[child_name]["count"]
    
    # Filter blocks: 
    # 1. Must have children
    # 2. Prioritize blocks with direct references when displaying top level
    blocks_with_children = {}
    for block_name, data in all_blocks.items():
        if data["children"]:
            # Create a simplified version for output
            blocks_with_children[block_name] = {
                "count": data["count"],
                "direct": data["direct_count"],
                "nested": data["nested_count"],
                "children": data["children"],
                "shape": data["shape"],
                "svg": data["svg"]
            }
    
    # Sort blocks by direct references first, then by total count (both descending)
    # This prioritizes blocks that appear directly in the model space
    sorted_blocks = {}
    for block_name, data in sorted(blocks_with_children.items(), 
                                  key=lambda x: (x[1]["direct"], x[1]["count"]), 
                                  reverse=True):
        sorted_blocks[block_name] = data
    
    # Create result object with summary and user-friendly description
    result = {
        "description": "Hierarchical tree of blocks with references > 0. Blocks are sorted by direct references first, then by total count.",
        "blocks": sorted_blocks,
        "summary": {
            "total_blocks_with_references": len(all_blocks),
            "blocks_with_children": len(blocks_with_children),
            "direct_references_only": sum(1 for data in all_blocks.values() if data["direct_count"] > 0 and data["nested_count"] == 0),
            "nested_references_only": sum(1 for data in all_blocks.values() if data["direct_count"] == 0 and data["nested_count"] > 0),
            "both_direct_and_nested": sum(1 for data in all_blocks.values() if data["direct_count"] > 0 and data["nested_count"] > 0)
        }
    }
    
    return result

def analyze_block_geometry(block):
    """
    Analyze the geometry of a block to determine its extents, complexity, and other characteristics.
    
    Args:
        block: ezdxf block object
        
    Returns:
        dict: Geometry information including bounding box, complexity metrics, etc.
    """
    # Initialize geometry data
    bounding_box = {
        "min_x": float('inf'),
        "min_y": float('inf'),
        "max_x": float('-inf'),
        "max_y": float('-inf')
    }
    
    entity_counts = defaultdict(int)
    total_points = 0
    total_length = 0
    
    # Process all entities in the block
    for entity in block:
        entity_type = entity.dxftype()
        entity_counts[entity_type] += 1
        
        try:
            # Extract points based on entity type to calculate bounding box
            points = []
            
            if entity_type == "LINE":
                points = [entity.dxf.start, entity.dxf.end]
            
            elif entity_type == "CIRCLE":
                center = entity.dxf.center
                radius = entity.dxf.radius
                points = [
                    Vec2(center.x - radius, center.y),
                    Vec2(center.x + radius, center.y),
                    Vec2(center.x, center.y - radius),
                    Vec2(center.x, center.y + radius)
                ]
            
            elif entity_type == "ARC":
                center = entity.dxf.center
                radius = entity.dxf.radius
                
                # Convert angles to radians
                start_angle_rad = math.radians(entity.dxf.start_angle)
                end_angle_rad = math.radians(entity.dxf.end_angle)
                
                # Add the arc's endpoints
                start_point = Vec2(
                    center.x + radius * math.cos(start_angle_rad),
                    center.y + radius * math.sin(start_angle_rad)
                )
                end_point = Vec2(
                    center.x + radius * math.cos(end_angle_rad),
                    center.y + radius * math.sin(end_angle_rad)
                )
                points = [start_point, end_point, center]
                
                # Check if the arc crosses any of the cardinal points (0°, 90°, 180°, 270°)
                # and add those points if they're included in the arc
                for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
                    if is_angle_between(angle, start_angle_rad, end_angle_rad):
                        points.append(Vec2(
                            center.x + radius * math.cos(angle),
                            center.y + radius * math.sin(angle)
                        ))
            
            elif entity_type == "TEXT" or entity_type == "MTEXT":
                # For text entities, just use the insertion point
                points = [entity.dxf.insert]
            
            elif entity_type == "POLYLINE" or entity_type == "LWPOLYLINE":
                # Get the points from the polyline
                try:
                    # First try get_points() method (newer ezdxf versions)
                    if hasattr(entity, 'get_points'):
                        vertices = entity.get_points()
                        for vertex in vertices:
                            # Handle both tuple format and Vec2/Vec3 objects
                            if isinstance(vertex, tuple) and len(vertex) >= 2:
                                points.append(Vec2(vertex[0], vertex[1]))
                            elif hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                                points.append(Vec2(vertex.x, vertex.y))
                    # Older versions or different representation
                    elif hasattr(entity, 'vertices'):
                        # Check if vertices is callable (as in older versions)
                        if callable(entity.vertices):
                            for vertex in entity.vertices():
                                if hasattr(vertex, 'dxf'):
                                    points.append(Vec2(vertex.dxf.location.x, vertex.dxf.location.y))
                                elif hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                                    points.append(Vec2(vertex.x, vertex.y))
                        # Or if it's a list (as in some implementations)
                        elif isinstance(entity.vertices, list):
                            for vertex in entity.vertices:
                                if hasattr(vertex, 'dxf'):
                                    points.append(Vec2(vertex.dxf.location.x, vertex.dxf.location.y))
                                elif hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                                    points.append(Vec2(vertex.x, vertex.y))
                except Exception as e:
                    logger.warning(f"Could not extract points from {entity_type}: {str(e)}")
            
            # Add more entity types as needed
            
            # Update bounding box
            for point in points:
                if hasattr(point, 'x') and hasattr(point, 'y'):
                    bounding_box["min_x"] = min(bounding_box["min_x"], point.x)
                    bounding_box["min_y"] = min(bounding_box["min_y"], point.y)
                    bounding_box["max_x"] = max(bounding_box["max_x"], point.x)
                    bounding_box["max_y"] = max(bounding_box["max_y"], point.y)
            
            # Update total points
            total_points += len(points)
            
            # Calculate length for certain entities
            if entity_type == "LINE":
                total_length += entity.dxf.start.distance(entity.dxf.end)
            elif entity_type == "POLYLINE" or entity_type == "LWPOLYLINE":
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        if hasattr(points[i], 'distance'):
                            total_length += points[i].distance(points[i+1])
        
        except Exception as e:
            # Log warning but continue processing
            logger.warning(f"Error processing {entity_type}: {str(e)}")
    
    # If no entities with geometry were found, set default bounding box
    if bounding_box["min_x"] == float('inf'):
        bounding_box = {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    
    # Calculate area
    width = bounding_box["max_x"] - bounding_box["min_x"]
    height = bounding_box["max_y"] - bounding_box["min_y"]
    area = width * height
    
    # Calculate complexity score (simplified)
    entity_count = sum(entity_counts.values())
    complexity = entity_count * 0.5 + total_points * 0.3 + (total_length / 100) * 0.2 if total_length > 0 else 0
    
    return {
        "bounding_box": bounding_box,
        "entity_counts": dict(entity_counts),
        "total_entities": entity_count,
        "total_points": total_points,
        "total_length": total_length,
        "area": area,
        "complexity": complexity
    }

def generate_block_svg(block, geometry):
    """
    Generate a simplified SVG representation of a block.
    
    Args:
        block: ezdxf block object
        geometry (dict): Geometry information from analyze_block_geometry
        
    Returns:
        str: SVG representation as a string
    """
    # Extract bounding box from geometry
    min_x = geometry["bounding_box"]["min_x"]
    min_y = geometry["bounding_box"]["min_y"]
    max_x = geometry["bounding_box"]["max_x"]
    max_y = geometry["bounding_box"]["max_y"]
    
    # Calculate dimensions
    width = max_x - min_x
    height = max_y - min_y
    
    # If the geometry is empty, return a minimal SVG
    if width == 0 or height == 0:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><text x="5" y="5" font-size="2" text-anchor="middle">Empty block</text></svg>'
    
    # Add padding for better visualization
    padding = 10
    view_min_x = min_x - padding
    view_min_y = min_y - padding
    view_width = width + 2 * padding
    view_height = height + 2 * padding
    
    # Start SVG with viewBox
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_min_x} {view_min_y} {view_width} {view_height}">\n'
    
    # Add a border for the bounding box
    svg += f'  <rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" fill="none" stroke="#999" stroke-width="0.5"/>\n'
    
    # Add entities
    for entity in block:
        entity_type = entity.dxftype()
        
        try:
            if entity_type == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                svg += f'  <line x1="{start.x}" y1="{start.y}" x2="{end.x}" y2="{end.y}" stroke="black" stroke-width="0.5"/>\n'
            
            elif entity_type == "CIRCLE":
                center = entity.dxf.center
                radius = entity.dxf.radius
                svg += f'  <circle cx="{center.x}" cy="{center.y}" r="{radius}" fill="none" stroke="black" stroke-width="0.5"/>\n'
            
            elif entity_type == "ARC":
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                
                # SVG arcs are complex, this is a simplified version
                start_rad = math.radians(start_angle)
                end_rad = math.radians(end_angle)
                
                # Calculate start and end points
                start_x = center.x + radius * math.cos(start_rad)
                start_y = center.y + radius * math.sin(start_rad)
                end_x = center.x + radius * math.cos(end_rad)
                end_y = center.y + radius * math.sin(end_rad)
                
                # Determine if the arc is larger than 180 degrees
                large_arc_flag = 0 if (end_angle - start_angle) % 360 <= 180 else 1
                
                svg += f'  <path d="M {start_x} {start_y} A {radius} {radius} 0 {large_arc_flag} 1 {end_x} {end_y}" fill="none" stroke="black" stroke-width="0.5"/>\n'
            
            elif entity_type == "TEXT" or entity_type == "MTEXT":
                pos = entity.dxf.insert
                text_content = getattr(entity, "text", "[TEXT]")
                
                # Limit text length for SVG readability
                if len(text_content) > 10:
                    text_content = text_content[:10] + "..."
                
                svg += f'  <text x="{pos.x}" y="{pos.y}" font-size="2" fill="blue">{text_content}</text>\n'
            
            elif entity_type == "POLYLINE" or entity_type == "LWPOLYLINE":
                # Get points for polyline
                points_list = []
                
                try:
                    # For LWPOLYLINE in newer ezdxf versions
                    if hasattr(entity, 'get_points'):
                        for vertex in entity.get_points():
                            # Vertex might be returned as (x, y) tuple
                            if isinstance(vertex, tuple) and len(vertex) >= 2:
                                points_list.append(f"{vertex[0]},{vertex[1]}")
                            else:
                                # Or as a Vec2 or Vec3 object
                                points_list.append(f"{vertex.x},{vertex.y}")
                    # For older ezdxf versions or different structure
                    elif hasattr(entity, 'vertices'):
                        # Check if vertices is callable or a list
                        if callable(entity.vertices):
                            for vertex in entity.vertices():
                                if hasattr(vertex, 'dxf'):
                                    points_list.append(f"{vertex.dxf.location.x},{vertex.dxf.location.y}")
                                elif hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                                    points_list.append(f"{vertex.x},{vertex.y}")
                        elif isinstance(entity.vertices, list):
                            for vertex in entity.vertices:
                                if hasattr(vertex, 'dxf'):
                                    points_list.append(f"{vertex.dxf.location.x},{vertex.dxf.location.y}")
                                elif hasattr(vertex, 'x') and hasattr(vertex, 'y'):
                                    points_list.append(f"{vertex.x},{vertex.y}")
                except Exception as e:
                    logger.warning(f"Could not create SVG for {entity_type}: {str(e)}")
                
                if points_list:
                    points_str = " ".join(points_list)
                    svg += f'  <polyline points="{points_str}" fill="none" stroke="black" stroke-width="0.5"/>\n'
            
            # Add other entity types as needed
        
        except Exception as e:
            # Skip entities that can't be processed for SVG
            logger.warning(f"Could not create SVG for {entity_type}: {str(e)}")
    
    # Close SVG
    svg += '</svg>'
    
    return svg

def main():
    """Main function to count blocks and output results as JSON."""
    try:
        # Path to the DXF file
        dxf_path = os.path.join("data", "data.dxf")
        output_path = "count-test.json"
        referenced_output_path = "count-referenced.json"
        detailed_output_path = "count-detailed.json"
        nested_output_path = "count-nested.json"
        total_count_path = "count-total.json"
        hierarchical_tree_path = "count-tree.json"
        
        # Check if the file exists
        if not os.path.exists(dxf_path):
            logger.error(f"File not found: {dxf_path}")
            return 1
        
        # Make sure ezdxf is properly installed
        if not hasattr(ezdxf, 'readfile'):
            logger.error("ezdxf library not properly installed or initialized")
            return 1
            
        # Analyze blocks
        block_data = analyze_blocks_in_dxf(dxf_path)
        
        # Create simplified count (non-zero counts only)
        simplified_count = create_simplified_count(block_data)
        
        # Create a count of only blocks with references (for easier viewing)
        referenced_count = create_referenced_blocks_only(block_data)
        
        # Create total reference count (direct + nested)
        total_count = create_total_reference_count(block_data)
        
        # Create nested hierarchy output
        nested_hierarchy = create_nested_hierarchy_output(block_data)
        
        # Create hierarchical tree output (simple to read format)
        hierarchical_tree = create_hierarchical_tree(block_data)
        
        # Output results as JSON
        logger.info(f"Writing results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(simplified_count, f, indent=2)
        
        # Write referenced blocks only
        logger.info(f"Writing referenced blocks to {referenced_output_path}")
        with open(referenced_output_path, 'w') as f:
            json.dump(referenced_count, f, indent=2)
        
        # Write total reference counts
        logger.info(f"Writing total reference counts to {total_count_path}")
        with open(total_count_path, 'w') as f:
            json.dump(total_count, f, indent=2)
        
        # Write nested hierarchy
        logger.info(f"Writing nested block hierarchy to {nested_output_path}")
        with open(nested_output_path, 'w') as f:
            json.dump(nested_hierarchy, f, indent=2)
            
        # Write hierarchical tree (simple format)
        logger.info(f"Writing hierarchical tree to {hierarchical_tree_path}")
        with open(hierarchical_tree_path, 'w') as f:
            json.dump(hierarchical_tree, f, indent=2)
        
        # For debugging - write detailed data too
        with open(detailed_output_path, 'w') as f:
            json.dump(block_data, f, indent=2)
        
        logger.info(f"Successfully wrote block counts to {output_path}")
        logger.info(f"Referenced blocks written to {referenced_output_path}")
        logger.info(f"Total reference counts written to {total_count_path}")
        logger.info(f"Nested block hierarchy written to {nested_output_path}")
        logger.info(f"Hierarchical tree written to {hierarchical_tree_path}")
        logger.info(f"Detailed block information written to {detailed_output_path}")
        print(f"Block count saved to {output_path}")
        print(f"Referenced blocks saved to {referenced_output_path}")
        print(f"Total reference counts saved to {total_count_path}")
        print(f"Nested block hierarchy saved to {nested_output_path}")
        print(f"Hierarchical tree saved to {hierarchical_tree_path}")
        print(f"Detailed block information saved to {detailed_output_path}")
        
        # Print some quick statistics
        print(f"\nSummary:")
        print(f"  Total block definitions: {block_data['summary']['total_definitions']}")
        print(f"  Total direct block references: {block_data['summary']['total_references']}")
        print(f"  Total nested block references: {block_data['summary']['total_nested_references']}")
        print(f"  Blocks with direct references: {block_data['summary']['blocks_with_references']}")
        print(f"  Blocks with nested blocks: {block_data['summary']['blocks_with_nested_blocks']}")
        print(f"  Blocks without any references: {block_data['summary']['blocks_without_references']}")
        print(f"  Deepest nesting level: {nested_hierarchy['summary']['deepest_nesting']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 