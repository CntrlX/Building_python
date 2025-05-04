"""
Count Reconciler Service

This service reconciles counts from different sources (DXF analysis and Vision API analysis)
to provide the most accurate final count.
"""

import logging
from typing import Dict, Any, List, Tuple
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CountReconciler:
    def __init__(self):
        """Initialize the count reconciler service."""
        pass
        
    def reconcile_counts(self, dxf_counts: Dict[str, Any], vision_counts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile counts from DXF analysis and Vision API analysis.
        
        Args:
            dxf_counts (dict): Counts from DXF analysis
            vision_counts (dict): Counts from Vision API analysis
            
        Returns:
            dict: Reconciled counts with confidence levels
        """
        reconciled_counts = {
            "doors": self._reconcile_category("doors", dxf_counts, vision_counts),
            "security_cameras": self._reconcile_category("security_cameras", dxf_counts, vision_counts),
            "furniture": self._reconcile_furniture(dxf_counts, vision_counts),
            "confidence_levels": {
                "doors": self._calculate_confidence("doors", dxf_counts, vision_counts),
                "security_cameras": self._calculate_confidence("security_cameras", dxf_counts, vision_counts),
                "furniture": self._calculate_furniture_confidence(dxf_counts, vision_counts)
            },
            "source_details": {
                "dxf": self._extract_source_details(dxf_counts),
                "vision": self._extract_source_details(vision_counts)
            }
        }
        
        return reconciled_counts
        
    def _reconcile_category(self, category: str, dxf_counts: Dict[str, Any], vision_counts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile counts for a specific category.
        
        Args:
            category (str): Category name
            dxf_counts (dict): DXF analysis counts
            vision_counts (dict): Vision API counts
            
        Returns:
            dict: Reconciled count data
        """
        dxf_count = self._extract_count(dxf_counts, category)
        vision_count = self._extract_count(vision_counts, category)
        
        # Get confidence levels
        dxf_confidence = self._get_confidence(dxf_counts, category)
        vision_confidence = self._get_confidence(vision_counts, category)
        
        # Weighted average based on confidence
        confidence_weights = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4
        }
        
        dxf_weight = confidence_weights.get(dxf_confidence, 0.4)
        vision_weight = confidence_weights.get(vision_confidence, 0.4)
        
        # If one source has high confidence and significantly different count, prefer it
        if abs(dxf_count - vision_count) > min(dxf_count, vision_count) * 0.5:
            if dxf_confidence == "high" and vision_confidence != "high":
                final_count = dxf_count
            elif vision_confidence == "high" and dxf_confidence != "high":
                final_count = vision_count
            else:
                # Weighted average
                final_count = round(
                    (dxf_count * dxf_weight + vision_count * vision_weight) / 
                    (dxf_weight + vision_weight)
                )
        else:
            # Counts are similar, use weighted average
            final_count = round(
                (dxf_count * dxf_weight + vision_count * vision_weight) / 
                (dxf_weight + vision_weight)
            )
        
        return {
            "count": final_count,
            "sources": {
                "dxf": {"count": dxf_count, "confidence": dxf_confidence},
                "vision": {"count": vision_count, "confidence": vision_confidence}
            }
        }
        
    def _reconcile_furniture(self, dxf_counts: Dict[str, Any], vision_counts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile furniture counts from both sources.
        
        Args:
            dxf_counts (dict): DXF analysis counts
            vision_counts (dict): Vision API counts
            
        Returns:
            dict: Reconciled furniture counts
        """
        furniture_types = ["tables", "chairs", "cabinets", "other"]
        reconciled_furniture = {}
        
        for furniture_type in furniture_types:
            dxf_furniture = self._extract_furniture_count(dxf_counts, furniture_type)
            vision_furniture = self._extract_furniture_count(vision_counts, furniture_type)
            
            dxf_confidence = self._get_furniture_confidence(dxf_counts, furniture_type)
            vision_confidence = self._get_furniture_confidence(vision_counts, furniture_type)
            
            # Use the same weighted average approach as category reconciliation
            confidence_weights = {
                "high": 1.0,
                "medium": 0.7,
                "low": 0.4
            }
            
            dxf_weight = confidence_weights.get(dxf_confidence, 0.4)
            vision_weight = confidence_weights.get(vision_confidence, 0.4)
            
            final_count = round(
                (dxf_furniture * dxf_weight + vision_furniture * vision_weight) / 
                (dxf_weight + vision_weight)
            )
            
            reconciled_furniture[furniture_type] = {
                "count": final_count,
                "sources": {
                    "dxf": {"count": dxf_furniture, "confidence": dxf_confidence},
                    "vision": {"count": vision_furniture, "confidence": vision_confidence}
                }
            }
        
        return reconciled_furniture
        
    def _calculate_confidence(self, category: str, dxf_counts: Dict[str, Any], vision_counts: Dict[str, Any]) -> str:
        """
        Calculate overall confidence for a category.
        
        Args:
            category (str): Category name
            dxf_counts (dict): DXF analysis counts
            vision_counts (dict): Vision API counts
            
        Returns:
            str: Confidence level (high/medium/low)
        """
        dxf_confidence = self._get_confidence(dxf_counts, category)
        vision_confidence = self._get_confidence(vision_counts, category)
        
        confidence_scores = {"high": 3, "medium": 2, "low": 1}
        dxf_score = confidence_scores.get(dxf_confidence, 1)
        vision_score = confidence_scores.get(vision_confidence, 1)
        
        # Calculate average confidence score
        avg_score = (dxf_score + vision_score) / 2
        
        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"
            
    def _calculate_furniture_confidence(self, dxf_counts: Dict[str, Any], vision_counts: Dict[str, Any]) -> Dict[str, str]:
        """
        Calculate confidence levels for furniture counts.
        
        Args:
            dxf_counts (dict): DXF analysis counts
            vision_counts (dict): Vision API counts
            
        Returns:
            dict: Confidence levels for each furniture type
        """
        furniture_types = ["tables", "chairs", "cabinets", "other"]
        confidence_levels = {}
        
        for furniture_type in furniture_types:
            dxf_confidence = self._get_furniture_confidence(dxf_counts, furniture_type)
            vision_confidence = self._get_furniture_confidence(vision_counts, furniture_type)
            
            confidence_scores = {"high": 3, "medium": 2, "low": 1}
            dxf_score = confidence_scores.get(dxf_confidence, 1)
            vision_score = confidence_scores.get(vision_confidence, 1)
            
            avg_score = (dxf_score + vision_score) / 2
            
            if avg_score >= 2.5:
                confidence_levels[furniture_type] = "high"
            elif avg_score >= 1.5:
                confidence_levels[furniture_type] = "medium"
            else:
                confidence_levels[furniture_type] = "low"
        
        return confidence_levels
        
    def _extract_count(self, counts: Dict[str, Any], category: str) -> int:
        """Extract count for a category from the counts dictionary."""
        try:
            if category in counts:
                return counts[category].get("count", 0)
            return 0
        except (AttributeError, KeyError):
            return 0
            
    def _extract_furniture_count(self, counts: Dict[str, Any], furniture_type: str) -> int:
        """Extract count for a specific furniture type."""
        try:
            if "furniture" in counts and furniture_type in counts["furniture"]:
                return counts["furniture"][furniture_type].get("count", 0)
            return 0
        except (AttributeError, KeyError):
            return 0
            
    def _get_confidence(self, counts: Dict[str, Any], category: str) -> str:
        """Get confidence level for a category."""
        try:
            if category in counts:
                return counts[category].get("confidence", "low")
            return "low"
        except (AttributeError, KeyError):
            return "low"
            
    def _get_furniture_confidence(self, counts: Dict[str, Any], furniture_type: str) -> str:
        """Get confidence level for a furniture type."""
        try:
            if "furniture" in counts and furniture_type in counts["furniture"]:
                return counts["furniture"][furniture_type].get("confidence", "low")
            return "low"
        except (AttributeError, KeyError):
            return "low"
            
    def _extract_source_details(self, counts: Dict[str, Any]) -> Dict[str, Any]:
        """Extract source-specific details from counts."""
        return {
            "metadata": counts.get("metadata", {}),
            "page_details": counts.get("page_details", []),
            "processing_info": counts.get("processing_info", {})
        }
        
    def export_reconciled_counts(self, reconciled_counts: Dict[str, Any], output_file: str) -> None:
        """
        Export reconciled counts to a JSON file.
        
        Args:
            reconciled_counts (dict): Reconciled count data
            output_file (str): Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(reconciled_counts, f, indent=2)
            logger.info(f"Exported reconciled counts to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting reconciled counts: {e}")
            raise 