"""Legal Document Adapter - Domain-specific adapter for legal document processing."""

from typing import Any, Dict, List
from dsla.adapters.base import Adapter, AdapterSchema, ToolDefinition


class LegalDocAdapter(Adapter):
    """
    Adapter for legal document analysis and processing.
    
    Provides specialized tools and templates for:
    - Contract analysis
    - Clause extraction
    - Risk assessment
    - Compliance checking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the legal document adapter."""
        super().__init__(domain="legal_doc", config=config)
    
    def get_prompt_template(self) -> str:
        """Get the legal document prompt template."""
        return """You are a legal document analysis expert. Your task is to analyze the following legal document.

Document Type: {document_type}
Document Content:
{content}

Analysis Focus: {analysis_focus}

Please provide a comprehensive analysis including:
1. Key clauses and provisions
2. Potential risks or concerns
3. Compliance considerations
4. Recommendations

Analysis:"""
    
    def get_schema(self) -> AdapterSchema:
        """Get the legal document schema."""
        return AdapterSchema(
            input_schema={
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "Type of legal document (e.g., contract, NDA, agreement)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The legal document content"
                    },
                    "analysis_focus": {
                        "type": "string",
                        "description": "Specific aspect to focus on (e.g., risk, compliance, obligations)",
                        "default": "general"
                    }
                },
                "required": ["document_type", "content"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "key_clauses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important clauses identified"
                    },
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identified risks"
                    },
                    "compliance_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Compliance considerations"
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommendations for the document"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Overall summary of analysis"
                    }
                },
                "required": ["summary"]
            }
        )
    
    def get_tools(self) -> List[ToolDefinition]:
        """Get legal document processing tools."""
        return [
            ToolDefinition(
                name="extract_clauses",
                description="Extract specific types of clauses from legal documents",
                parameters={
                    "type": "object",
                    "properties": {
                        "clause_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of clauses to extract (e.g., termination, liability, confidentiality)"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="assess_risk",
                description="Assess legal and financial risks in the document",
                parameters={
                    "type": "object",
                    "properties": {
                        "risk_categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Risk categories to assess (e.g., financial, regulatory, operational)"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="check_compliance",
                description="Check document compliance with specific regulations",
                parameters={
                    "type": "object",
                    "properties": {
                        "regulations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Regulations to check against (e.g., GDPR, CCPA, SOX)"
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "Legal jurisdiction"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="compare_documents",
                description="Compare two legal documents for differences",
                parameters={
                    "type": "object",
                    "properties": {
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of documents to compare"
                        }
                    }
                }
            )
        ]
    
    def adapt_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt input for legal document processing."""
        # Ensure analysis_focus has a default
        if "analysis_focus" not in raw_input:
            raw_input["analysis_focus"] = "general analysis"
        
        # Normalize document type
        if "document_type" in raw_input:
            raw_input["document_type"] = raw_input["document_type"].lower()
        
        return raw_input
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt output from legal document processing."""
        # Ensure all required fields are present with defaults
        adapted = {
            "key_clauses": raw_output.get("key_clauses", []),
            "risks": raw_output.get("risks", []),
            "compliance_notes": raw_output.get("compliance_notes", []),
            "recommendations": raw_output.get("recommendations", []),
            "summary": raw_output.get("summary", "Analysis completed"),
            "metadata": {
                "document_type": raw_output.get("document_type", "unknown"),
                "analysis_timestamp": raw_output.get("timestamp")
            }
        }
        return adapted
