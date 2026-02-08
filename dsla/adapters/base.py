"""Base Adapter interface for DSLA."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by the adapter."""
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")


class AdapterSchema(BaseModel):
    """Schema defining the structure of inputs and outputs for an adapter."""
    
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for input validation")
    output_schema: Dict[str, Any] = Field(..., description="JSON schema for output validation")


class Adapter(ABC):
    """
    Base Adapter class that all domain-specific adapters must inherit from.
    
    An adapter provides:
    - Domain-specific prompt templates
    - Input/output schema validation
    - Tool definitions for the domain
    - Adaptation logic for transforming inputs/outputs
    """
    
    def __init__(self, domain: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.
        
        Args:
            domain: The domain this adapter is specialized for
            config: Optional configuration dictionary
        """
        self.domain = domain
        self.config = config or {}
        self._tools: List[ToolDefinition] = []
        self._schema: Optional[AdapterSchema] = None
        self._prompt_template: str = ""
        
    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        Get the domain-specific prompt template.
        
        Returns:
            String template with placeholders for variables
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> AdapterSchema:
        """
        Get the input/output schema for this adapter.
        
        Returns:
            AdapterSchema defining expected input and output structures
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        """
        Get the list of tools available for this adapter.
        
        Returns:
            List of ToolDefinition objects
        """
        pass
    
    def adapt_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw input into domain-specific format.
        
        Args:
            raw_input: Raw input data
            
        Returns:
            Adapted input ready for processing
        """
        # Default implementation - can be overridden
        return raw_input
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw output into domain-specific format.
        
        Args:
            raw_output: Raw output from LLM
            
        Returns:
            Adapted output in domain-specific format
        """
        # Default implementation - can be overridden
        return raw_output
    
    def format_prompt(self, **kwargs) -> str:
        """
        Format the prompt template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt_template()
        return template.format(**kwargs)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against the adapter schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Basic validation - can be enhanced with jsonschema
        schema = self.get_schema()
        required_fields = schema.input_schema.get("required", [])
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data against the adapter schema.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Basic validation - can be enhanced with jsonschema
        schema = self.get_schema()
        required_fields = schema.output_schema.get("required", [])
        for field in required_fields:
            if field not in output_data:
                raise ValueError(f"Missing required field: {field}")
        return True
