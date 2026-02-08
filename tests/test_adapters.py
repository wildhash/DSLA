"""Tests for the Adapter base class and implementations."""

import pytest
from dsla.adapters.base import Adapter, AdapterSchema, ToolDefinition
from dsla.adapters.legal_doc import LegalDocAdapter
from dsla.adapters.trading_ops import TradingOpsAdapter


class TestAdapter:
    """Test base Adapter functionality."""
    
    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}}
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "type" in tool.parameters
    
    def test_adapter_schema_creation(self):
        """Test creating an adapter schema."""
        schema = AdapterSchema(
            input_schema={"type": "object", "required": ["field1"]},
            output_schema={"type": "object", "required": ["field2"]}
        )
        
        assert "required" in schema.input_schema
        assert "required" in schema.output_schema


class TestLegalDocAdapter:
    """Test LegalDocAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create a legal doc adapter instance."""
        return LegalDocAdapter()
    
    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.domain == "legal_doc"
        assert adapter.config == {}
    
    def test_get_prompt_template(self, adapter):
        """Test getting prompt template."""
        template = adapter.get_prompt_template()
        assert isinstance(template, str)
        assert "{document_type}" in template
        assert "{content}" in template
        assert "{analysis_focus}" in template
    
    def test_format_prompt(self, adapter):
        """Test formatting prompt with variables."""
        prompt = adapter.format_prompt(
            document_type="contract",
            content="Sample content",
            analysis_focus="risk assessment"
        )
        
        assert "contract" in prompt
        assert "Sample content" in prompt
        assert "risk assessment" in prompt
        assert "{" not in prompt  # No unformatted placeholders
    
    def test_get_schema(self, adapter):
        """Test getting adapter schema."""
        schema = adapter.get_schema()
        
        assert isinstance(schema, AdapterSchema)
        assert "document_type" in schema.input_schema["properties"]
        assert "content" in schema.input_schema["properties"]
        assert "document_type" in schema.input_schema["required"]
    
    def test_get_tools(self, adapter):
        """Test getting available tools."""
        tools = adapter.get_tools()
        
        assert len(tools) > 0
        assert all(isinstance(tool, ToolDefinition) for tool in tools)
        
        tool_names = [tool.name for tool in tools]
        assert "extract_clauses" in tool_names
        assert "assess_risk" in tool_names
        assert "check_compliance" in tool_names
    
    def test_validate_input_success(self, adapter):
        """Test successful input validation."""
        valid_input = {
            "document_type": "contract",
            "content": "Sample contract text"
        }
        
        assert adapter.validate_input(valid_input) is True
    
    def test_validate_input_missing_field(self, adapter):
        """Test input validation with missing required field."""
        invalid_input = {
            "document_type": "contract"
            # Missing 'content'
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            adapter.validate_input(invalid_input)
    
    def test_adapt_input(self, adapter):
        """Test input adaptation."""
        raw_input = {
            "document_type": "CONTRACT",
            "content": "Sample content"
        }
        
        adapted = adapter.adapt_input(raw_input)
        
        assert adapted["document_type"] == "contract"  # Normalized to lowercase
        assert "analysis_focus" in adapted  # Default added
    
    def test_adapt_output(self, adapter):
        """Test output adaptation."""
        raw_output = {
            "key_clauses": ["clause1"],
            "risks": ["risk1"],
            "summary": "Test summary"
        }
        
        adapted = adapter.adapt_output(raw_output)
        
        assert "key_clauses" in adapted
        assert "risks" in adapted
        assert "summary" in adapted
        assert "metadata" in adapted


class TestTradingOpsAdapter:
    """Test TradingOpsAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create a trading ops adapter instance."""
        return TradingOpsAdapter()
    
    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.domain == "trading_ops"
        assert adapter.config == {}
    
    def test_get_prompt_template(self, adapter):
        """Test getting prompt template."""
        template = adapter.get_prompt_template()
        assert isinstance(template, str)
        assert "{asset}" in template
        assert "{timeframe}" in template
        assert "{market_data}" in template
    
    def test_format_prompt(self, adapter):
        """Test formatting prompt with variables."""
        prompt = adapter.format_prompt(
            asset="BTC/USD",
            timeframe="1d",
            market_data={"price": 45000},
            analysis_type="technical"
        )
        
        assert "BTC/USD" in prompt
        assert "1d" in prompt
        # Note: market_data dict will be in the prompt as a dict representation
    
    def test_get_schema(self, adapter):
        """Test getting adapter schema."""
        schema = adapter.get_schema()
        
        assert isinstance(schema, AdapterSchema)
        assert "asset" in schema.input_schema["properties"]
        assert "market_data" in schema.input_schema["properties"]
        assert "trend" in schema.output_schema["properties"]
    
    def test_get_tools(self, adapter):
        """Test getting available tools."""
        tools = adapter.get_tools()
        
        assert len(tools) > 0
        
        tool_names = [tool.name for tool in tools]
        assert "calculate_indicators" in tool_names
        assert "backtest_strategy" in tool_names
        assert "calculate_risk_metrics" in tool_names
        assert "optimize_portfolio" in tool_names
    
    def test_validate_input_success(self, adapter):
        """Test successful input validation."""
        valid_input = {
            "asset": "BTC/USD",
            "market_data": {"price": 45000}
        }
        
        assert adapter.validate_input(valid_input) is True
    
    def test_adapt_input(self, adapter):
        """Test input adaptation."""
        raw_input = {
            "asset": "btc/usd",
            "market_data": {"price": 45000}
        }
        
        adapted = adapter.adapt_input(raw_input)
        
        assert adapted["asset"] == "BTC/USD"  # Normalized to uppercase
        assert "timeframe" in adapted  # Default added
        assert adapted["timeframe"] == "1d"
    
    def test_adapt_output(self, adapter):
        """Test output adaptation."""
        raw_output = {
            "trend": "bullish",
            "summary": "Test summary"
        }
        
        adapted = adapter.adapt_output(raw_output)
        
        assert "trend" in adapted
        assert "summary" in adapted
        assert "metadata" in adapted
        assert "support_levels" in adapted
        assert "resistance_levels" in adapted
