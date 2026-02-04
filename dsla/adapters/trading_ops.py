"""Trading Operations Adapter - Domain-specific adapter for trading operations."""

from typing import Any, Dict, List
from dsla.adapters.base import Adapter, AdapterSchema, ToolDefinition


class TradingOpsAdapter(Adapter):
    """
    Adapter for trading operations analysis and decision support.
    
    Provides specialized tools and templates for:
    - Market analysis
    - Trade signal generation
    - Risk management
    - Portfolio optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the trading operations adapter."""
        super().__init__(domain="trading_ops", config=config)
    
    def get_prompt_template(self) -> str:
        """Get the trading operations prompt template."""
        return """You are a trading operations expert. Analyze the following market data and provide actionable insights.

Asset: {asset}
Timeframe: {timeframe}
Market Data:
{market_data}

Analysis Type: {analysis_type}

Please provide:
1. Market trend analysis
2. Key support and resistance levels
3. Risk assessment
4. Trading recommendations

Analysis:"""
    
    def get_schema(self) -> AdapterSchema:
        """Get the trading operations schema."""
        return AdapterSchema(
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {
                        "type": "string",
                        "description": "Trading asset symbol (e.g., BTC/USD, AAPL, EUR/USD)"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Analysis timeframe (e.g., 1h, 4h, 1d)",
                        "default": "1d"
                    },
                    "market_data": {
                        "type": "object",
                        "description": "Market data including price, volume, indicators"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis (technical, fundamental, sentiment)",
                        "default": "technical"
                    }
                },
                "required": ["asset", "market_data"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "trend": {
                        "type": "string",
                        "enum": ["bullish", "bearish", "neutral"],
                        "description": "Overall market trend"
                    },
                    "support_levels": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Key support price levels"
                    },
                    "resistance_levels": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Key resistance price levels"
                    },
                    "signals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "strength": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        },
                        "description": "Trading signals"
                    },
                    "risk_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "Risk score (0-10)"
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Trading recommendations"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Overall analysis summary"
                    }
                },
                "required": ["trend", "summary"]
            }
        )
    
    def get_tools(self) -> List[ToolDefinition]:
        """Get trading operations tools."""
        return [
            ToolDefinition(
                name="calculate_indicators",
                description="Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of indicators to calculate"
                        },
                        "period": {
                            "type": "integer",
                            "description": "Period for calculation"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="backtest_strategy",
                description="Backtest a trading strategy with historical data",
                parameters={
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "description": "Strategy description or identifier"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date for backtest"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date for backtest"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="calculate_risk_metrics",
                description="Calculate risk metrics (VaR, Sharpe ratio, max drawdown, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Risk metrics to calculate"
                        },
                        "portfolio": {
                            "type": "object",
                            "description": "Portfolio positions"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="optimize_portfolio",
                description="Optimize portfolio allocation",
                parameters={
                    "type": "object",
                    "properties": {
                        "assets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Assets to include in optimization"
                        },
                        "constraints": {
                            "type": "object",
                            "description": "Portfolio constraints"
                        },
                        "objective": {
                            "type": "string",
                            "enum": ["maximize_return", "minimize_risk", "sharpe_ratio"],
                            "description": "Optimization objective"
                        }
                    }
                }
            )
        ]
    
    def adapt_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt input for trading operations."""
        # Ensure default values
        if "timeframe" not in raw_input:
            raw_input["timeframe"] = "1d"
        
        if "analysis_type" not in raw_input:
            raw_input["analysis_type"] = "technical"
        
        # Normalize asset symbol
        if "asset" in raw_input:
            raw_input["asset"] = raw_input["asset"].upper()
        
        return raw_input
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt output from trading operations."""
        # Ensure all required fields are present with defaults
        adapted = {
            "trend": raw_output.get("trend", "neutral"),
            "support_levels": raw_output.get("support_levels", []),
            "resistance_levels": raw_output.get("resistance_levels", []),
            "signals": raw_output.get("signals", []),
            "risk_score": raw_output.get("risk_score", 5.0),
            "recommendations": raw_output.get("recommendations", []),
            "summary": raw_output.get("summary", "Analysis completed"),
            "metadata": {
                "asset": raw_output.get("asset", "unknown"),
                "timeframe": raw_output.get("timeframe", "1d"),
                "analysis_timestamp": raw_output.get("timestamp")
            }
        }
        return adapted
