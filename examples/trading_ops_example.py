"""
Example: Trading Operations Analysis with DSLA

This example demonstrates how to use the trading_ops adapter
to analyze market data and generate trading insights.
"""

from dsla.adapters.trading_ops import TradingOpsAdapter


def main():
    """Run trading operations analysis example."""
    print("=" * 80)
    print("DSLA Example: Trading Operations Analysis")
    print("=" * 80)
    print()
    
    # Initialize the trading operations adapter
    adapter = TradingOpsAdapter()
    
    # Sample market data
    market_data = {
        "current_price": 45250.00,
        "open": 44800.00,
        "high": 45500.00,
        "low": 44200.00,
        "volume": 12345678,
        "indicators": {
            "rsi_14": 62.5,
            "macd": {"value": 125.3, "signal": 98.7, "histogram": 26.6},
            "ema_20": 44950.00,
            "ema_50": 43800.00,
            "ema_200": 41200.00
        }
    }
    
    # Prepare input data
    input_data = {
        "asset": "btc/usd",
        "timeframe": "1d",
        "market_data": market_data,
        "analysis_type": "technical"
    }
    
    # Validate input
    print("1. Validating input...")
    try:
        adapter.validate_input(input_data)
        print("   ✓ Input validation passed")
    except ValueError as e:
        print(f"   ✗ Input validation failed: {e}")
        return
    
    # Adapt input
    print("\n2. Adapting input...")
    adapted_input = adapter.adapt_input(input_data)
    print(f"   Asset: {adapted_input['asset']}")
    print(f"   Timeframe: {adapted_input['timeframe']}")
    print(f"   Analysis type: {adapted_input['analysis_type']}")
    
    # Generate prompt
    print("\n3. Generating prompt...")
    prompt = adapter.format_prompt(**adapted_input)
    print(f"   Prompt length: {len(prompt)} characters")
    print("\n   Prompt preview:")
    print("   " + "-" * 76)
    for line in prompt.split('\n')[:10]:
        print(f"   {line}")
    print("   ...")
    
    # Get available tools
    print("\n4. Available tools:")
    tools = adapter.get_tools()
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    
    # Get schema
    print("\n5. Schema information:")
    schema = adapter.get_schema()
    print(f"   Required input fields: {schema.input_schema.get('required', [])}")
    print(f"   Required output fields: {schema.output_schema.get('required', [])}")
    
    # Simulate output (in real scenario, this would come from LLM)
    print("\n6. Simulating analysis output...")
    simulated_output = {
        "trend": "bullish",
        "support_levels": [44200.00, 43800.00, 42500.00],
        "resistance_levels": [45500.00, 46800.00, 48000.00],
        "signals": [
            {
                "type": "buy",
                "strength": "moderate",
                "description": "Price above EMA-20 and EMA-50, indicating uptrend"
            },
            {
                "type": "buy",
                "strength": "weak",
                "description": "RSI at 62.5 shows room for upward movement before overbought"
            },
            {
                "type": "buy",
                "strength": "moderate",
                "description": "MACD histogram positive and increasing"
            }
        ],
        "risk_score": 4.5,
        "recommendations": [
            "Consider entry near support level at $44,200",
            "Set stop loss below $43,800 (EMA-50)",
            "Take profit targets: $45,500 (resistance 1) and $46,800 (resistance 2)",
            "Monitor RSI - exit if it reaches overbought territory (>70)",
            "Watch for volume confirmation on breakout above $45,500"
        ],
        "summary": "BTC/USD shows bullish momentum with price above key moving averages. Technical indicators support continued upward movement with moderate risk.",
        "asset": "BTC/USD",
        "timeframe": "1d"
    }
    
    # Adapt output
    adapted_output = adapter.adapt_output(simulated_output)
    
    print("\n7. Analysis Results:")
    print("   " + "=" * 76)
    print(f"\n   Market Trend: {adapted_output['trend'].upper()}")
    print(f"   Risk Score: {adapted_output['risk_score']}/10")
    
    print(f"\n   Summary: {adapted_output['summary']}")
    
    print(f"\n   Support Levels:")
    for level in adapted_output['support_levels']:
        print(f"   - ${level:,.2f}")
    
    print(f"\n   Resistance Levels:")
    for level in adapted_output['resistance_levels']:
        print(f"   - ${level:,.2f}")
    
    print(f"\n   Trading Signals ({len(adapted_output['signals'])}):")
    for signal in adapted_output['signals']:
        print(f"   - [{signal['type'].upper()}] {signal['strength'].upper()}: {signal['description']}")
    
    print(f"\n   Recommendations ({len(adapted_output['recommendations'])}):")
    for rec in adapted_output['recommendations']:
        print(f"   - {rec}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
