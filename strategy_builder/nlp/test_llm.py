"""
Test LLM Strategy Parser
"""

from strategy_builder.nlp.llm_parser import StrategyLLMParser, parse_with_llm


def test_simple_strategy():
    """Test parsing a simple strategy."""
    
    description = """
    Buy when price crosses above the 20-period moving average and RSI is below 30.
    Exit when price hits 2:1 risk-reward or RSI goes above 70.
    Use 1.5 ATR for stop loss. Only trade in uptrends on higher timeframe.
    Trade EURUSD on H1 timeframe.
    """
    
    print("="*60)
    print("Testing LLM Strategy Parser")
    print("="*60)
    print(f"\nStrategy Description:\n{description}")
    
    # Parse with LLM
    result = parse_with_llm(description, use_production=False)
    
    print("\n" + "="*60)
    print("Parsed Result:")
    print("="*60)
    
    import json
    print(json.dumps(result, indent=2))


def test_complex_strategy():
    """Test parsing a more complex strategy."""
    
    description = """
    I want to trade when the 50 EMA crosses above the 200 EMA on the daily chart,
    and the RSI is between 40 and 60. Enter on the H4 timeframe when price pulls 
    back to the 20 SMA and forms a bullish pin bar. Stop loss should be below the 
    pin bar low, and take profit at 3 times the risk. Only trade during London and 
    New York sessions. Trade GBPUSD, EURUSD, and USDJPY.
    """
    
    print("\n\n" + "="*60)
    print("Testing Complex Strategy")
    print("="*60)
    print(f"\nStrategy Description:\n{description}")
    
    # Parse with LLM
    parser = StrategyLLMParser(use_production=False)
    result = parser.parse_strategy(description)
    
    print("\n" + "="*60)
    print("Parsed Result:")
    print("="*60)
    
    import json
    print(json.dumps(result, indent=2))
    
    # Validate strategy
    print("\n" + "="*60)
    print("Strategy Validation:")
    print("="*60)
    
    validation = parser.validate_strategy(description)
    print(json.dumps(validation, indent=2))
    
    # Get suggestions
    print("\n" + "="*60)
    print("Improvement Suggestions:")
    print("="*60)
    
    suggestions = parser.suggest_improvements(description)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")


if __name__ == "__main__":
    test_simple_strategy()
    test_complex_strategy()
