"""
LLM Integration for Strategy Parsing

Uses OpenRouter API (Mistral 7B for testing, Claude Sonnet 4.5 for production)
to understand and translate user trading strategies.
"""

import requests
import json
import os
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class StrategyLLMParser:
    """
    Uses LLM to parse and understand trading strategy descriptions.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        use_production: bool = None
    ):
        """
        Args:
            api_key: OpenRouter API key (defaults to env variable)
            model: Model to use (defaults to env variable)
            use_production: If True, use production model (defaults to env variable)
        """
        # Load API key from environment
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please set it in .env file or pass as parameter."
            )
        
        # Determine which model to use
        if use_production is None:
            use_production = os.getenv('USE_PRODUCTION_LLM', 'False').lower() == 'true'
        
        if model:
            self.model = model
        elif use_production:
            self.model = os.getenv('OPENROUTER_MODEL_PROD', 'anthropic/claude-sonnet-4.5')
        else:
            self.model = os.getenv('OPENROUTER_MODEL_TEST', 'mistralai/mistral-7b-instruct:free')
        
        # API configuration
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.site_url = os.getenv('SITE_URL', 'https://fluxpointai.com')
        self.site_name = os.getenv('SITE_NAME', 'FluxPoint AI Trading Platform')
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
    
    def parse_strategy(self, description: str) -> Dict:
        """
        Use LLM to parse trading strategy description into structured format.
        
        Args:
            description: User's natural language strategy description
        
        Returns:
            Structured strategy dict
        """
        prompt = self._create_parsing_prompt(description)
        
        try:
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,  # Low temperature for consistent parsing
                    "max_tokens": 2000
                })
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract LLM response
            llm_output = result['choices'][0]['message']['content']
            
            # Parse JSON from LLM output
            parsed = self._extract_json(llm_output)
            
            return parsed
        
        except Exception as e:
            print(f"LLM parsing error: {e}")
            return {
                'error': str(e),
                'fallback': True,
                'raw_description': description
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for trading strategy parsing."""
        return """You are an expert trading strategy analyst. Your job is to parse natural language trading strategy descriptions into structured JSON format.

Extract:
1. **Indicators**: All technical indicators mentioned (MA, RSI, MACD, Bollinger Bands, ATR, etc.)
2. **Entry Conditions**: When to enter a trade (price crosses, indicator signals, etc.)
3. **Exit Conditions**: When to exit (take profit, stop loss, trailing stops, etc.)
4. **Filters**: Additional filters (trend direction, time of day, higher timeframe, etc.)
5. **Risk Management**: Stop loss, take profit, position sizing rules
6. **Timeframes**: Trading timeframes mentioned
7. **Symbols**: Currency pairs or assets to trade

Always respond with valid JSON only. No explanations, just JSON."""
    
    def _create_parsing_prompt(self, description: str) -> str:
        """Create parsing prompt for user's strategy."""
        return f"""Parse this trading strategy into structured JSON format:

"{description}"

Return JSON with this exact structure:
{{
  "strategy_name": "brief name for the strategy",
  "indicators": [
    {{"name": "MA", "parameters": {{"period": 20, "type": "SMA"}}}},
    {{"name": "RSI", "parameters": {{"period": 14}}}}
  ],
  "entry_conditions": [
    {{"type": "price_cross_above", "target": "MA20"}},
    {{"type": "rsi_oversold", "threshold": 30}}
  ],
  "exit_conditions": [
    {{"type": "take_profit", "value": 40, "unit": "pips"}},
    {{"type": "stop_loss", "value": 20, "unit": "pips"}}
  ],
  "filters": [
    {{"type": "trend", "direction": "up"}},
    {{"type": "higher_timeframe", "condition": "bullish"}}
  ],
  "risk_management": {{
    "stop_loss": {{"value": 20, "unit": "pips"}},
    "take_profit": {{"value": 40, "unit": "pips"}},
    "risk_reward": "1:2"
  }},
  "timeframes": ["H1", "D1"],
  "symbols": ["EURUSD", "GBPUSD"]
}}

Return ONLY valid JSON, no additional text."""
    
    def _extract_json(self, llm_output: str) -> Dict:
        """Extract JSON from LLM output (handles markdown code blocks)."""
        # Remove markdown code blocks if present
        if '```json' in llm_output:
            start = llm_output.find('```json') + 7
            end = llm_output.find('```', start)
            json_str = llm_output[start:end].strip()
        elif '```' in llm_output:
            start = llm_output.find('```') + 3
            end = llm_output.find('```', start)
            json_str = llm_output[start:end].strip()
        else:
            json_str = llm_output.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"LLM output: {llm_output}")
            return {
                'error': 'Failed to parse LLM output as JSON',
                'raw_output': llm_output
            }
    
    def validate_strategy(self, description: str) -> Dict:
        """
        Ask LLM to validate if a strategy description is complete.
        
        Args:
            description: Strategy description
        
        Returns:
            Validation result with suggestions
        """
        prompt = f"""Analyze this trading strategy and check if it's complete:

"{description}"

Check for:
1. Entry conditions (how to enter trades)
2. Exit conditions (take profit, stop loss)
3. Risk management rules
4. Indicators needed
5. Timeframes specified

Return JSON:
{{
  "is_complete": true/false,
  "missing_components": ["list of missing parts"],
  "suggestions": ["suggestions to improve the strategy"],
  "clarity_score": 0-10
}}"""
        
        try:
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                })
            )
            
            response.raise_for_status()
            result = response.json()
            llm_output = result['choices'][0]['message']['content']
            
            return self._extract_json(llm_output)
        
        except Exception as e:
            return {
                'error': str(e),
                'is_complete': False
            }
    
    def suggest_improvements(self, description: str) -> List[str]:
        """
        Get LLM suggestions to improve strategy clarity.
        
        Args:
            description: Strategy description
        
        Returns:
            List of improvement suggestions
        """
        prompt = f"""This user described their trading strategy:

"{description}"

Suggest 3-5 improvements to make it clearer and more specific. Focus on:
- Missing technical details
- Ambiguous conditions
- Risk management gaps
- Timeframe clarity

Return as JSON list: {{"suggestions": ["suggestion 1", "suggestion 2", ...]}}"""
        
        try:
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.5,
                    "max_tokens": 800
                })
            )
            
            response.raise_for_status()
            result = response.json()
            llm_output = result['choices'][0]['message']['content']
            
            parsed = self._extract_json(llm_output)
            return parsed.get('suggestions', [])
        
        except Exception as e:
            print(f"Suggestion error: {e}")
            return []


def parse_with_llm(description: str, use_production: bool = False) -> Dict:
    """
    Convenience function to parse strategy with LLM.
    
    Args:
        description: User's strategy description
        use_production: Use Claude Sonnet 4.5 (vs Mistral free)
    
    Returns:
        Parsed strategy dict
    """
    parser = StrategyLLMParser(use_production=use_production)
    return parser.parse_strategy(description)
