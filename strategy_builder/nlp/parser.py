"""
Natural Language Processing for Strategy Parsing

Converts user's natural language strategy descriptions into structured rules.
Uses hybrid approach: regex patterns + LLM for better understanding.
"""

import re
from typing import Dict, List, Tuple
import json

try:
    from .llm_parser import StrategyLLMParser
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class StrategyParser:
    """
    Parses natural language trading strategy descriptions.
    """
    
    def __init__(self):
        self.indicators = self._load_indicator_patterns()
        self.conditions = self._load_condition_patterns()
        self.actions = self._load_action_patterns()
    
    def _load_indicator_patterns(self) -> Dict:
        """Load patterns for technical indicators."""
        return {
            'moving_average': {
                'patterns': [
                    r'(?:moving average|MA)[\s\-]?(\d+)',
                    r'(\d+)[\s\-]?period (?:moving average|MA)',
                    r'(\d+)[\s\-]?(?:EMA|SMA)',
                ],
                'type': 'MA',
                'extract': ['period']
            },
            'rsi': {
                'patterns': [
                    r'RSI(?:[\s\-]?(\d+))?',
                    r'relative strength index(?:[\s\-]?(\d+))?',
                ],
                'type': 'RSI',
                'extract': ['period'],
                'default': {'period': 14}
            },
            'macd': {
                'patterns': [
                    r'MACD',
                    r'moving average convergence divergence',
                ],
                'type': 'MACD',
                'extract': []
            },
            'bollinger_bands': {
                'patterns': [
                    r'Bollinger Bands?(?:[\s\-]?(\d+))?',
                    r'BB(?:[\s\-]?(\d+))?',
                ],
                'type': 'BB',
                'extract': ['period'],
                'default': {'period': 20}
            },
            'atr': {
                'patterns': [
                    r'ATR(?:[\s\-]?(\d+))?',
                    r'average true range(?:[\s\-]?(\d+))?',
                ],
                'type': 'ATR',
                'extract': ['period'],
                'default': {'period': 14}
            },
            'stochastic': {
                'patterns': [
                    r'stochastic(?:[\s\-]?(\d+))?',
                    r'stoch(?:[\s\-]?(\d+))?',
                ],
                'type': 'STOCH',
                'extract': ['period'],
                'default': {'period': 14}
            },
        }
    
    def _load_condition_patterns(self) -> Dict:
        """Load patterns for trading conditions."""
        return {
            'price_cross_above': {
                'patterns': [
                    r'price (?:crosses?|breaks?) above (\w+)',
                    r'when price (?:goes?|moves?) above (\w+)',
                    r'price > (\w+)',
                ],
                'type': 'cross_above',
                'variables': ['indicator']
            },
            'price_cross_below': {
                'patterns': [
                    r'price (?:crosses?|breaks?) below (\w+)',
                    r'when price (?:goes?|moves?) below (\w+)',
                    r'price < (\w+)',
                ],
                'type': 'cross_below',
                'variables': ['indicator']
            },
            'indicator_cross': {
                'patterns': [
                    r'(\w+) crosses? above (\w+)',
                    r'(\w+) crosses? below (\w+)',
                ],
                'type': 'indicator_cross',
                'variables': ['indicator1', 'indicator2', 'direction']
            },
            'rsi_oversold': {
                'patterns': [
                    r'RSI (?:is )?(?:below|under|less than) (\d+)',
                    r'RSI < (\d+)',
                ],
                'type': 'rsi_oversold',
                'variables': ['threshold']
            },
            'rsi_overbought': {
                'patterns': [
                    r'RSI (?:is )?(?:above|over|greater than) (\d+)',
                    r'RSI > (\d+)',
                ],
                'type': 'rsi_overbought',
                'variables': ['threshold']
            },
            'trend_confirmation': {
                'patterns': [
                    r'(?:in |during )?(?:an? )?uptrend',
                    r'trend(?:ing)? (?:is )?up',
                    r'bullish trend',
                ],
                'type': 'uptrend',
                'variables': []
            },
            'higher_timeframe': {
                'patterns': [
                    r'(?:on |in )?(?:the )?(\w+) (?:timeframe|TF)',
                    r'higher timeframe (?:is )?(\w+)',
                ],
                'type': 'higher_tf',
                'variables': ['timeframe', 'condition']
            },
        }
    
    def _load_action_patterns(self) -> Dict:
        """Load patterns for trading actions."""
        return {
            'buy': {
                'patterns': [
                    r'(?:enter |take a? |open a? )?(?:long|buy)(?: position)?',
                    r'go long',
                ],
                'type': 'BUY'
            },
            'sell': {
                'patterns': [
                    r'(?:enter |take a? |open a? )?(?:short|sell)(?: position)?',
                    r'go short',
                ],
                'type': 'SELL'
            },
            'stop_loss': {
                'patterns': [
                    r'stop loss (?:at |of )?(\d+(?:\.\d+)?) ?(?:pips?|ATR)?',
                    r'SL (?:at |of )?(\d+(?:\.\d+)?) ?(?:pips?|ATR)?',
                    r'stop (?:at |of )?(\d+(?:\.\d+)?) ?(?:pips?|ATR)?',
                ],
                'type': 'stop_loss',
                'variables': ['value', 'unit']
            },
            'take_profit': {
                'patterns': [
                    r'take profit (?:at |of )?(\d+(?:\.\d+)?) ?(?:pips?|ATR)?',
                    r'TP (?:at |of )?(\d+(?:\.\d+)?) ?(?:pips?|ATR)?',
                    r'target (?:at |of )?(\d+(?:\.\d+)?) ?(?:pips?|ATR)?',
                ],
                'type': 'take_profit',
                'variables': ['value', 'unit']
            },
            'risk_reward': {
                'patterns': [
                    r'risk[:\s]reward (?:of |ratio )?(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)',
                    r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?) (?:RR|risk reward)',
                ],
                'type': 'risk_reward',
                'variables': ['risk', 'reward']
            },
        }
    
    def parse(self, description: str, use_llm: bool = True) -> Dict:
        """
        Parse strategy description into structured rules.
        Uses hybrid approach: LLM first, fallback to regex if LLM fails.
        
        Args:
            description: Natural language strategy description
            use_llm: Whether to use LLM parsing (default True)
        
        Returns:
            Dict with parsed components
        """
        # Try LLM parsing first (more accurate for complex strategies)
        if use_llm and LLM_AVAILABLE:
            try:
                llm_parser = StrategyLLMParser(use_production=False)  # Use Mistral for testing
                llm_result = llm_parser.parse_strategy(description)
                
                if 'error' not in llm_result:
                    # LLM parsing succeeded
                    llm_result['parsing_method'] = 'llm'
                    llm_result['validation_errors'] = self._validate_strategy(llm_result)
                    llm_result['is_valid'] = len(llm_result['validation_errors']) == 0
                    return llm_result
            except Exception as e:
                print(f"LLM parsing failed, falling back to regex: {e}")
        
        # Fallback to regex-based parsing
        description = description.lower()
        
        result = {
            'indicators': self._extract_indicators(description),
            'entry_conditions': self._extract_conditions(description, 'entry'),
            'exit_conditions': self._extract_conditions(description, 'exit'),
            'filters': self._extract_filters(description),
            'risk_management': self._extract_risk_management(description),
            'timeframes': self._extract_timeframes(description),
            'parsing_method': 'regex',
            'validation_errors': []
        }
        
        # Validate completeness
        result['validation_errors'] = self._validate_strategy(result)
        result['is_valid'] = len(result['validation_errors']) == 0
        
        return result
    
    def _extract_indicators(self, text: str) -> List[Dict]:
        """Extract indicator references from text."""
        found_indicators = []
        
        for name, config in self.indicators.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    params = {}
                    
                    # Extract parameters
                    if config['extract']:
                        for i, param_name in enumerate(config['extract']):
                            if match.groups() and i < len(match.groups()) and match.group(i + 1):
                                params[param_name] = int(match.group(i + 1))
                    
                    # Apply defaults
                    if 'default' in config:
                        for key, value in config['default'].items():
                            if key not in params:
                                params[key] = value
                    
                    found_indicators.append({
                        'name': config['type'],
                        'parameters': params,
                        'matched_text': match.group(0)
                    })
        
        return found_indicators
    
    def _extract_conditions(self, text: str, condition_type: str = 'entry') -> List[Dict]:
        """Extract trading conditions."""
        conditions = []
        
        # Look for entry/exit keywords
        if condition_type == 'entry':
            entry_phrases = re.findall(r'(?:enter|entry|buy|sell|long|short).*?(?:\.|$)', text)
            search_text = ' '.join(entry_phrases)
        else:
            exit_phrases = re.findall(r'(?:exit|close|stop|target|take profit).*?(?:\.|$)', text)
            search_text = ' '.join(exit_phrases)
        
        for name, config in self.conditions.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    condition = {
                        'type': config['type'],
                        'matched_text': match.group(0),
                        'variables': {}
                    }
                    
                    # Extract variables
                    if config['variables']:
                        for i, var_name in enumerate(config['variables']):
                            if match.groups() and i < len(match.groups()) and match.group(i + 1):
                                condition['variables'][var_name] = match.group(i + 1)
                    
                    conditions.append(condition)
        
        return conditions
    
    def _extract_filters(self, text: str) -> List[Dict]:
        """Extract filter conditions (trend, higher TF, etc.)."""
        filters = []
        
        # Trend filters
        if re.search(r'uptrend|bullish|trend(?:ing)? up', text, re.IGNORECASE):
            filters.append({'type': 'trend', 'direction': 'up'})
        elif re.search(r'downtrend|bearish|trend(?:ing)? down', text, re.IGNORECASE):
            filters.append({'type': 'trend', 'direction': 'down'})
        
        # Higher timeframe
        htf_match = re.search(r'higher (?:timeframe|TF)', text, re.IGNORECASE)
        if htf_match:
            filters.append({'type': 'higher_timeframe', 'check': 'alignment'})
        
        return filters
    
    def _extract_risk_management(self, text: str) -> Dict:
        """Extract risk management rules."""
        risk_mgmt = {}
        
        for name, config in self.actions.items():
            if config['type'] in ['stop_loss', 'take_profit', 'risk_reward']:
                for pattern in config['patterns']:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        risk_mgmt[config['type']] = {
                            'value': match.group(1) if match.groups() else None,
                            'matched_text': match.group(0)
                        }
        
        return risk_mgmt
    
    def _extract_timeframes(self, text: str) -> List[str]:
        """Extract timeframe references."""
        timeframes = []
        
        tf_patterns = [
            (r'M(\d+)', 'M{}'),
            (r'(\d+)[\s\-]?minute', 'M{}'),
            (r'H(\d+)', 'H{}'),
            (r'(\d+)[\s\-]?hour', 'H{}'),
            (r'D1|daily', 'D1'),
            (r'W1|weekly', 'W1'),
        ]
        
        for pattern, format_str in tf_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if '{}' in format_str and match.groups():
                    tf = format_str.format(match.group(1))
                else:
                    tf = format_str
                if tf not in timeframes:
                    timeframes.append(tf)
        , use_llm: bool = True) -> Dict:
    """
    Convenience function to parse a strategy description.
    
    Args:
        description: Natural language strategy description
        use_llm: Use LLM for parsing (default True, fallback to regex)
    
    Returns:
        Parsed strategy dict
    """
    parser = StrategyParser()
    return parser.parse(description, use_llm=use_llmmanagement found. Please specify stop loss and/or take profit.")
        
        if not parsed['indicators']:
            errors.append("No indicators found. Strategies typically use at least one indicator.")
        
        return errors


def parse_strategy_description(description: str) -> Dict:
    """
    Convenience function to parse a strategy description.
    
    Args:
        description: Natural language strategy description
    
    Returns:
        Parsed strategy dict
    """
    parser = StrategyParser()
    return parser.parse(description)
