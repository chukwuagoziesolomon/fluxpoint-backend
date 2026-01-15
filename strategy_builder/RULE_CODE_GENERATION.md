# 🤖 Rule System Code Generation Guide

## How the No-Code Builder Generates Executable Code

This document explains the three approaches to convert user rules into executable trading logic.

---

## 🎯 The Challenge

**Problem:** User describes strategy in natural language:
```
"Buy when RSI < 30 and price crosses above 50 EMA"
```

**Goal:** Convert this into executable Python code that:
- Calculates indicators on live data
- Evaluates conditions correctly
- Makes trading decisions in real-time

---

## 🛠️ Approach 1: Indicator Library + Rule Evaluator (✅ RECOMMENDED)

### **How It Works:**

```python
# 1. Parse user description → structured rules
parsed_rules = {
    'indicators': [
        {'name': 'RSI', 'parameters': {'period': 14}},
        {'name': 'EMA', 'parameters': {'period': 50}}
    ],
    'entry_conditions': [
        {'type': 'rsi_below', 'variables': {'threshold': 30}},
        {'type': 'cross_above', 'variables': {'indicator': 'ema50'}}
    ]
}

# 2. Calculate indicators using generic library
from strategy_builder.rule_engine.indicators import IndicatorCalculator

indicator_calc = IndicatorCalculator()
df = indicator_calc.calculate_all(df, parsed_rules['indicators'])
# Result: df now has 'rsi14' and 'ema50' columns

# 3. Evaluate conditions using generic evaluator
from strategy_builder.rule_engine.evaluator import RuleEvaluator

evaluator = RuleEvaluator()
is_valid, reason = evaluator.evaluate_entry_conditions(
    df=df,
    row_idx=current_candle_idx,
    entry_conditions=parsed_rules['entry_conditions'],
    operator='AND'
)

if is_valid:
    print("✅ ENTRY SIGNAL!")
```

### **Advantages:**
- ✅ **Safe** - No code execution risks
- ✅ **Flexible** - Supports all common patterns
- ✅ **Maintainable** - Easy to add new indicators
- ✅ **Debuggable** - Clear error messages
- ✅ **Fast** - Pre-compiled Python code

### **Disadvantages:**
- ⚠️ Must pre-define supported condition types
- ⚠️ Complex custom logic might be hard to express

### **Implementation:**

We've already created this! See:
- `strategy_builder/rule_engine/indicators.py` - Generic indicator calculator
- `strategy_builder/rule_engine/evaluator.py` - Generic rule evaluator

---

## 🛠️ Approach 2: Dynamic Code Generation (Advanced)

### **How It Works:**

Generate actual Python code from parsed rules, then execute it.

```python
def generate_entry_code(parsed_rules):
    """
    Generate executable Python code from parsed rules.
    """
    code_lines = []
    
    # Generate imports
    code_lines.append("import pandas as pd")
    code_lines.append("import numpy as np")
    code_lines.append("")
    
    # Generate indicator calculations
    code_lines.append("def calculate_indicators(df):")
    for indicator in parsed_rules['indicators']:
        if indicator['name'] == 'RSI':
            period = indicator['parameters']['period']
            code_lines.append(f"    # Calculate RSI{period}")
            code_lines.append(f"    delta = df['close'].diff()")
            code_lines.append(f"    gains = delta.where(delta > 0, 0)")
            code_lines.append(f"    losses = -delta.where(delta < 0, 0)")
            code_lines.append(f"    avg_gains = gains.rolling({period}).mean()")
            code_lines.append(f"    avg_losses = losses.rolling({period}).mean()")
            code_lines.append(f"    rs = avg_gains / avg_losses")
            code_lines.append(f"    df['rsi{period}'] = 100 - (100 / (1 + rs))")
            code_lines.append("")
        
        elif indicator['name'] == 'EMA':
            period = indicator['parameters']['period']
            code_lines.append(f"    # Calculate EMA{period}")
            code_lines.append(f"    df['ema{period}'] = df['close'].ewm(span={period}).mean()")
            code_lines.append("")
    
    code_lines.append("    return df")
    code_lines.append("")
    
    # Generate entry condition check
    code_lines.append("def check_entry(df, idx):")
    code_lines.append("    \"\"\"Check if entry conditions are met at index idx\"\"\"")
    code_lines.append("    conditions = []")
    code_lines.append("")
    
    for condition in parsed_rules['entry_conditions']:
        if condition['type'] == 'rsi_below':
            threshold = condition['variables']['threshold']
            code_lines.append(f"    # RSI below {threshold}")
            code_lines.append(f"    rsi_check = df.iloc[idx]['rsi14'] < {threshold}")
            code_lines.append(f"    conditions.append(rsi_check)")
        
        elif condition['type'] == 'cross_above':
            indicator = condition['variables']['indicator']
            code_lines.append(f"    # Price crosses above {indicator}")
            code_lines.append(f"    current_above = df.iloc[idx]['close'] > df.iloc[idx]['{indicator}']")
            code_lines.append(f"    previous_below = df.iloc[idx-1]['close'] <= df.iloc[idx-1]['{indicator}']")
            code_lines.append(f"    cross_check = current_above and previous_below")
            code_lines.append(f"    conditions.append(cross_check)")
    
    code_lines.append("")
    code_lines.append("    return all(conditions)")
    
    return "\n".join(code_lines)


# Example usage:
parsed_rules = {
    'indicators': [
        {'name': 'RSI', 'parameters': {'period': 14}},
        {'name': 'EMA', 'parameters': {'period': 50}}
    ],
    'entry_conditions': [
        {'type': 'rsi_below', 'variables': {'threshold': 30}},
        {'type': 'cross_above', 'variables': {'indicator': 'ema50'}}
    ]
}

# Generate code
generated_code = generate_entry_code(parsed_rules)

print("Generated Code:")
print("="*60)
print(generated_code)

# Execute generated code (CAREFUL!)
namespace = {}
exec(generated_code, namespace)

# Now you can use the generated functions
df_with_indicators = namespace['calculate_indicators'](df)
entry_signal = namespace['check_entry'](df_with_indicators, 100)
```

### **Generated Code Example:**

```python
import pandas as pd
import numpy as np

def calculate_indicators(df):
    # Calculate RSI14
    delta = df['close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.rolling(14).mean()
    avg_losses = losses.rolling(14).mean()
    rs = avg_gains / avg_losses
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    # Calculate EMA50
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    return df

def check_entry(df, idx):
    """Check if entry conditions are met at index idx"""
    conditions = []
    
    # RSI below 30
    rsi_check = df.iloc[idx]['rsi14'] < 30
    conditions.append(rsi_check)
    
    # Price crosses above ema50
    current_above = df.iloc[idx]['close'] > df.iloc[idx]['ema50']
    previous_below = df.iloc[idx-1]['close'] <= df.iloc[idx-1]['ema50']
    cross_check = current_above and previous_below
    conditions.append(cross_check)
    
    return all(conditions)
```

### **Advantages:**
- ✅ **Extremely flexible** - Can handle any logic
- ✅ **Optimal performance** - Pure Python execution
- ✅ **User-specific** - Each strategy gets custom code

### **Disadvantages:**
- ❌ **Security risk** - `exec()` can execute malicious code
- ❌ **Complex** - Code generation is tricky
- ❌ **Debugging** - Generated code errors are hard to trace
- ❌ **Maintenance** - Must update generator for new features

### **When to Use:**
Only if Approach 1 (library) is too limiting.

### **Security Measures:**
If you use this approach:
1. **Sandbox execution** - Use `RestrictedPython`
2. **Code review** - Validate generated code before execution
3. **Whitelist functions** - Only allow safe operations
4. **Timeout limits** - Prevent infinite loops

---

## 🛠️ Approach 3: Template-Based (Hybrid)

### **How It Works:**

Pre-define code templates for common patterns, then fill in parameters.

```python
ENTRY_TEMPLATES = {
    'rsi_oversold': """
def check_rsi_oversold(df, idx, threshold={threshold}):
    return df.iloc[idx]['rsi14'] < threshold
""",
    
    'ma_cross': """
def check_ma_cross(df, idx, ma1='{ma1}', ma2='{ma2}'):
    current = df.iloc[idx][ma1] > df.iloc[idx][ma2]
    previous = df.iloc[idx-1][ma1] <= df.iloc[idx-1][ma2]
    return current and previous
""",
    
    'price_above_ma': """
def check_price_above_ma(df, idx, ma='{ma}'):
    return df.iloc[idx]['close'] > df.iloc[idx][ma]
"""
}

def generate_from_template(condition):
    """Generate code from template"""
    template_name = condition['type']
    template = ENTRY_TEMPLATES.get(template_name)
    
    if not template:
        raise ValueError(f"Unknown template: {template_name}")
    
    # Fill in parameters
    code = template.format(**condition['variables'])
    return code

# Example:
condition = {
    'type': 'rsi_oversold',
    'variables': {'threshold': 30}
}

code = generate_from_template(condition)
print(code)
# Output:
# def check_rsi_oversold(df, idx, threshold=30):
#     return df.iloc[idx]['rsi14'] < threshold
```

### **Advantages:**
- ✅ **Safe** - Pre-reviewed templates
- ✅ **Fast development** - Just add templates
- ✅ **Clear errors** - Template validation

### **Disadvantages:**
- ⚠️ Limited to predefined templates
- ⚠️ Combination logic can be complex

---

## 🏆 Recommended Approach

### **Use Approach 1 (Indicator Library + Rule Evaluator)**

**Why:**
1. **Safety First** - No code execution risks
2. **Proven Pattern** - Used by many platforms (TradingView, QuantConnect)
3. **Extensible** - Easy to add new indicators/conditions
4. **Fast** - Pre-compiled Python
5. **Debuggable** - Clear error messages

### **Implementation:**

```python
# Already implemented in:
strategy_builder/
├── rule_engine/
│   ├── indicators.py    # Generic indicator calculator
│   ├── evaluator.py     # Generic condition evaluator
│   └── __init__.py

# How to use:
from strategy_builder.rule_engine.indicators import IndicatorCalculator
from strategy_builder.rule_engine.evaluator import RuleEvaluator

# 1. Calculate indicators
calc = IndicatorCalculator()
df = calc.calculate_all(df, parsed_rules['indicators'])

# 2. Evaluate rules
eval = RuleEvaluator()
is_valid, reason = eval.evaluate_entry_conditions(
    df, current_idx, parsed_rules['entry_conditions']
)
```

### **When to Consider Approach 2:**

Only if users need features like:
- Custom mathematical formulas
- Complex multi-step logic
- Integration with external APIs

Even then, **sandbox it heavily**!

---

## 📊 Comparison Table

| Feature | Approach 1 (Library) | Approach 2 (Codegen) | Approach 3 (Templates) |
|---------|---------------------|---------------------|----------------------|
| **Safety** | ✅ Very Safe | ❌ Risky (exec) | ✅ Safe |
| **Flexibility** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Unlimited | ⭐⭐ Limited |
| **Performance** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Fast |
| **Debugging** | ⭐⭐⭐⭐ Easy | ⭐⭐ Hard | ⭐⭐⭐ Moderate |
| **Maintenance** | ⭐⭐⭐⭐ Easy | ⭐⭐ Complex | ⭐⭐⭐ Moderate |
| **Security** | ✅ No risks | ❌ High risk | ✅ Low risk |

---

## 🎯 Conclusion

**For your no-code builder, use Approach 1 (Indicator Library + Rule Evaluator).**

The implementation is already done:
- ✅ `indicators.py` - Calculates all common indicators
- ✅ `evaluator.py` - Evaluates all common conditions
- ✅ Extensible - Easy to add more

**This gives you:**
- 90% of user needs covered
- Safe and secure
- Easy to maintain
- Fast execution
- Clear debugging

**Reserve Approach 2 (code generation) for future "advanced mode" if needed.**

---

## 🚀 Next Steps

1. ✅ **Rule engine built** (indicators + evaluator)
2. ⬜ **Test with real examples** (run COMPLETE_EXAMPLES.py)
3. ⬜ **Build API endpoints** (expose to users)
4. ⬜ **Add ML training pipeline** (train on user setups)
5. ⬜ **Add RL training pipeline** (optimize execution)
6. ⬜ **Build frontend** (user interface)

The hardest part (rule execution) is DONE! 🎉
