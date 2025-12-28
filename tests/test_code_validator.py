"""
Tests for Code Validator
"""

import pytest
from agent_school.utils.code_validator import CodeValidator


class TestCodeValidator:
    """Test suite for CodeValidator"""

    def test_validate_syntax_valid_code(self):
        """Test syntax validation with valid code"""
        code = """
def hello():
    print("Hello, World!")
    return True
"""
        is_valid, error = CodeValidator.validate_syntax(code)
        assert is_valid is True
        assert error is None

    def test_validate_syntax_invalid_code(self):
        """Test syntax validation with invalid code"""
        code = """
def hello()
    print("Hello")  # Missing colon
"""
        is_valid, error = CodeValidator.validate_syntax(code)
        assert is_valid is False
        assert error is not None
        assert "Syntax error" in error

    def test_check_dangerous_operations(self):
        """Test detection of dangerous operations"""
        code = """
def dangerous_function():
    eval("print('dangerous')")
    exec("import os")
"""
        warnings = CodeValidator.check_dangerous_operations(code)
        assert len(warnings) > 0
        assert any("eval" in w for w in warnings)
        assert any("exec" in w for w in warnings)

    def test_check_dangerous_operations_safe_code(self):
        """Test that safe code produces no warnings"""
        code = """
def safe_function():
    return 1 + 1
"""
        warnings = CodeValidator.check_dangerous_operations(code)
        assert len(warnings) == 0

    def test_check_imports_allowed(self):
        """Test that allowed imports pass"""
        code = """
import requests
import json
from typing import List
"""
        warnings = CodeValidator.check_imports(code)
        assert len(warnings) == 0

    def test_check_imports_suspicious(self):
        """Test that suspicious imports are flagged"""
        code = """
import socket
import pickle
"""
        warnings = CodeValidator.check_imports(code)
        assert len(warnings) > 0

    def test_extract_functions(self):
        """Test function extraction from code"""
        code = """
def function1(arg1: str) -> str:
    \"\"\"Docstring for function1\"\"\"
    return arg1

def function2(arg1, arg2):
    pass

async def async_function():
    pass
"""
        functions = CodeValidator.extract_functions(code)
        assert len(functions) == 3

        # Check function1
        func1 = next(f for f in functions if f["name"] == "function1")
        assert func1["has_docstring"] is True
        assert "arg1" in func1["args"]
        assert func1["is_async"] is False

        # Check async_function
        async_func = next(f for f in functions if f["name"] == "async_function")
        assert async_func["is_async"] is True

    def test_validate_code_comprehensive(self):
        """Test comprehensive code validation"""
        code = """
import requests

def fetch_data(url: str) -> dict:
    \"\"\"Fetch data from URL\"\"\"
    response = requests.get(url)
    return response.json()
"""
        result = CodeValidator.validate_code(code)
        assert result["is_valid"] is True
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "fetch_data"
        assert result["stats"]["functions"] == 1

    def test_validate_code_strict_mode(self):
        """Test strict mode validation"""
        code = """
def use_eval():
    eval("1 + 1")
"""
        result = CodeValidator.validate_code(code, strict=True)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
