"""
Code Validator

Validates and sanitizes generated code for safety and correctness.
"""

import ast
import re
from typing import List, Dict, Any, Tuple, Optional


class CodeValidator:
    """
    Validates generated Python code for safety and correctness.

    This class performs static analysis on generated code to ensure it:
    - Is valid Python syntax
    - Doesn't contain dangerous operations (without explicit approval)
    - Follows basic security best practices
    - Has required function signatures
    """

    # Dangerous operations that should be flagged
    DANGEROUS_OPERATIONS = {
        "eval",
        "exec",
        "__import__",
        "compile",
        "open",  # File operations should be explicit
        "os.system",
        "subprocess",
        "shutil.rmtree",
    }

    # Allowed imports (whitelist)
    ALLOWED_IMPORTS = {
        "requests",
        "httpx",
        "playwright",
        "asyncio",
        "playwright.async_api",
        "playwright.sync_api",
        "bs4",
        "lxml",
        "json",
        "typing",
        "dataclasses",
        "datetime",
        "time",
        "re",
        "urllib",
        "logging",
    }

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that code is syntactically correct Python.

        Args:
            code: Python code as string

        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"

    @classmethod
    def check_dangerous_operations(cls, code: str) -> List[str]:
        """
        Check for potentially dangerous operations in code.

        Args:
            code: Python code as string

        Returns:
            List of warnings about dangerous operations
        """
        warnings = []

        try:
            tree = ast.parse(code)
        except:
            return ["Could not parse code for safety analysis"]

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.DANGEROUS_OPERATIONS:
                        warnings.append(
                            f"Potentially dangerous operation: {node.func.id} at line {node.lineno}"
                        )
                elif isinstance(node.func, ast.Attribute):
                    full_name = cls._get_full_attr_name(node.func)
                    if any(danger in full_name for danger in cls.DANGEROUS_OPERATIONS):
                        warnings.append(
                            f"Potentially dangerous operation: {full_name} at line {node.lineno}"
                        )

        return warnings

    @classmethod
    def check_imports(cls, code: str) -> List[str]:
        """
        Check if all imports are from the allowed list.

        Args:
            code: Python code as string

        Returns:
            List of warnings about suspicious imports
        """
        warnings = []

        try:
            tree = ast.parse(code)
        except:
            return ["Could not parse code for import analysis"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not cls._is_import_allowed(alias.name):
                        warnings.append(
                            f"Suspicious import: {alias.name} at line {node.lineno}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and not cls._is_import_allowed(node.module):
                    warnings.append(
                        f"Suspicious import: {node.module} at line {node.lineno}"
                    )

        return warnings

    @staticmethod
    def extract_functions(code: str) -> List[Dict[str, Any]]:
        """
        Extract all function definitions from code.

        Args:
            code: Python code as string

        Returns:
            List of function info dicts
        """
        functions = []

        try:
            tree = ast.parse(code)
        except:
            return []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "has_docstring": ast.get_docstring(node) is not None,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
                functions.append(func_info)

        return functions

    @classmethod
    def validate_code(cls, code: str, strict: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive validation on code.

        Args:
            code: Python code as string
            strict: If True, treat warnings as errors

        Returns:
            Validation result with is_valid, warnings, and errors
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "functions": [],
            "stats": {
                "lines": len(code.splitlines()),
                "functions": 0,
            }
        }

        # Check syntax
        syntax_valid, syntax_error = cls.validate_syntax(code)
        if not syntax_valid:
            result["is_valid"] = False
            result["errors"].append(syntax_error)
            return result

        # Check for dangerous operations
        danger_warnings = cls.check_dangerous_operations(code)
        result["warnings"].extend(danger_warnings)

        # Check imports
        import_warnings = cls.check_imports(code)
        result["warnings"].extend(import_warnings)

        # Extract functions
        functions = cls.extract_functions(code)
        result["functions"] = functions
        result["stats"]["functions"] = len(functions)

        # In strict mode, warnings are errors
        if strict and result["warnings"]:
            result["is_valid"] = False
            result["errors"] = result["warnings"]
            result["warnings"] = []

        return result

    @staticmethod
    def _get_full_attr_name(node: ast.Attribute) -> str:
        """Recursively get full attribute name (e.g., os.system)."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{CodeValidator._get_full_attr_name(node.value)}.{node.attr}"
        return node.attr

    @classmethod
    def _is_import_allowed(cls, module_name: str) -> bool:
        """Check if import is in allowed list."""
        # Check exact match or prefix match (e.g., "requests.auth" matches "requests")
        for allowed in cls.ALLOWED_IMPORTS:
            if module_name == allowed or module_name.startswith(allowed + "."):
                return True
        return False


def validate_workflow_code(code: str, strict: bool = False) -> bool:
    """
    Convenience function to validate workflow code.

    Args:
        code: Python code to validate
        strict: If True, treat warnings as errors

    Returns:
        True if code is valid
    """
    validator = CodeValidator()
    result = validator.validate_code(code, strict=strict)

    if not result["is_valid"]:
        print("❌ Code validation failed:")
        for error in result["errors"]:
            print(f"   - {error}")
        return False

    if result["warnings"]:
        print("⚠️  Code validation warnings:")
        for warning in result["warnings"]:
            print(f"   - {warning}")

    print(f"✓ Code validation passed ({result['stats']['lines']} lines, {result['stats']['functions']} functions)")
    return True
