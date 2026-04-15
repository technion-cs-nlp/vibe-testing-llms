"""
Helpers for extracting and normalizing model-generated code.
"""

from __future__ import annotations

import ast
import logging
from typing import List, Optional

from src.vibe_testing.utils import extract_all_code_blocks


class EntryPointRenamer(ast.NodeTransformer):
    """
    AST node transformer that renames a function definition and all its references.
    """

    def __init__(self, old_name: str, new_name: str):
        """
        Initialize the renamer.

        Args:
            old_name (str): The original function name to replace.
            new_name (str): The new function name to use.
        """
        self.old_name = old_name
        self.new_name = new_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Visit a function definition node and rename if it matches.

        Args:
            node (ast.FunctionDef): The function definition node.

        Returns:
            ast.FunctionDef: The node with updated name if it matches.
        """
        self.generic_visit(node)
        if node.name == self.old_name:
            node.name = self.new_name
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        """
        Visit an async function definition node and rename if it matches.

        Args:
            node (ast.AsyncFunctionDef): The async function definition node.

        Returns:
            ast.AsyncFunctionDef: The node with updated name if it matches.
        """
        self.generic_visit(node)
        if node.name == self.old_name:
            node.name = self.new_name
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """
        Visit a name node and update references to the old function name.

        Args:
            node (ast.Name): The name node.

        Returns:
            ast.Name: The node with updated identifier if it matches.
        """
        if node.id == self.old_name:
            node.id = self.new_name
        return node


class CodeSanitizer:
    """
    Sanitizes raw model outputs so they can be executed reliably.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the sanitizer.

        Args:
            logger (Optional[logging.Logger]): Optional logger for debug output.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def sanitize(self, raw_output: str, required_entry_point: str) -> List[str]:
        """
        Extract code blocks, normalize whitespace, and rename the entry function.

        Args:
            raw_output (str): Raw text returned by the model, possibly containing
                markdown fences or prose.
            required_entry_point (str): The function name required by the dataset.

        Returns:
            List[str]: List of cleaned code strings whose primary function matches the entry point.
        """
        all_code_candidates = extract_all_code_blocks(raw_output, required_entry_point)

        # Filter out blocks with syntax errors and strip whitespace
        valid_code_blocks = []
        for block in all_code_candidates:
            stripped_block = block.strip()
            if not stripped_block:
                continue
            try:
                ast.parse(stripped_block)
                renamed = self._rename_entry_point_ast(stripped_block, required_entry_point)
                # Ensure the renamed variant is still valid python.
                ast.parse(renamed)
                valid_code_blocks.append(renamed.strip())
            except SyntaxError:
                self.logger.debug(
                    "Dropping code block with syntax error for entry point %s.",
                    required_entry_point,
                )
                continue

        # If no valid code blocks found, return the raw output as a single-item list
        if not valid_code_blocks:
            return [raw_output.strip()] if raw_output.strip() else []
        return valid_code_blocks

    def _rename_entry_point_ast(self, code: str, entry_point: str) -> str:
        """
        Rename a function definition and all its references using AST parsing.

        This method is more robust than regex-based approaches because it:
        - Correctly identifies the target function (last top-level function)
        - Updates both the definition and all references (including recursive calls)
        - Preserves code structure, decorators, and comments

        Args:
            code (str): The extracted code.
            entry_point (str): The function name mandated by the benchmark.

        Returns:
            str: Code with the renamed function definition and all references updated.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            self.logger.debug(
                "SyntaxError while parsing code, skipping AST rename for entry point %s.",
                entry_point,
            )
            return code

        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entry_point
            ):
                self.logger.debug(
                    "Entry point %s already present, no renaming needed.",
                    entry_point,
                )
                return code

        func_nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not func_nodes:
            self.logger.debug(
                "No function definitions found while attempting to enforce entry point %s.",
                entry_point,
            )
            return code

        target = func_nodes[-1]
        old_name = target.name
        self.logger.debug("Renaming entry point from %s to %s.", old_name, entry_point)

        renamer = EntryPointRenamer(old_name, entry_point)
        new_tree = renamer.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            new_code = ast.unparse(new_tree)
        except AttributeError:
            self.logger.warning(
                "ast.unparse not available (requires Python 3.9+), returning original code."
            )
            return code

        return new_code
