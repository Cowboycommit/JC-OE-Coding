"""Code introspection utilities for automatic documentation generation."""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    """Information about a discovered function."""
    name: str
    module: str
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False


@dataclass
class ClassInfo:
    """Information about a discovered class."""
    name: str
    module: str
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    attributes: List[Dict[str, str]] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Information about a discovered module."""
    name: str
    file_path: str
    docstring: Optional[str] = None
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


@dataclass
class DependencyInfo:
    """Information about a project dependency."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    category: str = "runtime"  # runtime, dev, optional


class CodeIntrospector:
    """Introspect Python code to extract documentation metadata."""

    # Known library descriptions for common packages
    KNOWN_LIBRARIES = {
        'pandas': 'Data manipulation and analysis library',
        'numpy': 'Numerical computing library',
        'scikit-learn': 'Machine learning library',
        'sklearn': 'Machine learning library',
        'matplotlib': 'Data visualization library',
        'plotly': 'Interactive visualization library',
        'streamlit': 'Web application framework',
        'flask': 'Lightweight web framework',
        'django': 'Full-featured web framework',
        'fastapi': 'Modern API framework',
        'requests': 'HTTP client library',
        'pytest': 'Testing framework',
        'nltk': 'Natural language processing toolkit',
        'spacy': 'Industrial-strength NLP library',
        'tensorflow': 'Machine learning framework',
        'torch': 'Deep learning framework',
        'pytorch': 'Deep learning framework',
        'keras': 'High-level neural network API',
        'scipy': 'Scientific computing library',
        'python-docx': 'Word document creation library',
        'openpyxl': 'Excel file handling library',
        'pyyaml': 'YAML parsing library',
        'yaml': 'YAML parsing library',
        'json': 'JSON handling (standard library)',
        'os': 'Operating system interface (standard library)',
        'sys': 'System-specific parameters (standard library)',
        're': 'Regular expressions (standard library)',
        'pathlib': 'Object-oriented paths (standard library)',
        'typing': 'Type hints (standard library)',
        'dataclasses': 'Data classes (standard library)',
        'collections': 'Container datatypes (standard library)',
        'itertools': 'Iterator functions (standard library)',
        'functools': 'Higher-order functions (standard library)',
    }

    def __init__(self, project_root: str):
        """Initialize introspector with project root path.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)

    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name.

        Args:
            file_path: Path to Python file

        Returns:
            Dotted module name
        """
        try:
            rel_path = file_path.relative_to(self.project_root)
            parts = list(rel_path.parts)
            if parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3]
            if parts[-1] == '__init__':
                parts = parts[:-1]
            return '.'.join(parts)
        except ValueError:
            return file_path.stem

    def parse_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Parse a Python file and extract module information.

        Args:
            file_path: Path to Python file

        Returns:
            ModuleInfo or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except (IOError, UnicodeDecodeError):
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        module_name = self.get_module_name(file_path)
        module_docstring = ast.get_docstring(tree)

        classes = []
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = self._parse_class(node, module_name, str(file_path))
                classes.append(class_info)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._parse_function(node, module_name, str(file_path))
                functions.append(func_info)

        return ModuleInfo(
            name=module_name,
            file_path=str(file_path),
            docstring=module_docstring,
            classes=classes,
            functions=functions,
            imports=list(set(imports))
        )

    def _parse_class(self, node: ast.ClassDef, module: str, file_path: str) -> ClassInfo:
        """Parse a class definition node.

        Args:
            node: AST class definition node
            module: Module name
            file_path: Source file path

        Returns:
            ClassInfo instance
        """
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")

        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._parse_function(item, module, file_path)
                methods.append(method_info)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attr = {
                        'name': item.target.id,
                        'type': self._get_annotation_string(item.annotation)
                    }
                    attributes.append(attr)

        return ClassInfo(
            name=node.name,
            module=module,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            bases=bases,
            methods=methods,
            attributes=attributes,
            decorators=decorators
        )

    def _parse_function(self, node, module: str, file_path: str) -> FunctionInfo:
        """Parse a function definition node.

        Args:
            node: AST function definition node
            module: Module name
            file_path: Source file path

        Returns:
            FunctionInfo instance
        """
        parameters = []
        for arg in node.args.args:
            param = {'name': arg.arg}
            if arg.annotation:
                param['type'] = self._get_annotation_string(arg.annotation)
            parameters.append(param)

        return_type = None
        if node.returns:
            return_type = self._get_annotation_string(node.returns)

        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        return FunctionInfo(
            name=node.name,
            module=module,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )

    def _get_annotation_string(self, node) -> str:
        """Convert an annotation AST node to string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_string(node.value)
            slice_val = self._get_annotation_string(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_annotation_string(e) for e in node.elts]
            return ', '.join(elements)
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.BinOp):
            # Handle Union types with | operator
            left = self._get_annotation_string(node.left)
            right = self._get_annotation_string(node.right)
            return f"{left} | {right}"
        return "Any"

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name from AST node."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))

    def _get_decorator_name(self, node) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "unknown"


def discover_project(project_root: str, source_dirs: List[str] = None) -> Dict[str, Any]:
    """Discover all Python modules, classes, and functions in a project.

    Args:
        project_root: Path to project root
        source_dirs: List of source directories to scan (relative to root)

    Returns:
        Dictionary with discovered modules, classes, functions
    """
    root = Path(project_root)
    introspector = CodeIntrospector(project_root)

    if source_dirs is None:
        source_dirs = ['.']

    modules = []
    all_classes = []
    all_functions = []

    for source_dir in source_dirs:
        source_path = root / source_dir
        if not source_path.exists():
            continue

        for py_file in source_path.rglob('*.py'):
            # Skip test files and __pycache__
            if '__pycache__' in str(py_file):
                continue
            if py_file.name.startswith('test_') or py_file.name.endswith('_test.py'):
                continue

            module_info = introspector.parse_file(py_file)
            if module_info:
                modules.append(module_info)
                all_classes.extend(module_info.classes)
                all_functions.extend(module_info.functions)

    return {
        'modules': modules,
        'classes': all_classes,
        'functions': all_functions,
    }


def get_classes(project_root: str, source_dirs: List[str] = None) -> List[ClassInfo]:
    """Get all classes discovered in the project.

    Args:
        project_root: Path to project root
        source_dirs: Source directories to scan

    Returns:
        List of ClassInfo objects
    """
    result = discover_project(project_root, source_dirs)
    return result['classes']


def get_functions(project_root: str, source_dirs: List[str] = None) -> List[FunctionInfo]:
    """Get all functions discovered in the project.

    Args:
        project_root: Path to project root
        source_dirs: Source directories to scan

    Returns:
        List of FunctionInfo objects
    """
    result = discover_project(project_root, source_dirs)
    return result['functions']


def get_modules(project_root: str, source_dirs: List[str] = None) -> List[ModuleInfo]:
    """Get all modules discovered in the project.

    Args:
        project_root: Path to project root
        source_dirs: Source directories to scan

    Returns:
        List of ModuleInfo objects
    """
    result = discover_project(project_root, source_dirs)
    return result['modules']


def get_dependencies(project_root: str) -> List[DependencyInfo]:
    """Extract dependencies from requirements files and pyproject.toml.

    Args:
        project_root: Path to project root

    Returns:
        List of DependencyInfo objects
    """
    root = Path(project_root)
    dependencies = []
    seen = set()

    introspector = CodeIntrospector(project_root)

    # Parse requirements.txt
    req_file = root / 'requirements.txt'
    if req_file.exists():
        deps = _parse_requirements_file(req_file)
        for dep in deps:
            if dep.name not in seen:
                dep.description = introspector.KNOWN_LIBRARIES.get(
                    dep.name.lower().replace('-', '_'),
                    introspector.KNOWN_LIBRARIES.get(dep.name.lower(), None)
                )
                dependencies.append(dep)
                seen.add(dep.name)

    # Parse requirements-dev.txt
    req_dev_file = root / 'requirements-dev.txt'
    if req_dev_file.exists():
        deps = _parse_requirements_file(req_dev_file)
        for dep in deps:
            if dep.name not in seen:
                dep.category = 'dev'
                dep.description = introspector.KNOWN_LIBRARIES.get(
                    dep.name.lower().replace('-', '_'),
                    introspector.KNOWN_LIBRARIES.get(dep.name.lower(), None)
                )
                dependencies.append(dep)
                seen.add(dep.name)

    # Parse pyproject.toml if exists
    pyproject = root / 'pyproject.toml'
    if pyproject.exists():
        deps = _parse_pyproject(pyproject)
        for dep in deps:
            if dep.name not in seen:
                dep.description = introspector.KNOWN_LIBRARIES.get(
                    dep.name.lower().replace('-', '_'),
                    introspector.KNOWN_LIBRARIES.get(dep.name.lower(), None)
                )
                dependencies.append(dep)
                seen.add(dep.name)

    return dependencies


def _parse_requirements_file(file_path: Path) -> List[DependencyInfo]:
    """Parse a requirements.txt file.

    Args:
        file_path: Path to requirements file

    Returns:
        List of DependencyInfo objects
    """
    dependencies = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('-'):
                continue

            # Parse package name and version
            match = re.match(r'^([a-zA-Z0-9_-]+)\s*([<>=!]+.+)?$', line)
            if match:
                name = match.group(1)
                version = match.group(2).strip() if match.group(2) else None
                dependencies.append(DependencyInfo(name=name, version=version))

    return dependencies


def _parse_pyproject(file_path: Path) -> List[DependencyInfo]:
    """Parse dependencies from pyproject.toml.

    Args:
        file_path: Path to pyproject.toml

    Returns:
        List of DependencyInfo objects
    """
    dependencies = []

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return dependencies

    try:
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)
    except Exception:
        return dependencies

    # Get dependencies from [project] section
    project = data.get('project', {})
    for dep in project.get('dependencies', []):
        match = re.match(r'^([a-zA-Z0-9_-]+)', dep)
        if match:
            name = match.group(1)
            version = dep[len(name):].strip() if len(dep) > len(name) else None
            dependencies.append(DependencyInfo(name=name, version=version))

    # Get optional dependencies
    for group, deps in project.get('optional-dependencies', {}).items():
        for dep in deps:
            match = re.match(r'^([a-zA-Z0-9_-]+)', dep)
            if match:
                name = match.group(1)
                version = dep[len(name):].strip() if len(dep) > len(name) else None
                dependencies.append(DependencyInfo(
                    name=name,
                    version=version,
                    category='optional'
                ))

    return dependencies
