from datetime import date, datetime
from typing import Any

from rag_project.knowledge_base import MetadataField, MetadataSchema, MetadataValidationError
from rag_project.knowledge_base.metadata import validate_typed_value


ALLOWED_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains"}
COMPARISON_OPERATORS = {
    "$eq": "==",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
ORDERABLE_TYPES = {"int", "float", "date", "datetime"}


class MilvusFilterBuilder:
    def __init__(self, *, max_depth: int = 3, max_conditions: int = 20):
        self.max_depth = max_depth
        self.max_conditions = max_conditions
        self._condition_count = 0

    def build(self, *, kb_id: str, metadata_schema: MetadataSchema, filters: dict[str, Any] | None) -> str:
        self._condition_count = 0
        base_expr = f'kb_id == "{_escape_string(kb_id)}"'
        if not filters:
            return base_expr
        if not isinstance(filters, dict):
            raise MetadataValidationError("filters must be a JSON object")
        filter_expr = self._build_node(metadata_schema, filters, depth=1)
        return f"{base_expr} and ({filter_expr})" if filter_expr else base_expr

    def _build_node(self, metadata_schema: MetadataSchema, node: dict[str, Any], *, depth: int) -> str:
        if depth > self.max_depth:
            raise MetadataValidationError("metadata filter exceeds maximum depth")

        parts: list[str] = []
        for key, value in node.items():
            if key in {"$and", "$or"}:
                parts.append(self._build_logical(metadata_schema, key, value, depth=depth))
                continue
            field = metadata_schema.validate_filterable_field(key)
            parts.append(self._build_field_condition(field, value))
        return " and ".join(f"({part})" for part in parts if part)

    def _build_logical(self, metadata_schema: MetadataSchema, operator: str, value: Any, *, depth: int) -> str:
        if not isinstance(value, list) or not value:
            raise MetadataValidationError(f"{operator} must be a non-empty list")
        expressions = []
        for item in value:
            if not isinstance(item, dict):
                raise MetadataValidationError(f"{operator} items must be JSON objects")
            expressions.append(self._build_node(metadata_schema, item, depth=depth + 1))
        joiner = " and " if operator == "$and" else " or "
        return joiner.join(f"({expr})" for expr in expressions if expr)

    def _build_field_condition(self, field: MetadataField, value: Any) -> str:
        if not isinstance(value, dict):
            value = {"$eq": value}
        if not value:
            raise MetadataValidationError(f"filter condition for {field.name} cannot be empty")

        parts: list[str] = []
        for operator, operand in value.items():
            if operator not in ALLOWED_OPERATORS:
                raise MetadataValidationError(f"unsupported filter operator for {field.name}: {operator}")
            self._condition_count += 1
            if self._condition_count > self.max_conditions:
                raise MetadataValidationError("metadata filter exceeds maximum condition count")
            parts.append(self._operator_expr(field, operator, operand))
        return " and ".join(parts)

    def _operator_expr(self, field: MetadataField, operator: str, operand: Any) -> str:
        if operator in COMPARISON_OPERATORS:
            if operator not in {"$eq", "$ne"} and field.type not in ORDERABLE_TYPES:
                raise MetadataValidationError(f"operator {operator} is not supported for {field.type}")
            validate_typed_value(field, operand)
            return f"{field.name} {COMPARISON_OPERATORS[operator]} {_format_value(operand)}"

        if operator in {"$in", "$nin"}:
            if field.type == "string_array":
                raise MetadataValidationError(f"operator {operator} is not supported for string_array")
            if not isinstance(operand, list) or not operand:
                raise MetadataValidationError(f"operator {operator} for {field.name} requires a non-empty list")
            for item in operand:
                validate_typed_value(field, item)
            milvus_op = "in" if operator == "$in" else "not in"
            return f"{field.name} {milvus_op} [{', '.join(_format_value(item) for item in operand)}]"

        if operator == "$contains":
            if field.type == "string_array":
                item_field = field.model_copy(update={"type": "string"})
                validate_typed_value(item_field, operand)
                return f"array_contains({field.name}, {_format_value(operand)})"
            if field.type == "string":
                validate_typed_value(field, operand)
                return f'{field.name} like "%{_escape_like(str(operand))}%"'
            raise MetadataValidationError(f"operator $contains is not supported for {field.type}")

        raise MetadataValidationError(f"unsupported filter operator for {field.name}: {operator}")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) or isinstance(value, float):
        return str(value)
    if isinstance(value, datetime):
        return f'"{_escape_string(value.isoformat())}"'
    if isinstance(value, date):
        return f'"{_escape_string(value.isoformat())}"'
    return f'"{_escape_string(str(value))}"'


def _escape_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_like(value: str) -> str:
    return _escape_string(value).replace("%", "\\%").replace("_", "\\_")
