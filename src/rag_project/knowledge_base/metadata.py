from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MetadataFieldType = Literal["string", "int", "float", "bool", "date", "datetime", "string_array"]


class MetadataValidationError(ValueError):
    """Raised when document metadata or filters violate a knowledge base schema."""


class MetadataField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: MetadataFieldType
    required: bool = False
    filterable: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("metadata field name cannot be empty")
        if not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError("metadata field name must be alphanumeric/underscore and cannot start with a digit")
        return value


class MetadataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[MetadataField] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "MetadataSchema":
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metadata fields: {', '.join(duplicates)}")
        return self

    @property
    def field_map(self) -> dict[str, MetadataField]:
        return {field.name: field for field in self.fields}

    def get_field(self, name: str) -> MetadataField | None:
        return self.field_map.get(name)

    def validate_document_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            raise MetadataValidationError("metadata must be a JSON object")

        field_map = self.field_map
        unknown = sorted(set(metadata) - set(field_map))
        if unknown:
            raise MetadataValidationError(f"unknown metadata fields: {', '.join(unknown)}")

        missing = sorted(field.name for field in self.fields if field.required and field.name not in metadata)
        if missing:
            raise MetadataValidationError(f"missing required metadata fields: {', '.join(missing)}")

        for name, value in metadata.items():
            self.validate_value(name, value)
        return metadata

    def validate_filterable_field(self, name: str) -> MetadataField:
        field = self.get_field(name)
        if field is None:
            raise MetadataValidationError(f"filter field is not declared in metadata schema: {name}")
        if not field.filterable:
            raise MetadataValidationError(f"metadata field is not filterable: {name}")
        return field

    def validate_value(self, name: str, value: Any) -> None:
        field = self.get_field(name)
        if field is None:
            raise MetadataValidationError(f"metadata field is not declared in schema: {name}")
        validate_typed_value(field, value)


def validate_typed_value(field: MetadataField, value: Any) -> None:
    if value is None:
        if field.required:
            raise MetadataValidationError(f"metadata field is required: {field.name}")
        return

    type_name = field.type
    valid = {
        "string": _is_string,
        "int": _is_int,
        "float": _is_float,
        "bool": _is_bool,
        "date": _is_date,
        "datetime": _is_datetime,
        "string_array": _is_string_array,
    }[type_name](value)

    if not valid:
        raise MetadataValidationError(f"metadata field {field.name} must be {type_name}")


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float(value: Any) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_date(value: Any) -> bool:
    if isinstance(value, date) and not isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_datetime(value: Any) -> bool:
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _is_string_array(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
