import hashlib
import uuid
from typing import Optional, Any, List, Dict, Tuple

import dill
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field
from pydantic import PrivateAttr
from pymoo.core.plot import Plot
from sqlmodel import SQLModel, Field as SQLField


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        revalidate_instances="always",
        validate_default=True,
        validate_return=True,
        validate_assignment=True,
    )


class OptimizationVariable(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
    _client_id: Optional[str] = PrivateAttr(default=None)

    def get_client_id(self):
        return self._client_id

    def set_client_id(self, client_id: str):
        self._client_id = client_id


class OptimizationValueFunction(BaseModel):
    name: str
    string: str
    understandability: Optional[int] = Field(default=None)
    complexity: Optional[int] = Field(default=None)
    maintainability: Optional[int] = Field(default=None)
    overall_maintainability: Optional[int] = Field(default=None)


class OptimizationValue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Optional[str]
    data: Any | OptimizationValueFunction

    @staticmethod
    def convert_type(data: Any):
        data_type = type(data)
        if data_type == np.ndarray:
            return data.tolist()
        elif data_type == np.float64:
            return float(data)
        elif data_type == np.int64:
            return int(data)
        elif data_type == np.bool_:
            return bool(data)
        else:
            return data

    def __init__(self, **data):
        data["data"] = self.convert_type(data["data"])
        if data["type"] == "function":
            if type(data["data"]) != OptimizationValueFunction:
                data["data"] = OptimizationValueFunction(**data["data"])
        else:
            data["type"] = type(data["data"]).__name__
        super().__init__(**data)


class OptimizationBinary(OptimizationVariable):
    pass


class OptimizationChoice(OptimizationVariable):
    options: Dict[str, OptimizationValue]


class OptimizationReal(OptimizationVariable):
    bounds: Tuple[float, float]


class OptimizationInteger(OptimizationVariable):
    bounds: Tuple[int, int]


class OptimizationObjective(BaseModel):
    type: str


class OptimizationClient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    name: str
    host: str
    port: int


class OptimizationPrepareRequest(BaseModel):
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    host: str
    port: int
    name: str

    def __init__(self, **data):
        transformed_variables: Dict[str, OptimizationVariable] = {}
        for variable_id, variable in data["variables"].items():
            if variable["type"] == OptimizationBinary.__name__:
                transformed_variables[variable_id] = OptimizationBinary(**variable)
            elif variable["type"] == OptimizationChoice.__name__:
                transformed_variables[variable_id] = OptimizationChoice(**variable)
            elif variable["type"] == OptimizationInteger.__name__:
                transformed_variables[variable_id] = OptimizationInteger(**variable)
            elif variable["type"] == OptimizationReal.__name__:
                transformed_variables[variable_id] = OptimizationReal(**variable)
            else:
                raise ValueError(f"Variable type {variable['type']} is not supported.")
        data["variables"] = transformed_variables
        super().__init__(**data)


class OptimizationPrepareResponse(BaseModel):
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    num_workers: int


class OptimizationEvaluatePrepareRequest(BaseModel):
    worker_id: str
    variable_values: Dict[str, OptimizationValue]


class OptimizationEvaluateRunRequest(BaseModel):
    worker_id: str


class OptimizationEvaluateRunResponse(BaseModel):
    objectives: List[float]
    inequality_constraints: List[float]
    equality_constraints: List[float]
    _client: Optional[OptimizationClient] = PrivateAttr(default=None)

    def set_client(self, client: OptimizationClient):
        self._client = client

    def get_client(self) -> OptimizationClient:
        return self._client


class OptimizationInterpretation(BaseModel):
    objectives: List[List[float]]
    solutions: List[Dict[str, OptimizationValue]]
    decision_index: int
    plots: List[Plot]


class Cache(SQLModel, table=True):
    key: str = SQLField(primary_key=True)
    value: bytes

    def __hash__(self):
        return int.from_bytes(
            bytes=hashlib.sha256(dill.dumps(self)).digest(),
            byteorder="big"
        )
