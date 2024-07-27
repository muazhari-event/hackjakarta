import uuid

import numpy as np
from pydantic.v1 import BaseModel as PydanticBaseModel, ConfigDict, Field
from pydantic.v1 import PrivateAttr


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_all=True,
        validate_assignment=True,
    )


class OptimizationVariable(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
    client_id: Optional[str] = PrivateAttr(default=None)


class OptimizationValueFunction(BaseModel):
    name: str
    string: str
    understandability: int
    complexity: int
    maintainability: int
    overall_maintainability: int


class OptimizationValue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    data: Optional[Any | OptimizationValueFunction]

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
        converted_data = self.convert_type(data["data"])
        if data["type"] == "function":
            data["data"] = OptimizationValueFunction(**converted_data)
        else:
            data["type"] = type(converted_data).__name__
        super().__init__(**data)


class OptimizationBinary(OptimizationVariable):
    pass


class OptimizationChoice(OptimizationVariable):
    options: Dict[str, Any | OptimizationValue]


class OptimizationReal(OptimizationVariable):
    bounds: Tuple[float, float]


class OptimizationInteger(OptimizationVariable):
    bounds: Tuple[int, int]


class OptimizationObjective(BaseModel):
    type: str


class OptimizationClient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    host: str
    port: int


class OptimizationPrepareRequest(BaseModel):
    variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal]
    host: str
    name: str
    port: int

    def __init__(self, **data):
        transformed_variables: Dict[str, OptimizationVariable] = {}
        for variable_id, variable in data["variables"].items():
            if variable["type"] == "binary":
                transformed_variables[variable_id] = OptimizationBinary(**variable)
            elif variable["type"] == "choice":
                transformed_variables[variable_id] = OptimizationChoice(**variable)
            elif variable["type"] == "integer":
                transformed_variables[variable_id] = OptimizationInteger(**variable)
            elif variable["type"] == "real":
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
    client: Optional[OptimizationClient] = PrivateAttr(default=None)


class OptimizationInterpretation(BaseModel):
    objectives: List[List[float]]
    solutions: List[Dict[str, OptimizationValue]]
    decision_index: int
    plots: List[Plot]
