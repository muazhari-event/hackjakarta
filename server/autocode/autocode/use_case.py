from typing import Dict, List, Callable, Any

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Variable, Binary, Choice, Integer, Real

from server.autocode.autocode.datastore import OneDatastore
from server.autocode.autocode.gateway import EvaluationGateway
from server.autocode.autocode.model import OptimizationPrepareRequest, OptimizationChoice, OptimizationBinary, \
    OptimizationInteger, OptimizationReal, OptimizationVariable, OptimizationClient, OptimizationEvaluateRunRequest
from server.autocode.autocode.setting import ApplicationSetting

from ray.util.queue import Queue


class OptimizationProblem(ElementwiseProblem):

    def __init__(
            self,
            application_setting: ApplicationSetting,
            optimization_gateway: EvaluationGateway,
            num_objectives: int,
            num_inequality_constraints: int,
            num_equality_constraints: int,
            variables: Dict[str, OptimizationBinary | OptimizationChoice | OptimizationInteger | OptimizationReal],
            clients: Dict[str, OptimizationClient],
            evaluator: Callable[[List[OptimizationEvaluateRunRequest]], Dict[str, Any]],
            queue: Queue
    ):
        self.application_setting = application_setting
        self.optimization_gateway = optimization_gateway
        self.variables = variables
        self.clients = clients
        self.evaluator = evaluator
        self.queue = queue
        self.vars: Dict[str, Variable] = {}
        for variable_id, variable in variables.items():
            variable_type = type(variable)
            if variable_type == OptimizationBinary:
                self.vars[variable_id] = Binary()
            elif variable_type == OptimizationChoice:
                self.vars[variable_id] = Choice(options=list(variable.options.values()))
            elif variable_type == OptimizationInteger:
                self.vars[variable_id] = Integer(bounds=variable.bounds)
            elif variable_type == OptimizationReal:
                self.vars[variable_id] = Real(bounds=variable.bounds)
            else:
                raise ValueError(f"Variable type '{variable_type}' is not supported.")

        super().__init__(
            n_var=len(self.vars),
            n_obj=num_objectives,
            n_ieq_constr=num_inequality_constraints,
            n_eq_constr=num_equality_constraints
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pass


class OptimizationUseCase:
    def __init__(
            self,
            optimization_gateway: EvaluationGateway,
            one_datastore: OneDatastore,
            application_setting: ApplicationSetting
    ):
        self.optimization_gateway = optimization_gateway
        self.one_datastore = one_datastore
        self.application_setting = application_setting

    def prepare(self, request: OptimizationPrepareRequest):
        pass

    def reset(self):
        pass

    def run(
            self,
            objectives: List[OptimizationObjective],
            evaluator: Callable[[List[OptimizationEvaluateRunRequest]], Dict[str, Any]],
    ):
        pass

    def plot(self, result: Result, decision_index: int) -> List[Plot]:
        pass

    def interpret(
            self,
            objectives: List[OptimizationObjective],
            result: Result,
            weights: List[float]
    ) -> OptimizationInterpretation:
        pass

    def minimize(
            self,
            objectives: List[OptimizationObjective],
            variables: Dict[str, OptimizationVariable],
            evaluator: Callable[[List[OptimizationEvaluateRunRequest]], Dict[str, Any]],
            clients: Dict[str, OptimizationClient],
    ) -> Result:
        pass

    def get_decision_index(self, result: Result, weights: List[float]) -> int:
        pass
