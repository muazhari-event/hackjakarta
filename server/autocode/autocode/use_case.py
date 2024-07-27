from server.autocode.autocode.datastore import OneDatastore
from server.autocode.autocode.gateway import EvaluationGateway
from server.autocode.autocode.model import OptimizationPrepareRequest
from server.autocode.autocode.setting import ApplicationSetting


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
