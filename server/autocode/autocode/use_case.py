import asyncio
from typing import Dict, List, Callable, Any, Coroutine

from pymoo.core.plot import Plot
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.core.variable import Variable, Binary, Choice, Integer, Real

from server.autocode.autocode.datastore import OneDatastore
from server.autocode.autocode.gateway import EvaluationGateway
from server.autocode.autocode.model import OptimizationPrepareRequest, OptimizationChoice, OptimizationBinary, \
    OptimizationInteger, OptimizationReal, OptimizationVariable, OptimizationClient, OptimizationEvaluateRunRequest, \
    OptimizationValue, OptimizationEvaluatePrepareRequest, OptimizationEvaluateRunResponse, OptimizationObjective, \
    OptimizationInterpretation
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
            evaluator: Callable[[List[OptimizationEvaluateRunResponse]], Dict[str, Any]],
            queue: Queue
    ):
        self.application_setting = application_setting
        self.optimization_gateway = optimization_gateway
        self.variables = variables
        self.clients = clients
        self.evaluator = evaluator
        self.queue = queue
        self.num_objectives = num_objectives
        self.num_inequality_constraints = num_inequality_constraints
        self.num_equality_constraints = num_equality_constraints
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

    def _evaluate(self, X, out, *args, **kwargs):
        worker_id: str = self.queue.get()

        client_to_variable_values: Dict[str, Dict[str, OptimizationValue]] = {}
        for variable_id, variable in self.variables.items():
            value: Any = X[variable_id]
            if type(value) is not OptimizationValue:
                value: OptimizationValue = OptimizationValue(
                    data=value
                )
            else:
                value: OptimizationValue = value

            if client_to_variable_values.get(variable.client_id) is None:
                client_to_variable_values[variable.client_id] = {}
            client_to_variable_values[variable.client_id][variable_id] = value

        prepare_futures: List[Coroutine] = []
        for client_id, variable_values in client_to_variable_values.items():
            prepare_request: OptimizationEvaluatePrepareRequest = OptimizationEvaluatePrepareRequest(
                worker_id=worker_id,
                variable_values=variable_values
            )
            client: OptimizationClient = self.clients[client_id]
            prepare_response = self.optimization_gateway.evaluate_prepare(
                client=client,
                request=prepare_request
            )
            prepare_futures.append(prepare_response)

        asyncio.get_event_loop().run_until_complete(asyncio.gather(*prepare_futures))

        run_futures: List[Coroutine] = []
        for client in self.clients.values():
            run_request: OptimizationEvaluateRunRequest = OptimizationEvaluateRunRequest(
                worker_id=worker_id
            )
            run_response = self.optimization_gateway.evaluate_run(
                client=client,
                request=run_request
            )
            run_futures.append(run_response)

        self.queue.put(worker_id)

        results: List[OptimizationEvaluateRunResponse] = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*run_futures)
        )

        for result, client in zip(results, self.clients.values()):
            result.client = client

        evaluator_output: Dict[str, Any] = self.evaluator(results)
        evaluator_output.setdefault("F", [])
        evaluator_output.setdefault("G", [])
        evaluator_output.setdefault("H", [])

        if len(evaluator_output["F"]) != self.num_objectives:
            raise ValueError(f"Number of objectives {len(evaluator_output['F'])} does not match {self.num_objectives}.")
        if len(evaluator_output["G"]) != self.num_inequality_constraints:
            raise ValueError(
                f"Number of inequality constraints {len(evaluator_output['G'])} does not match {self.num_inequality_constraints}.")
        if len(evaluator_output["H"]) != self.num_equality_constraints:
            raise ValueError(
                f"Number of equality constraints {len(evaluator_output['H'])} does not match {self.num_equality_constraints}.")

        out.update(evaluator_output)


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
