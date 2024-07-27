from server.autocode.autocode.model import OptimizationEvaluatePrepareRequest, OptimizationClient


class EvaluationGateway:
    def __init__(self):
        pass

    async def evaluate_prepare(self, client: OptimizationClient, request: OptimizationEvaluatePrepareRequest):
        pass

    async def evaluate_run(self, client: OptimizationClient, request: OptimizationEvaluatePrepareRequest):
        pass
