from server.autocode.autocode.controller import OptimizationController, HealthController


class ApiRouter:
    def __init__(
            self,
            optimization_controller: OptimizationController,
            health_controller: HealthController,
    ):
        self.optimization_controller = optimization_controller
        self.health_controller = health_controller

