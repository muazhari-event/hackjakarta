from dependency_injector.containers import DeclarativeContainer
from dependency_injector import providers

class SettingContainer(DeclarativeContainer):
    application = providers.Singleton(
        ApplicationSetting
    )

class GatewayContainer(DeclarativeContainer):
    settings = providers.DependenciesContainer()

    evaluation = providers.Singleton(
        EvaluationGateway,
        application_setting=settings.application
    )


class DatastoreContainer(DeclarativeContainer):
    one = providers.Singleton(
        OneDatastore
    )


class UseCaseContainer(DeclarativeContainer):
    settings = providers.DependenciesContainer()
    gateways = providers.DependenciesContainer()
    datastores = providers.DependenciesContainer()

    one = providers.Singleton(
        OptimizationUseCase,
        evaluation_gateway=gateways.evaluation,
        one_datastore=datastores.one,
        application_setting=settings.application
    )


class ControllerContainer(DeclarativeContainer):
    use_cases = providers.DependenciesContainer()

    optimization = providers.Singleton(
        OptimizationController,
        optimization_use_case=use_cases.optimization
    )

    health = providers.Singleton(
        HealthController,
    )


class RouterContainer(DeclarativeContainer):
    controllers = providers.DependenciesContainer()

    api = providers.Singleton(
        ApiRouter,
        optimization_controller=controllers.optimization,
        health_controller=controllers.health
    )


class MainContainer(DeclarativeContainer):
    settings = providers.Container(
        SettingContainer
    )
    gateways = providers.Container(
        GatewayContainer,
    )
    datastores = providers.Container(
        DatastoreContainer,
    )
    use_cases = providers.Container(
        UseCaseContainer,
        gateways=gateways,
    )
    controllers = providers.Container(
        ControllerContainer,
        use_cases=use_cases
    )
    routers = providers.Container(
        RouterContainer,
        controllers=controllers
    )

