# Service

Сценарий приложения: входной JSON прибора → карточка.

`get_description_service` принимает `DeviceStatsModel` и вызывает `src.graph.build_card`. Роутеры ходят только сюда, не в граф и не в клиент модели.
