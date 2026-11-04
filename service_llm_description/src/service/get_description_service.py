from src.graph import build_card
from src.schemas import CardResponse, DeviceStatsModel


async def get_description_service(device: DeviceStatsModel) -> CardResponse:
    """
    Build a product-card draft for the given instrument.

    Parameters
    ----------
    device : DeviceStatsModel
        Parsed input JSON: name, manufacturer, specs, description.

    Returns
    -------
    card : CardResponse
        SEO fields, key specs, markdown, and optional web sources.

    See Also
    --------
    * :func:`src.graph.build_card`

    """
    return await build_card(device)
