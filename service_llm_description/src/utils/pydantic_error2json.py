def _pydantic_errors2json(errors: list) -> list[dict]:
    """
    Make pydantic error dicts JSON-serializable.

    Parameters
    ----------
    errors : list
        Output of ``RequestValidationError.errors()``.

    Returns
    -------
    payload : list of dict
        Same structure with non-JSON ``ctx`` values stringified.

    """
    payload: list[dict] = []
    for item in errors:
        row = dict(item)
        ctx = row.get("ctx")
        if isinstance(ctx, dict):
            row["ctx"] = {
                key: value
                if isinstance(value, (str, int, float, bool, type(None)))
                else str(value)
                for key, value in ctx.items()
            }
        payload.append(row)
    return payload
