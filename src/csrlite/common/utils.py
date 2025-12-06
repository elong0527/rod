# pyre-strict
import polars as pl


def apply_common_filters(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    population_filter: str | None,
    observation_filter: str | None,
    parameter_filter: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply standard population, observation, and parameter filters.

    Returns:
        Tuple of (filtered_population, filtered_observation_pre_id_match)
    """
    # Apply population filter
    if population_filter:
        population_filtered = population.filter(pl.sql_expr(population_filter))
    else:
        population_filtered = population

    # Apply observation filter
    observation_filtered = observation
    if observation_filter:
        observation_filtered = observation_filtered.filter(pl.sql_expr(observation_filter))

    # Apply parameter filter
    if parameter_filter:
        observation_filtered = observation_filtered.filter(pl.sql_expr(parameter_filter))

    return population_filtered, observation_filtered
