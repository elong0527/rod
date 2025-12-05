# pyre-strict
import polars as pl


def _to_pop(
    population: pl.DataFrame,
    id: str,
    group: str,
    total: bool = True,
    missing_group: str = "error",
) -> pl.DataFrame:
    # prepare data
    pop = population.select(id, group)

    # validate data
    if pop[id].is_duplicated().any():
        raise ValueError(f"The '{id}' column in the population DataFrame is not unique.")

    if missing_group == "error" and pop[group].is_null().any():
        raise ValueError(
            f"Missing values found in the '{group}' column of the population DataFrame, "
            "and 'missing_group' is set to 'error'."
        )

    # Convert group to Enum for consistent categorical ordering
    u_pop = pop[group].unique().sort().to_list()

    # handle total column
    if total:
        pop_total = pop.with_columns(pl.lit("Total").alias(group))
        pop = pl.concat([pop, pop_total]).with_columns(
            pl.col(group).cast(pl.Enum(u_pop + ["Total"]))
        )
    else:
        pop = pop.with_columns(pl.col(group).cast(pl.Enum(u_pop)))

    return pop


def count_subject(
    population: pl.DataFrame,
    id: str,
    group: str,
    total: bool = True,
    missing_group: str = "error",
) -> pl.DataFrame:
    """
    Counts subjects by group and optionally includes a 'Total' column.

    Args:
        population (pl.DataFrame): DataFrame containing subject population data,
                                   must include 'id' and 'group' columns.
        id (str): The name of the subject ID column (e.g., "USUBJID").
        group (str): The name of the treatment group column (e.g., "TRT01A").
        total (bool, optional): If True, adds a 'Total' group with counts across all groups.
                                Defaults to True.
        missing_group (str, optional): How to handle missing values in the group column.
                                       "error" will raise a ValueError. Defaults to "error".

    Returns:
        pl.DataFrame: A DataFrame with subject counts ('n_subj_pop') for each group.
    """

    pop = _to_pop(
        population=population,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    return pop.group_by(group).agg(pl.len().alias("n_subj_pop")).sort(group)


def count_subject_with_observation(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    id: str,
    group: str,
    variable: str,
    total: bool = True,
    missing_group: str = "error",
    pct_digit: int = 1,
    max_n_width: int | None = None,
) -> pl.DataFrame:
    """
    Counts subjects and observations by group and a specified variable,
    calculating percentages based on population denominators.

    Args:
        population (pl.DataFrame): DataFrame containing subject population data,
                                   must include 'id' and 'group' columns.
        observation (pl.DataFrame): DataFrame containing observation data,
                                    must include 'id' and 'variable' columns.
        id (str): The name of the subject ID column (e.g., "USUBJID").
        group (str): The name of the treatment group column (e.g., "TRT01A").
        variable (str): The name of the variable to count observations for (e.g., "AESOC").
        total (bool, optional): Not yet implemented. Defaults to True.
        missing_group (str, optional): How to handle missing values in the group column.
                                       "error" will raise a ValueError. Defaults to "error".
        pct_digit (int, optional): Number of decimal places for percentage formatting.
                                   Defaults to 1.
        max_n_width (int, optional): Fixed width for subject count formatting. If None, inferred
                                     from data. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with counts and percentages of subjects and observations
                      grouped by 'group' and 'variable'.
    """

    # prepare data
    pop = _to_pop(
        population=population,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    if not observation[id].is_in(pop[id].to_list()).all():
        # Get IDs that are in obs but not in pop
        missing_ids = (
            observation.filter(~pl.col(id).is_in(pop[id].to_list()))
            .select(id)
            .unique()
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"Some '{id}' values in the observation DataFrame are not present in the population "
            f"DataFrame: {missing_ids}"
        )

    obs = observation.select(id, variable).join(pop, on=id, how="inner")

    df_pop = count_subject(
        population=population,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    # Count observations and subjects by group and variable
    df_obs_counts = obs.group_by(group, variable).agg(
        pl.len().alias("n_obs"), pl.n_unique(id).alias("n_subj")
    )

    # Create all combinations of groups and variables to ensure no missing groups
    unique_groups = df_pop.select(group)
    unique_variables = obs.select(variable).unique()

    # Cross join to get all combinations
    all_combinations = unique_groups.join(unique_variables, how="cross")

    # Left join to preserve all combinations, filling missing counts with 0
    df_obs = (
        all_combinations.join(df_obs_counts, on=[group, variable], how="left")
        .join(df_pop, on=group, how="left")
        .with_columns([pl.col("n_obs").fill_null(0), pl.col("n_subj").fill_null(0)])
        .with_columns(pct_subj=(pl.col("n_subj") / pl.col("n_subj_pop") * 100))
        .with_columns(
            pct_subj_fmt=(
                pl.when(pl.col("pct_subj").is_null() | pl.col("pct_subj").is_nan())
                .then(0.0)
                .otherwise(pl.col("pct_subj"))
                .round(pct_digit, mode="half_away_from_zero")
                .cast(pl.String)
            )
        )
    )

    # Calculate max widths for proper alignment
    if max_n_width is None:
        max_n_width = df_obs.select(pl.col("n_subj").cast(pl.String).str.len_chars().max()).item()

    # Infer max percentage width from pct_digit
    max_pct_width = 3 if pct_digit == 0 else 4 + pct_digit

    # Format with padding for alignment
    df_obs = (
        df_obs.with_columns(
            [
                pl.col("pct_subj_fmt").str.pad_start(max_pct_width, " "),
                pl.col("n_subj")
                .cast(pl.String)
                .str.pad_start(max_n_width, " ")
                .alias("n_subj_fmt"),
            ]
        )
        .with_columns(
            n_pct_subj_fmt=pl.concat_str(
                [pl.col("n_subj_fmt"), pl.lit(" ("), pl.col("pct_subj_fmt"), pl.lit(")")]
            )
        )
        .sort(group, variable)
    )

    return df_obs
