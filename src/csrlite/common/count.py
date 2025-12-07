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
    variable: str | list[str],
    total: bool = True,
    missing_group: str = "error",
    pct_digit: int = 1,
    max_n_width: int | None = None,
) -> pl.DataFrame:
    """
    Counts subjects and observations by group and a specified variable(s),
    calculating percentages based on population denominators.

    Args:
        population (pl.DataFrame): DataFrame containing subject population data,
                                   must include 'id' and 'group' columns.
        observation (pl.DataFrame): DataFrame containing observation data,
                                    must include 'id' and 'variable' columns.
        id (str): The name of the subject ID column (e.g., "USUBJID").
        group (str): The name of the treatment group column (e.g., "TRT01A").
        variable (str | list[str]): The name(s) of the variable(s) to count observations for.
                                    If a list is provided, counts are generated hierarchically.
        total (bool, optional): If True, adds a 'Total' column. Defaults to True.
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

    # Normalize variable to list
    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = variable

    # prepare data
    pop = _to_pop(
        population=population,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    # Select all required columns (id + all variables)
    # Cast variables to String and fill nulls with "__missing__" for explicit missing values
    obs = observation.select(id, *variables).join(pop, on=id, how="left")

    for var in variables:
        obs = obs.with_columns(pl.col(var).cast(pl.String).fill_null("__missing__"))

    if not obs[id].is_in(pop[id].to_list()).all():
        # Get IDs that are in obs but not in pop
        missing_ids = (
            obs.filter(~pl.col(id).is_in(pop[id].to_list()))
            .select(id)
            .unique()
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"Some '{id}' values in the observation DataFrame are not present in the population "
            f"DataFrame: {missing_ids}"
        )

    df_pop = count_subject(
        population=population,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    all_levels_df = []

    # Iterate through levels of hierarchy
    # Level 1: variables[0]
    # Level 2: variables[0], variables[1]
    # ...
    for i in range(1, len(variables) + 1):
        current_vars = variables[:i]

        # Count observations and subjects by group and current variables
        df_obs_counts = obs.group_by(group, *current_vars).agg(
            pl.len().alias("n_obs"), pl.n_unique(id).alias("n_subj")
        )

        # Create all combinations of groups and variables to ensure no missing groups
        unique_groups = df_pop.select(group)
        unique_variables = obs.select(current_vars).unique()

        # Cross join to get all combinations
        all_combinations = unique_groups.join(unique_variables, how="cross")

        # Left join to preserve all combinations, filling missing counts with 0
        df_level = (
            all_combinations.join(df_obs_counts, on=[group, *current_vars], how="left")
            .join(df_pop, on=group, how="left")
            .with_columns([pl.col("n_obs").fill_null(0), pl.col("n_subj").fill_null(0)])
        )

        # Cast current variables to String (already done but safe to repeat or ensure)
        df_level = df_level.with_columns([pl.col(c).cast(pl.String) for c in current_vars])

        # Add missing columns for this level (if any) with "__all__" to match the full structure
        # This is important for stacking later
        for var in variables:
            if var not in df_level.columns:
                df_level = df_level.with_columns(pl.lit("__all__").cast(pl.String).alias(var))

        all_levels_df.append(df_level)

    # Stack all levels
    df_obs = pl.concat(all_levels_df, how="diagonal")

    # Format
    df_obs = df_obs.with_columns(
        pct_subj=(pl.col("n_subj") / pl.col("n_subj_pop") * 100)
    ).with_columns(
        pct_subj_fmt=(
            pl.when(pl.col("pct_subj").is_null() | pl.col("pct_subj").is_nan())
            .then(0.0)
            .otherwise(pl.col("pct_subj"))
            .round(pct_digit, mode="half_away_from_zero")
            .cast(pl.String)
        )
    )

    # Calculate max widths for proper alignment
    if max_n_width is None:
        max_n_width = df_obs.select(pl.col("n_subj").cast(pl.String).str.len_chars().max()).item()

    # Infer max percentage width from pct_digit
    max_pct_width = 3 if pct_digit == 0 else 4 + pct_digit

    # Format with padding for alignment
    df_obs = df_obs.with_columns(
        [
            pl.col("pct_subj_fmt").str.pad_start(max_pct_width, " "),
            pl.col("n_subj").cast(pl.String).str.pad_start(max_n_width, " ").alias("n_subj_fmt"),
        ]
    ).with_columns(
        n_pct_subj_fmt=pl.concat_str(
            [pl.col("n_subj_fmt"), pl.lit(" ("), pl.col("pct_subj_fmt"), pl.lit(")")]
        )
    )

    # Sort
    # 1. Group
    # 2. Variables (hierarchical)
    # Custom sort order: __all__ -> Values -> __missing__

    sort_exprs = [pl.col(group)]
    for var in variables:
        # Create a sort key: 0 for __all__, 1 for values, 2 for __missing__
        sort_key_col = f"__sort_key_{var}__"
        df_obs = df_obs.with_columns(
            pl.when(pl.col(var) == "__all__")
            .then(0)
            .when(pl.col(var) == "__missing__")
            .then(2)
            .otherwise(1)
            .alias(sort_key_col)
        )
        sort_exprs.append(pl.col(sort_key_col))
        sort_exprs.append(pl.col(var))

    df_obs = df_obs.sort(sort_exprs).select(pl.exclude(r"^__sort_key_.*$"))

    # Add __variable__ column with indentation
    if len(variables) > 0:
        # Handle first level
        var_expr = (
            pl.when(pl.col(variables[0]) == "__missing__")
            .then(pl.lit("Missing"))
            .otherwise(pl.col(variables[0]))
        )

        for i in range(1, len(variables)):
            var_expr = (
                pl.when(pl.col(variables[i]) == "__all__")
                .then(var_expr)
                .when(pl.col(variables[i]) == "__missing__")
                .then(pl.lit(" " * 4 * i) + pl.lit("Missing"))
                .otherwise(pl.lit(" " * 4 * i) + pl.col(variables[i]))
            )
        df_obs = df_obs.with_columns(var_expr.alias("__variable__"))

    # Add __id__ column
    df_obs = df_obs.with_row_index(name="__id__", offset=1)

    return df_obs
