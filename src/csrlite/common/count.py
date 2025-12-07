# pyre-strict
import polars as pl

from .config import config


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
        population (pl.DataFrame): DataFrame containing subject population data.
        id (str): The name of the subject ID column.
        group (str): The name of the treatment group column.
        total (bool, optional): If True, adds a 'Total' group. Defaults to True.
        missing_group (str, optional): How to handle missing values ("error", "ignore").

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


def count_summary_data(
    population: pl.DataFrame,
    observation: pl.DataFrame,
    id: str,
    group: str,
    variable: str | list[str],
    total: bool = True,
    missing_group: str = "error",
) -> pl.DataFrame:
    """
    Generates numeric summary data (counts and percentages) for observations.
    Does NOT perform string formatting.

    Returns:
        pl.DataFrame: DataFrame with columns:
            - [group]: Group column
            - [variable]: Variable columns
            - n_obs: Count of observations
            - n_subj: Count of unique subjects with observation
            - n_subj_pop: Total subjects in group
            - pct_subj: Percentage of subjects (0-100)
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
    obs = observation.select(id, *variables).join(pop, on=id, how="left")

    for var in variables:
        obs = obs.with_columns(pl.col(var).cast(pl.String).fill_null(config.missing_str))

    # Check for IDs in observation that are not in population
    if not obs[id].is_in(pop[id].to_list()).all():
        missing_ids = (
            obs.filter(~pl.col(id).is_in(pop[id].to_list()))
            .select(id)
            .unique()
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"Some '{id}' values in the observation DataFrame are not present in the population: "
            f"{missing_ids}"
        )

    df_pop = count_subject(
        population=population,
        id=id,
        group=group,
        total=total,
        missing_group=missing_group,
    )

    all_levels_df = []

    # Iterate through hierarchies
    for i in range(1, len(variables) + 1):
        current_vars = variables[:i]

        # Aggregation
        df_obs_counts = obs.group_by(group, *current_vars).agg(
            pl.len().alias("n_obs"), pl.n_unique(id).alias("n_subj")
        )

        # Cross join for all combinations
        unique_groups = df_pop.select(group)
        unique_variables = obs.select(current_vars).unique()
        all_combinations = unique_groups.join(unique_variables, how="cross")

        # Join back
        df_level = (
            all_combinations.join(df_obs_counts, on=[group, *current_vars], how="left")
            .join(df_pop, on=group, how="left")
            .with_columns([pl.col("n_obs").fill_null(0), pl.col("n_subj").fill_null(0)])
        )

        df_level = df_level.with_columns([pl.col(c).cast(pl.String) for c in current_vars])

        # Add missing columns with "__all__"
        for var in variables:
            if var not in df_level.columns:
                df_level = df_level.with_columns(pl.lit("__all__").cast(pl.String).alias(var))

        all_levels_df.append(df_level)

    # Stack
    df_obs = pl.concat(all_levels_df, how="diagonal")

    # Calculate percentage
    df_obs = df_obs.with_columns(pct_subj=(pl.col("n_subj") / pl.col("n_subj_pop") * 100))

    return df_obs


def format_summary_table(
    df: pl.DataFrame,
    group: str,
    variable: str | list[str],
    pct_digit: int = 1,
    max_n_width: int | None = None,
) -> pl.DataFrame:
    """
    Formats numeric summary data into display strings (e.g., "n ( pct)").
    Adds indentation and sorting.
    """
    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = variable

    df_fmt = df.with_columns(
        pct_subj_fmt=(
            pl.when(pl.col("pct_subj").is_null() | pl.col("pct_subj").is_nan())
            .then(0.0)
            .otherwise(pl.col("pct_subj"))
            .round(pct_digit, mode="half_away_from_zero")
            .cast(pl.String)
        )
    )

    if max_n_width is None:
        max_n_width = df_fmt.select(pl.col("n_subj").cast(pl.String).str.len_chars().max()).item()

    max_pct_width = 3 if pct_digit == 0 else 4 + pct_digit

    df_fmt = df_fmt.with_columns(
        [
            pl.col("pct_subj_fmt").str.pad_start(max_pct_width, " "),
            pl.col("n_subj").cast(pl.String).str.pad_start(max_n_width, " ").alias("n_subj_fmt"),
        ]
    ).with_columns(
        n_pct_subj_fmt=pl.concat_str(
            [pl.col("n_subj_fmt"), pl.lit(" ("), pl.col("pct_subj_fmt"), pl.lit(")")]
        )
    )

    # Sorting Logic
    sort_exprs = [pl.col(group)]
    for var in variables:
        # 0 for __all__, 1 for values, 2 for config.missing_str
        sort_key_col = f"__sort_key_{var}__"
        df_fmt = df_fmt.with_columns(
            pl.when(pl.col(var) == "__all__")
            .then(0)
            .when(pl.col(var) == config.missing_str)
            .then(2)
            .otherwise(1)
            .alias(sort_key_col)
        )
        sort_exprs.append(pl.col(sort_key_col))
        sort_exprs.append(pl.col(var))

    df_fmt = df_fmt.sort(sort_exprs).select(pl.exclude(r"^__sort_key_.*$"))

    # Indentation logic
    if len(variables) > 0:
        var_expr = (
            pl.when(pl.col(variables[0]) == config.missing_str)
            .then(pl.lit("Missing"))
            .otherwise(pl.col(variables[0]))
        )

        for i in range(1, len(variables)):
            var_expr = (
                pl.when(pl.col(variables[i]) == "__all__")
                .then(var_expr)
                .when(pl.col(variables[i]) == config.missing_str)
                .then(pl.lit(" " * 4 * i) + pl.lit("Missing"))
                .otherwise(pl.lit(" " * 4 * i) + pl.col(variables[i]))
            )
        df_fmt = df_fmt.with_columns(var_expr.alias("__variable__"))

    df_fmt = df_fmt.with_row_index(name="__id__", offset=1)
    return df_fmt


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
    Legacy wrapper for backward compatibility (mostly for tests that rely on the old signature),
    but now strictly composing the new functions.
    """
    df_raw = count_summary_data(
        population=population,
        observation=observation,
        id=id,
        group=group,
        variable=variable,
        total=total,
        missing_group=missing_group,
    )

    return format_summary_table(
        df=df_raw,
        group=group,
        variable=variable,
        pct_digit=pct_digit,
        max_n_width=max_n_width,
    )
