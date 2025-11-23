import polars as pl

def _to_pop(
    population,
    id,
    group,
    total = True,
    missing_group = "error",
) -> pl.DataFrame:
    
    # prepare data
    pop = population.select(id, group)

    # validate data
    if pop[id].is_duplicated().any():
        raise ValueError(f"The '{id}' column in the population DataFrame is not unique.")

    if missing_group == "error" and pop[group].is_null().any():
        raise ValueError(f"Missing values found in the '{group}' column of the population DataFrame, and 'missing_group' is set to 'error'.")

    # prepare analysis ready data
    df_pop = pop.group_by(group).agg(pl.len().alias("n_subj_pop")).sort(group)

    # handle total column
    if total:
        u_pop = pop[group].unique().sort().to_list()
        pop_total = pop.with_columns(pl.lit("Total").alias(group))
        pop = pl.concat([pop, pop_total]).with_columns(
            pl.col(group).cast(pl.Enum(u_pop + ["Total"]))
        )
    
    return pop

def count_subject(
    population,
    id,
    group,
    total = True,
    missing_group = "error",
) -> pl.DataFrame:
    """
    Counts subjects by group and optionally includes a 'Total' column.

    Args:
        population (pl.DataFrame): DataFrame containing subject population data,
                                   must include 'id' and 'group' columns.
        id (str): The name of the subject ID column (e.g., "USUBJID").
        group (str): The name of the treatment group column (e.g., "TRT01A").
        total (bool, optional): If True, adds a 'Total' group with counts across all groups. Defaults to True.
        missing_group (str, optional): How to handle missing values in the group column.
                                       "error" will raise a ValueError. Defaults to "error".

    Returns:
        pl.DataFrame: A DataFrame with subject counts ('n_subj_pop') for each group.
    """

    pop = _to_pop(
        population = population,
        id = id,
        group = group,
        total = total,
        missing_group = missing_group,
    )
 
    return pop.group_by(group).agg(pl.len().alias("n_subj_pop")).sort(group)

def count_subject_with_observation(
        population,
        observation,
        id,
        group,
        variable,
        total = True,
        missing_group = "error",
        pct_digit = 1
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
        pct_digit (int, optional): Number of decimal places for percentage formatting. Defaults to 1.

    Returns:
        pl.DataFrame: A DataFrame with counts and percentages of subjects and observations
                      grouped by 'group' and 'variable'.
    """

    # prepare data
    pop = _to_pop(
        population = population,
        id = id,
        group = group,
        total = total,
        missing_group = missing_group,
    )

    obs = observation.select(id, variable).join(pop, on = id, how = "left")

    if obs[id].is_in(pop[id]).all() == False:
        # Get IDs that are in obs but not in pop
        missing_ids = obs.filter(~pl.col(id).is_in(pop[id])).select(id).unique().to_series().to_list()
        raise ValueError(f"Some '{id}' values in the observation DataFrame are not present in the population DataFrame: {missing_ids}")

    df_pop =count_subject(
        population = population,
        id = id,
        group = group,
        total = total,
        missing_group = missing_group,
    )

    df_obs = (
        obs
        .group_by(group, variable)
        .agg(
            pl.len().alias("n_obs"),
            pl.n_unique(id).alias("n_subj")
        )
        .join(df_pop, on = group)
        .with_columns(
            pct_subj = (pl.col("n_subj") / pl.col("n_subj_pop") * 100)
        )
        .with_columns(
            pct_subj_fmt = pl.col("pct_subj").round(pct_digit, mode = "half_away_from_zero").cast(pl.String)
        )
        .with_columns(
            n_pct_subj_fmt = pl.format("{} ({})", pl.col("n_subj"), pl.col("pct_subj_fmt"))
        )
        .sort(group, variable)
    )

    return df_obs
