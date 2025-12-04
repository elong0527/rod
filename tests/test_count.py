# pyre-strict
import polars as pl
import pytest

from tlfyaml.count import count_subject, count_subject_with_observation


@pytest.fixture
def population_data():
    return pl.DataFrame({
        "USUBJID": ["01", "02", "03", "04", "05"],
        "TRT01A": ["A", "A", "B", "B", "B"]
    })

@pytest.fixture
def observation_data():
    return pl.DataFrame({
        "USUBJID": ["01", "01", "03", "04"],
        "AESOC": ["Infection", "Headache", "Infection", "Headache"]
    })

def test_count_subject(population_data):
    result = count_subject(
        population=population_data,
        id="USUBJID",
        group="TRT01A",
        total=True
    )
    
    # Expected: A=2, B=3, Total=5
    assert result.filter(pl.col("TRT01A") == "A")["n_subj_pop"][0] == 2
    assert result.filter(pl.col("TRT01A") == "B")["n_subj_pop"][0] == 3
    assert result.filter(pl.col("TRT01A") == "Total")["n_subj_pop"][0] == 5

def test_count_subject_no_total(population_data):
    result = count_subject(
        population=population_data,
        id="USUBJID",
        group="TRT01A",
        total=False
    )
    
    assert "Total" not in result["TRT01A"].to_list()
    assert result.height == 2

def test_count_subject_missing_group_error():
    pop_missing = pl.DataFrame({
        "USUBJID": ["01", "02"],
        "TRT01A": ["A", None]
    })
    
    with pytest.raises(ValueError, match="Missing values found"):
        count_subject(pop_missing, "USUBJID", "TRT01A", missing_group="error")

def test_count_subject_duplicate_id_error():
    pop_dup = pl.DataFrame({
        "USUBJID": ["01", "01"],
        "TRT01A": ["A", "B"]
    })
    
    with pytest.raises(ValueError, match="not unique"):
        count_subject(pop_dup, "USUBJID", "TRT01A")

def test_count_subject_with_observation(population_data, observation_data):
    result = count_subject_with_observation(
        population=population_data,
        observation=observation_data,
        id="USUBJID",
        group="TRT01A",
        variable="AESOC",
        total=True
    )
    
    # Check structure
    assert "n_obs" in result.columns
    assert "n_subj" in result.columns
    assert "pct_subj_fmt" in result.columns
    
    # Check specific values
    # Group A: 2 subjects. Obs: 01 (Infection, Headache). 
    # Infection: 1 subj (50%), Headache: 1 subj (50%)
    
    row_a_inf = result.filter((pl.col("TRT01A") == "A") & (pl.col("AESOC") == "Infection"))
    assert row_a_inf["n_subj"][0] == 1
    assert row_a_inf["pct_subj"][0] == 50.0
    
    # Group B: 3 subjects. Obs: 03 (Infection), 04 (Headache).
    # Infection: 1 subj (33.3%), Headache: 1 subj (33.3%)
    row_b_inf = result.filter((pl.col("TRT01A") == "B") & (pl.col("AESOC") == "Infection"))
    assert row_b_inf["n_subj"][0] == 1
    assert abs(row_b_inf["pct_subj"][0] - 33.3) < 0.1

def test_count_subject_with_observation_missing_id_in_pop(population_data):
    obs_bad = pl.DataFrame({
        "USUBJID": ["99"], # Not in pop
        "AESOC": ["X"]
    })
    
    with pytest.raises(ValueError, match="not present in the population"):
        count_subject_with_observation(
            population=population_data,
            observation=obs_bad,
            id="USUBJID",
            group="TRT01A",
            variable="AESOC"
        )
