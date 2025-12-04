# pyre-strict
from unittest.mock import MagicMock, patch

from tlfyaml.plan import KeywordRegistry, PlanExpander, StudyPlan


def test_keyword_registry_load():
    registry = KeywordRegistry()
    data = {
        "population": [{"name": "pop1", "filter": "f1"}],
        "observation": [{"name": "obs1", "filter": "f2"}],
        "parameter": [{"name": "param1", "filter": "f3"}],
        "group": [{"name": "grp1", "variable": "var1"}],
        "data": [{"name": "ds1", "path": "p1"}]
    }
    
    registry.load_from_dict(data)
    
    assert registry.get_population("pop1").filter == "f1"
    assert registry.get_observation("obs1").filter == "f2"
    assert registry.get_parameter("param1").filter == "f3"
    assert registry.get_group("grp1").variable == "var1"
    assert registry.get_data_source("ds1").path == "p1"

def test_plan_expander_simple():
    registry = KeywordRegistry()
    expander = PlanExpander(registry)
    
    plan_data = {
        "analysis": "ae_summary",
        "population": "pop1",
        "observation": "obs1",
        "parameter": "param1",
        "group": "grp1"
    }
    
    plans = expander.expand_plan(plan_data)
    
    assert len(plans) == 1
    assert plans[0].analysis == "ae_summary"
    assert plans[0].population == "pop1"
    assert plans[0].observation == "obs1"
    assert plans[0].parameter == "param1"

def test_plan_expander_multiple():
    registry = KeywordRegistry()
    expander = PlanExpander(registry)
    
    plan_data = {
        "analysis": "ae_summary",
        "population": ["pop1", "pop2"],
        "observation": ["obs1"],
        "parameter": ["param1", "param2"],
        "group": "grp1"
    }
    
    plans = expander.expand_plan(plan_data)
    
    # 2 pops * 1 obs * 2 params = 4 plans
    assert len(plans) == 4
    
    ids = {p.id for p in plans}
    assert "ae_summary_pop1_obs1_param1" in ids
    assert "ae_summary_pop2_obs1_param2" in ids

def test_study_plan_init():
    study_data = {
        "study": {"name": "Test Study"},
        "data": [{"name": "adsl", "path": "adsl.parquet"}],
        "plans": []
    }
    
    with patch("tlfyaml.plan.pl.read_parquet") as mock_read:
        mock_read.return_value = MagicMock()
        
        plan = StudyPlan(study_data)
        
        assert "adsl" in plan.datasets
        assert plan.keywords.get_data_source("adsl") is not None
