from app.services.guided_builder import DeckAnalyzer, front_cost

MDFC = "{2}{R} // {1}{G}"


def test_front_cost_keeps_front_face_only():
    assert front_cost(MDFC) == "{2}{R}"
    assert front_cost("{U}") == "{U}"
    assert front_cost(None) == ""
    assert front_cost(MDFC).count("{G}") == 0


def test_estimate_cmc_uses_front_face():
    assert DeckAnalyzer(None)._estimate_cmc(MDFC) == 3
