from app.jobs.scryfall_sync import extract_card_data


def _card(**kw):
    base = {"id": "abc", "oracle_id": "o1", "name": "X", "layout": "normal", "released_at": "2024-01-01"}
    base.update(kw)
    return base


def test_transform_dfc_uses_face_text_and_costs():
    card = _card(
        name="Delver // Insectile",
        layout="transform",
        card_faces=[
            {"name": "Delver", "mana_cost": "{U}", "oracle_text": "Flip it."},
            {"name": "Insectile", "mana_cost": "", "oracle_text": "Flying"},
        ],
    )
    data = extract_card_data(card, set())
    assert data["oracle_text"] == "Flip it.\n//\nFlying"
    assert data["mana_cost"] == "{U}"


def test_modal_dfc_joins_both_costs():
    card = _card(
        layout="modal_dfc",
        card_faces=[
            {"mana_cost": "{2}{R}", "oracle_text": "A"},
            {"mana_cost": "{1}{G}", "oracle_text": "B"},
        ],
    )
    data = extract_card_data(card, set())
    assert data["mana_cost"] == "{2}{R} // {1}{G}"


def test_top_level_values_untouched():
    card = _card(
        mana_cost="{1}{W}",
        oracle_text="Top text",
        card_faces=[{"mana_cost": "{9}", "oracle_text": "Face"}],
    )
    data = extract_card_data(card, set())
    assert data["oracle_text"] == "Top text"
    assert data["mana_cost"] == "{1}{W}"


def test_no_faces_unchanged():
    data = extract_card_data(_card(mana_cost="{G}", oracle_text="Hi"), set())
    assert (data["mana_cost"], data["oracle_text"]) == ("{G}", "Hi")
    data = extract_card_data(_card(), set())
    assert not data["oracle_text"] and not data["mana_cost"]
