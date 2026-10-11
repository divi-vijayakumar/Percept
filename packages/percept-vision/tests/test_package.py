import percept


def test_version_is_the_developer_preview():
    assert percept.__version__ == "0.9.0"


def test_public_surface_is_small():
    assert len(percept.__all__) <= 30
