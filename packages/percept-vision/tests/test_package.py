import percept


def test_version_is_1_0_dev():
    assert percept.__version__.startswith("1.0.0")


def test_public_surface_is_small():
    assert len(percept.__all__) <= 30
