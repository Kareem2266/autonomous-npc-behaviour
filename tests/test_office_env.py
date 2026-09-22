import pytest

from autonomous_npc.envs.office_env import OfficeEnv


@pytest.mark.parametrize("invalid_size", [1, 0, -1])
def test_rejects_grid_sizes_smaller_than_two(invalid_size):
    with pytest.raises(ValueError, match="grid less than 2x2"):
        OfficeEnv(size=invalid_size)


@pytest.mark.parametrize("valid_size", [2, 5, 8])
def test_stores_valid_grid_size(valid_size):
    environment = OfficeEnv(size=valid_size)

    assert environment.size == valid_size
