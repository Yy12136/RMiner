"""DDS bridge for a pair of BrainCo Revo2 hands in Isaac Lab."""

from dds.inspire_dds import InspireDDS


class BraincoDDS(InspireDDS):
    """Reuses the 12-channel shared-memory transport with BrainCo topics.

    Commands are normalized Revo2 positions, in right-hand then left-hand
    order, matching the native BrainCo controller in xr_teleoperate.
    """

    def __init__(self, node_name: str = "brainco"):
        super().__init__(node_name)
        self.input_shm_name = "isaac_brainco_state"
        self.output_shm_name = "isaac_brainco_cmd"
