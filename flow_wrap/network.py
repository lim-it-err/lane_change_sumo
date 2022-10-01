from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
import numpy as np
class SingalJC(Network):
    """
    net_params

    """


    pass

if __name__ == "__main__":
    from flow.core.params import VehicleParams
    from flow.core.params import NetParams

    additional_params = {
        'scaling':1,
        'speed_limit':1,
    }

    singal_jc = SingalJC(
        name = 'singalJC',
        vehicles = VehicleParams(),
        net_params = NetParams(additional_params)
    )