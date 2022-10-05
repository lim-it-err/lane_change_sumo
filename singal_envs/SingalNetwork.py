from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
import numpy as np
import json

class SingalJC(Network):
    """
    net_params

    """
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Instantiate the network class."""


        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
    
    def specify_nodes(self, node_file="network_nodes.json"):
        with open(node_file, "r") as f:
            data = json.load(f)
            node_data = data["nodes"]
        return node_data


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