"""Contains the merge network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 100  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

ADDITIONAL_NET_PARAMS = {
    "merge_length": 100,
    "pre_merge_length": 200,
    "post_merge_length": 100,
    "merge_lanes_from_east": 2,
    "merge_lanes_from_west": 1,  
    "pre_highway_lanes": 4,
    "post_highway_lanes": 5,
    "speed_limit": 30,
}


class SingalNetwork(Network):


    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        angle = pi / 4
        angle2 = pi / 6
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]

        nodes = [
            {
                "id": "inflow_highway",
                "x": -INFLOW_EDGE_LEN,
                "y": 0
            },
            {
                "id": "left",
                "y": 0,
                "x": 0
            },
            {
                "id": "center",
                "y": 0,
                "x": premerge,
                "radius": 10
            },
            {
                "id": "right",
                "y": 0,
                "x": premerge + postmerge
            },
            {
                "id": "inflow_merge_west",
                "x": premerge - (merge + INFLOW_EDGE_LEN) * cos(angle2),
                "y": -(merge + INFLOW_EDGE_LEN) * sin(angle2)
            },
            {
                "id": "inflow_merge_east",
                "x": premerge - (merge + INFLOW_EDGE_LEN) * cos(angle),
                "y": -(merge + INFLOW_EDGE_LEN) * sin(angle)
            },
            {
                "id": "bottom",
                "x": premerge - merge * cos(angle),
                "y": -merge * sin(angle)
            },
        ]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]

        edges = [{
            "id": "inflow_highway",
            "type": "highwayType1",
            "from": "inflow_highway",
            "to": "left",
            "length": INFLOW_EDGE_LEN
        }, {
            "id": "left",
            "type": "highwayType1",
            "from": "left",
            "to": "center",
            "length": premerge
        }, {
            "id": "inflow_merge_west",
            "type": "mergeType1",
            "from": "inflow_merge_west",
            "to": "bottom",
            "length": INFLOW_EDGE_LEN
        },
        {
            "id": "inflow_merge_east",
            "type": "mergeType2",
            "from": "inflow_merge_east",
            "to": "bottom",
            "length": INFLOW_EDGE_LEN
        },
         {
            "id": "bottom",
            "type": "mergeType2",
            "from": "bottom",
            "to": "center",
            "length": merge
        }, {
            "id": "center",
            "type": "highwayType2",
            "from": "center",
            "to": "right",
            "length": postmerge
        },
        
        ]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        pre_h_lanes = net_params.additional_params["pre_highway_lanes"]
        post_h_lanes = net_params.additional_params["post_highway_lanes"]
        m_lanes_e = net_params.additional_params["merge_lanes_from_east"]
        m_lanes_w = net_params.additional_params["merge_lanes_from_west"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType1",
            "numLanes": pre_h_lanes,
            "speed": speed
        }, 
        {
            "id": "highwayType2",
            "numLanes": post_h_lanes,
            "speed": speed
        },
        {
            "id": "mergeType1",
            "numLanes": m_lanes_w,
            "speed": speed
        },
        {
            "id": "mergeType2",
            "numLanes": m_lanes_e,
            "speed": speed
        }
        ]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "inflow_highway": ["inflow_highway", "left", "center"],
            "left": ["left", "center"],
            "center": ["center"],
            "inflow_merge_west": ["inflow_merge_west", "bottom", "center"],
            "inflow_merge_east": ["inflow_merge_east", "bottom", "center"],

            "bottom": ["bottom", "center"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]

        edgestarts = [("inflow_highway", 0), ("left", INFLOW_EDGE_LEN + 0.1),
                      ("center", INFLOW_EDGE_LEN + premerge + 22.6),
                      ("inflow_merge_east",
                       INFLOW_EDGE_LEN + premerge + postmerge + 30),
                       ("inflow_merge_west",
                       INFLOW_EDGE_LEN + premerge + postmerge + 22.6),
                      ("bottom",
                       2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.7)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]

        internal_edgestarts = [
            (":left", INFLOW_EDGE_LEN), (":center",
                                         INFLOW_EDGE_LEN + premerge + 0.1),
            (":bottom", 2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.6)
        ]

        return internal_edgestarts