"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network.
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.core.params import NetParams, InFlows, SumoCarFollowingParams
from SingalNetwork import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import RLController
from SingalEnv import SingalEnv
from SingalNetwork import SingalNetwork


HORIZON = 600
N_ROLLOUTS = 20
N_CPUS = 6
FLOW_RATE = 10000
NUM_RL = 250

additional_net_params = ADDITIONAL_NET_PARAMS.copy()


vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
    ),
    num_vehicles=0)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
    ),
    num_vehicles=0)



inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour= FLOW_RATE * 0.1,
    departLane="free",
    departSpeed=10)

inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=FLOW_RATE * 0.9,
    departLane="free",
    departSpeed=10
    )

inflow.add(
    veh_type="rl",
    edge="inflow_merge_west",
    vehs_per_hour=3000,
    departLane="free",
    departSpeed=7.5
    )
inflow.add(
    veh_type="rl",
    edge="inflow_merge_east",
    vehs_per_hour=3000,
    departLane="free",
    departSpeed=7.5
    )

flow_params = dict(
    exp_tag="SingalEnv",

    env_name=SingalEnv,

    network=SingalNetwork,

    simulator='traci',

    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
    ),

    env=EnvParams(
        horizon=HORIZON,
        # evaluate = True,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL,
        },
    ),

    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),
    veh=vehicles,
    initial=InitialConfig(),
)
