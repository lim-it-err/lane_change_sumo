from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController

from envs import SingalEnv
from envs import SingalNetwork

# time horizon of a single rollout
HORIZON = 1000
# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS * 4

vehicles = VehicleParams()
vehicles.add(
    veh_id = "car",
    acceleration_controller=(),
    lane_change_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=0,
    num_vehicles=0,
    car_following_params=None,
    lane_change_params=None,
    color=None,
)
vehicles.add(
    veh_id = "bus",
    acceleration_controller=(),
    lane_change_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=0,
    num_vehicles=0,
    car_following_params=None,
    lane_change_params=None,
    color=None,
)
vehicles.add(
    veh_id = "emergency",
    acceleration_controller=(),
    lane_change_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=0,
    num_vehicles=0,
    car_following_params=None,
    lane_change_params=None,
    color=None,
)

inflow = InFlows()

inflow.add(
    veh_type="car",
    edge="436220650",
    vehs_per_hour=1800,
    departLane="random",
    end="3600.00",
    departSpeed=10)
inflow.add(
    veh_type="car",
    edge="558884038",
    vehs_per_hour=1800,
    departLane="random",
    end="3600.00",
    departSpeed=10)
inflow.add(
    veh_type="car",
    edge="70707218-AddedOffRampNode",
    vehs_per_hour=1800,
    departLane="random",
    end="3600.00",
    departSpeed=10)

net_params = NetParams(inflows=inflow)

flow_params=dict(
    exp_tag="Singal",
    env_name=SingalEnv,
    network=SingalNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),
    env=EnvParams(
        warmup_steps=40,
        sims_per_step=1,
        horizon=HORIZON,
    ),
    net=NetParams(
        inflows=inflow,
    ),
    veh=vehicles,

)