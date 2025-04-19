
import robosuite, os
print("robosuite ver =", robosuite.__version__)
print("located at    =", os.path.dirname(robosuite.__file__))

from robosuite.controllers import load_part_controller_config
ctrl_cfg = load_part_controller_config(default_controller="OSC_POSITION")
ctrl_cfg["control_delta"] = True     # 仍可在这里改参数


from robosuite.controllers.composite import ALL_COMPOSITE_CONTROLLERS
print("Registered composite controllers:", ALL_COMPOSITE_CONTROLLERS)
