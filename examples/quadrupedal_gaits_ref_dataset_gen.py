import os
import sys
import json 
import numpy as np

import crocoddyl
import example_robot_data
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

def save_ref_trajectory(xs_list, ps_list, feet_ids, ref_trajectory_file):
    N_frames = len(xs_list)
    ref_trajectory = {}
    ref_trajectory["LoopMode"] = "Wrap"
    ref_trajectory["FrameDuration"] = 0.01*N_frames
    ref_trajectory["EnableCycleOffsetPosition"] = True
    ref_trajectory["EnableCycleOffsetRotation"] = True
    ref_trajectory["MotionWeight"] = 1.0
    ref_trajectory["Frames"] = []

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12

    for i, x in enumerate(xs_list):
        res = x[:POS_SIZE + ROT_SIZE + JOINT_POS_SIZE].tolist() #body pos ori joint pos
        body_pos = x[:POS_SIZE]
        for f_id in feet_ids:
            foot_pos = np.array(ps_list[str(f_id)][i])
            res += (foot_pos-body_pos).tolist() # foot pos
        joint_vel = x[POS_SIZE + ROT_SIZE + JOINT_POS_SIZE:]
        res += x[POS_SIZE + ROT_SIZE + JOINT_POS_SIZE:].tolist() # body vel joint vel

        for f_id in feet_ids:
            res += [0, 0, 0] # foot vel
        ref_trajectory["Frames"].append(res)
    print([len(ref_trajectory["Frames"][k]) for k in range(len(ref_trajectory["Frames"]))])
    with open(ref_trajectory_file, 'w') as outfile:
        json.dump(ref_trajectory, outfile)
# x = [0.00000, 0.00000, 0.33664, -0.00000, -0.00000, 0.00000, 1.00000, 0.03194, 0.78700, -1.48894, -0.08535, 0.80080, -1.51533, 0.03109, 0.75470, -1.42664, -0.08302, 0.76849, -1.45315, 0.15776, -0.12481, -0.29699, 0.15774, 0.10481, -0.29702, -0.20824, -0.12480, -0.30523, -0.20826, 0.10480, -0.30526, 0.00000, 0.00408, -0.12840, 0.00000, 0.00000, -0.29563, 0.10464, 0.44520, -0.83136, 0.10272, 0.54360, -0.77664, -0.14160, 0.38592, -0.73296, -0.14304, 0.60168, -0.89184, -0.00370, 0.02690, 0.10606, -0.03973, 0.03932, 0.11548, -0.00179, -0.04556, 0.10773, -0.04073, -0.03280, 0.11275]
# len(x) = 42
def generate_random_gait_phases(kwarg_dict):
    bounding_step_height_range = kwarg_dict['bounding_step_height_range']
    bounding_step_length_range = kwarg_dict['bounding_step_length_range']
    bounding_time_step_range = kwarg_dict['bounding_time_step_range']
    bounding_step_knots_range = kwarg_dict['bounding_step_knots_range']
    bounding_support_knots_range = kwarg_dict['bounding_support_knots_range']

    jumping_step_height_range = kwarg_dict['jumping_step_height_range']
    jumping_step_length_range = kwarg_dict['jumping_step_length_range']
    jumping_time_step_range = kwarg_dict['jumping_time_step_range']
    jumping_ground_knots_range = kwarg_dict['jumping_ground_knots_range']
    jumping_flying_knots_range = kwarg_dict['jumping_flying_knots_range']

    bounding_step_height = np.random.uniform(bounding_step_height_range[0], bounding_step_height_range[-1])
    bounding_step_length = np.random.uniform(bounding_step_length_range[0], bounding_step_length_range[-1])
    bounding_time_step = np.random.uniform(bounding_time_step_range[0], bounding_time_step_range[-1])
    bounding_step_knots = np.random.randint(bounding_step_knots_range[0], bounding_step_knots_range[-1])
    bounding_support_knots = np.random.randint(bounding_support_knots_range[0], bounding_support_knots_range[-1])
    
    jumping_step_height = np.random.uniform(jumping_step_height_range[0], jumping_step_height_range[-1])
    jumping_step_length = np.random.uniform(jumping_step_length_range[0], jumping_step_length_range[-1])
    jumping_time_step = np.random.uniform(jumping_time_step_range[0], jumping_time_step_range[-1])
    jumping_ground_knots = np.random.randint(jumping_ground_knots_range[0], jumping_ground_knots_range[-1])
    jumping_flying_knots = np.random.randint(jumping_flying_knots_range[0], jumping_flying_knots_range[-1])
    gait_type = np.random.choice(['bounding', 'jumping'], 1, p=[0.5, 0.5])
    if gait_type == 'bounding':
        GAITPHASES = [{
                    'bounding': {
                        'stepLength': bounding_step_length,
                        'stepHeight': bounding_step_height,
                        'timeStep': bounding_time_step,
                        'stepKnots': bounding_step_knots,
                        'supportKnots': bounding_support_knots
                    }
                }]
    else: 
        GAITPHASE = {
                    'jumping': {
                                'jumpHeight': jumping_step_height,
                                'jumpLength': [jumping_step_length, 0.0, 0.],
                                'timeStep': jumping_time_step,
                                'groundKnots': jumping_ground_knots,
                                'flyingKnots': jumping_flying_knots
                    }
                    }
    yield GAITPHASE  

# generate random gait trajectories
def generate_random_gait_trajectories(kwarg_dict):
    N_trajectories = kwarg_dict['N_trajectories']
    ref_trajectory_file = kwarg_dict['ref_trajectory_file']
    t_count = 0
    solver = [None] * N_trajectories
    gait_phases = generate_random_gait_phases(kwarg_dict)
    for phase in gait_phases: 
        for key, value in phase.items():
            if key == 'bounding':
                # Creating a bounding problem
                solver[t_count] = crocoddyl.SolverFDDP(
                    gait.createBoundingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                                value['stepKnots'], value['supportKnots']))
            elif key == 'jumping':
                # Creating a jumping problem
                solver[t_count] = crocoddyl.SolverFDDP(
                    gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                                value['groundKnots'], value['flyingKnots']))
            else:
                raise ValueError("Unknown gait phase")
                    # Added the callback functions
            if WITHDISPLAY and WITHPLOT:
                display = crocoddyl.GepettoDisplay(robot, 10, 10, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
                solver[t_count].setCallbacks(
                    [crocoddyl.CallbackLogger(),
                    # crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackDisplay(display)])
            elif WITHDISPLAY:
                display = crocoddyl.GepettoDisplay(robot, 10, 10, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
                solver[t_count].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
            elif WITHPLOT:
                solver[t_count].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
            else:
                solver[t_count].setCallbacks([crocoddyl.CallbackVerbose()])

            # Solving the problem with the DDP solver
            xs = [x0] * (solver[t_count].problem.T + 1)
            us = solver[t_count].problem.quasiStatic([x0] * solver[t_count].problem.T)
            status = solver[t_count].solve(xs, us, 200, False)
            if key == 'bounding':
                bound_solve_status_list.append(status)
            elif key == 'jumping':
                jump_solve_status_list.append(status)
            else:
                raise ValueError("Unknown gait phase")
            if status:
                x0 = solver[t_count].xs[-1]
                t_count += 1
 
            solve_status_list.append(status)

            # Defining the final state as initial one for the next phase
            # Display the entire motion
    if WITHDISPLAY:
        display = crocoddyl.GepettoDisplay(robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        for j, phase in enumerate(gait_phase):
            if not solve_status_list[t_count]:
                 continue
            fs = display.getForceTrajectoryFromSolver(solver[t_count])
            ps = display.getFrameTrajectoryFromSolver(solver[t_count])

            models = solver[t_count].problem.runningModels.tolist() + [solver[t_count].problem.terminalModel]
            dts = [m.dt if hasattr(m, "differential") else 0. for m in models]
            xs_list = xs_list +  [x for x in solver[t_count].xs]
            fs_list = fs_list + fs
            dts_list = dts_list + dts
            for key, value in ps.items():
                ps_dict[key] +=  value
    print(f"iter {i}  bounding: {sum(bound_solve_status_list)}/{len(bound_solve_status_list)} \
                      jumping: {sum(jump_solve_status_list)}/{len(jump_solve_status_list)} \
                      total: {sum(solve_status_list)}/{len(solve_status_list)}")


    print(f"number of success solve: {sum(solve_status_list)}/{len(solve_status_list)}")
    print(f"number of success bound solve: {sum(bound_solve_status_list)}/{len(bound_solve_status_list)}")
    print(f"number of success jump solve: {sum(jump_solve_status_list)}/{len(jump_solve_status_list)}")

    print(f"ps_dict: {ps_dict.keys()}")
    print(f"lfFootId: {lfFootId}, rfFootId: {rfFootId}, lhFootId: {lhFootId}, rhFootId: {rhFootId}")
    #IsaacGym order [FL, FR, RL, RR].

    display = crocoddyl.GepettoDisplay(robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    display.display(xs_list, fs=fs_list, ps=ps_dict, dts=dts_list, factor=1.)
    save_ref_trajectory(xs_list, ps_dict, feet_ids, ref_trajectory_file)



def generate_gait_phases(kwarg_dict):
    
    bounding_step_height_range = kwarg_dict['bounding_step_height_range']
    bounding_step_length_range = kwarg_dict['bounding_step_length_range']
    bounding_time_step_range = kwarg_dict['bounding_time_step_range']
    bounding_step_knots_range = kwarg_dict['bounding_step_knots_range']
    bounding_support_knots = kwarg_dict['bounding_support_knots']

    jumping_step_height_range = kwarg_dict['jumping_step_height_range']
    jumping_step_length_range = kwarg_dict['jumping_step_length_range']
    jumping_time_step_range = kwarg_dict['jumping_time_step_range']
    jumping_ground_knots_range = kwarg_dict['jumping_ground_knots_range']
    jumping_flying_knots_range = kwarg_dict['jumping_flying_knots_range']



    for bounding_step_height in bounding_step_height_range:
        for bounding_step_length in bounding_step_length_range:
            for bounding_time_step in bounding_time_step_range:
                for bounding_step_knots in bounding_step_knots_range:
                    for bounding_support_knots in bounding_support_knots:
                        for jumping_step_height in jumping_step_height_range:
                            for jumping_step_length in jumping_step_length_range:
                                for jumping_time_step in jumping_time_step_range:
                                    for jumping_ground_knot in jumping_ground_knots_range:
                                        for jumping_flying_knot in jumping_flying_knots_range:
                                            GAITPHASES = [{
                                                'bounding': {
                                                    'stepLength': bounding_step_length,
                                                    'stepHeight': bounding_step_height,
                                                    'timeStep': bounding_time_step,
                                                    'stepKnots': bounding_step_knots,
                                                    'supportKnots': bounding_support_knots
                                                }
                                            }, {
                                                'jumping': {
                                                            'jumpHeight': jumping_step_height,
                                                            'jumpLength': [jumping_step_length, 0.0, 0.],
                                                            'timeStep': jumping_time_step,
                                                            'groundKnots': jumping_ground_knot,
                                                            'flyingKnots': jumping_flying_knot
                                                }
                                            }]
                                            yield GAITPHASES

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Loading the robot model
# robot = example_robot_data.load('robot')
robot = example_robot_data.load('a1')

# Defining the initial state of the robot
q0 = robot.model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
# lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
lfFoot, rfFoot, lhFoot, rhFoot = 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
gait = SimpleQuadrupedalGaitProblem(robot.model, lfFoot, rfFoot, lhFoot, rhFoot)
cameraTF = [5., 3.68, 1.84, 0.2, 0.62, 0.72, 0.22]
lfFootId = robot.model.getFrameId(lfFoot)
rfFootId = robot.model.getFrameId(rfFoot)
lhFootId = robot.model.getFrameId(lhFoot)
rhFootId = robot.model.getFrameId(rhFoot)
feet_ids = [lfFootId, rfFootId, lhFootId, rhFootId]
compb_counter = 1
kwargs = {}
kwargs['bounding_step_height_range'] = np.arange(0.05, 0.15, 0.05) # length 1
compb_counter*=len(kwargs['bounding_step_height_range'])
kwargs['bounding_step_length_range'] = np.arange(0.1, 0.2, 0.05) # length 2
compb_counter*=len(kwargs['bounding_step_length_range'])
kwargs['bounding_time_step_range'] = [1e-2] # length 1
compb_counter*=len(kwargs['bounding_time_step_range'])
kwargs['bounding_step_knots_range'] = range(15, 25, 5) # length 2
compb_counter*=len(kwargs['bounding_step_knots_range'])
kwargs['bounding_support_knots'] = [5]
compb_counter*=len(kwargs['bounding_support_knots'])
kwargs['jumping_step_height_range'] = np.arange(0.4, 0.6, 0.1) # length 3
compb_counter*=len(kwargs['jumping_step_height_range'])
kwargs['jumping_step_length_range'] = np.arange(0.5, 1.0, 0.1)  # length 4
compb_counter*=len(kwargs['jumping_step_length_range'])
kwargs['jumping_time_step_range'] = [1e-2] # length 1
compb_counter*=len(kwargs['jumping_time_step_range'])
kwargs['jumping_ground_knots_range'] = range(5, 15, 5) # length 2
compb_counter*=len(kwargs['jumping_ground_knots_range'])
kwargs['jumping_flying_knots_range'] = range(15, 25, 5) # length 2
compb_counter*=len(kwargs['jumping_flying_knots_range'])
# 1*2*1*2*3*4*1*2*2 = 384

gait_phases = generate_gait_phases(kwargs)
solver = [None] * compb_counter
fs_list = []
dts_list = []
ps_dict = {str(f_id): [] for f_id in feet_ids}
xs_list = []
solve_status_list = []
bound_solve_status_list = []
jump_solve_status_list = []
for i, gait_phase in enumerate(gait_phases): # 384
    # if i % 5 == 0:
    #     x0 = np.concatenate([q0, v0])
    for j, phase in enumerate(gait_phase): # 2
        for key, value in phase.items():
            if key == 'bounding':
                # Creating a bounding problem
                solver[2*i+j] = crocoddyl.SolverFDDP(
                    gait.createBoundingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                                value['stepKnots'], value['supportKnots']))
            elif key == 'jumping':
                # Creating a jumping problem
                solver[2*i+j] = crocoddyl.SolverFDDP(
                    gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                                value['groundKnots'], value['flyingKnots']))
            else:
                raise ValueError("Unknown gait phase")
                    # Added the callback functions
            if WITHDISPLAY and WITHPLOT:
                display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
                solver[2*i+j].setCallbacks(
                    [crocoddyl.CallbackLogger(),
                    # crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackDisplay(display)])
            elif WITHDISPLAY:
                display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
                solver[2*i+j].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
            elif WITHPLOT:
                solver[2*i+j].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
            else:
                solver[2*i+j].setCallbacks([crocoddyl.CallbackVerbose()])

            # Solving the problem with the DDP solver
            xs = [x0] * (solver[2*i+j].problem.T + 1)
            us = solver[2*i+j].problem.quasiStatic([x0] * solver[2*i+j].problem.T)
            status = solver[2*i+j].solve(xs, us, 100, False)
            if key == 'bounding':
                bound_solve_status_list.append(status)
            elif key == 'jumping':
                jump_solve_status_list.append(status)
            else:
                raise ValueError("Unknown gait phase")
            if status:
                x0 = solver[2*i+j].xs[-1]
 
            solve_status_list.append(status)

            # Defining the final state as initial one for the next phase
            # Display the entire motion
    if WITHDISPLAY:
        display = crocoddyl.GepettoDisplay(robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        for j, phase in enumerate(gait_phase):
            # display.displayFromSolver(solver[2*i+j])
            # if not solve_status_list[2*i+j]:
            #     continue
            fs = display.getForceTrajectoryFromSolver(solver[2*i+j])
            ps = display.getFrameTrajectoryFromSolver(solver[2*i+j])

            models = solver[2*i+j].problem.runningModels.tolist() + [solver[2*i+j].problem.terminalModel]
            dts = [m.dt if hasattr(m, "differential") else 0. for m in models]
            xs_list = xs_list +  [x for x in solver[2*i+j].xs]
            fs_list = fs_list + fs
            dts_list = dts_list + dts
            for key, value in ps.items():
                ps_dict[key] +=  value
            #print(ps)
            #break
            #ps_list = ps_list + ps
    print(f"iter {i}  bounding: {sum(bound_solve_status_list)}/{len(bound_solve_status_list)} \
                      jumping: {sum(jump_solve_status_list)}/{len(jump_solve_status_list)} \
                      total: {sum(solve_status_list)}/{len(solve_status_list)}")
    if i >=1:
        break
print(f"number of success solve: {sum(solve_status_list)}/{len(solve_status_list)}")
print(f"number of success bound solve: {sum(bound_solve_status_list)}/{len(bound_solve_status_list)}")
print(f"number of success jump solve: {sum(jump_solve_status_list)}/{len(jump_solve_status_list)}")

print(f"ps_dict: {ps_dict.keys()}")
print(f"lfFootId: {lfFootId}, rfFootId: {rfFootId}, lhFootId: {lhFootId}, rhFootId: {rhFootId}")
#IsaacGym order [FL, FR, RL, RR].

display = crocoddyl.GepettoDisplay(robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
display.display(xs_list, fs=fs_list, ps=ps_dict, dts=dts_list, factor=1.)

ref_trajectory_file = "ref_trajectory.txt"
save_ref_trajectory(xs_list, ps_dict, feet_ids, ref_trajectory_file)
