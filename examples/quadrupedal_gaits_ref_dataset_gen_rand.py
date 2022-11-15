import os
import sys
import json 
import numpy as np
import pickle
import sys, tty, termios
import crocoddyl
import example_robot_data
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution
def playback_trajectory(data, playback_speed=1.0, repeat=False):
    xs_list = data["xs_list"]
    ps_dict  = data["ps_dict"]
    dts_list = data["dts_list"]
    fs_list = data["fs_list"]
    # repeat display till user control + c
    print("Press ctrl+c to exit")
    while True:
        display = crocoddyl.GepettoDisplay(robot,frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        display.display(xs_list, fs=fs_list, ps=ps_dict, dts=dts_list, factor=playback_speed)
        if not repeat:
            break
        # continously check control+c
        try:
            pass # do nothing
        except KeyboardInterrupt:
            break
        

def save_ref_trajectory(xs_list, ps_list, feet_ids, ref_trajectory_file):
    """ saves the reference trajectory in a json file"""
    N_frames = len(xs_list)
    ref_trajectory = {}
    ref_trajectory["LoopMode"] = "Wrap"
    ref_trajectory["FrameDuration"] = 0.01
    ref_trajectory["EnableCycleOffsetPosition"] = True
    ref_trajectory["EnableCycleOffsetRotation"] = True
    ref_trajectory["MotionWeight"] = 1.0
    ref_trajectory["Frames"] = []

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    max_forward_vel = 0.0
    for i, x in enumerate(xs_list):
        res = x[:POS_SIZE + ROT_SIZE + JOINT_POS_SIZE].tolist() #body pos ori joint pos
        body_pos = x[:POS_SIZE]
        body_quat = x[POS_SIZE:POS_SIZE + ROT_SIZE]
        body_rot = quat2rotm(body_quat)
        for f_id in feet_ids:
            foot_pos = np.array(ps_list[str(f_id)][i])
            res += (body_rot.T @(foot_pos-body_pos)).tolist() # foot pos in base frame
        res += x[POS_SIZE + ROT_SIZE + JOINT_POS_SIZE:].tolist() # body vel joint vel
        max_forward_vel = max(max_forward_vel, x[POS_SIZE + ROT_SIZE + JOINT_POS_SIZE])
        for f_id in feet_ids:
            res += [0, 0, 0] # foot vel
        ref_trajectory["Frames"].append(res)
    with open(ref_trajectory_file, 'w') as outfile:
        json.dump(ref_trajectory, outfile)
    print("Max forward velocity: ", max_forward_vel)
# quaternion to rotation matrix function
def quat2rotm(q):
    """
    Convert quaternion to rotation matrix
    """
    x, y, z, w = q
    R = np.array([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
                  [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
                  [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]])
    return R

def generate_random_gait_phases(kwarg_dict):
    """
    Generate random gait phases
    gait_dist: distribution of gait types (bounding, jumping and trotting)
    """
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

    trotting_step_height_range = kwarg_dict['trotting_step_height_range']
    trotting_step_length_range = kwarg_dict['trotting_step_length_range']
    trotting_time_step_range = kwarg_dict['trotting_time_step_range']
    trotting_step_knots_range = kwarg_dict['trotting_step_knots_range']
    trotting_support_knots_range = kwarg_dict['trotting_support_knots_range']

    gait_distribution = kwarg_dict['gait_distribution']
    if bounding_time_step_range[0]< bounding_time_step_range[1]:
        bounding_time_step = np.random.uniform(bounding_time_step_range[0], bounding_time_step_range[1])
    else:
        bounding_time_step = bounding_time_step_range[0]
    if bounding_step_height_range[0]< bounding_step_height_range[1]:
        bounding_step_height = np.random.uniform(bounding_step_height_range[0], bounding_step_height_range[1])
    else:
        bounding_step_height = bounding_step_height_range[0]
    if bounding_step_length_range[0]< bounding_step_length_range[1]:
        bounding_step_length = np.random.uniform(bounding_step_length_range[0], bounding_step_length_range[1])
    else:
        bounding_step_length = bounding_step_length_range[0]
    if bounding_step_knots_range[0]< bounding_step_knots_range[1]:
        bounding_step_knots = np.random.randint(bounding_step_knots_range[0], bounding_step_knots_range[1])
    else:
        bounding_step_knots = bounding_step_knots_range[0]
    if bounding_support_knots_range[0]< bounding_support_knots_range[1]:
        bounding_support_knots = np.random.randint(bounding_support_knots_range[0], bounding_support_knots_range[1])
    else:
        bounding_support_knots = bounding_support_knots_range[0]
    if jumping_time_step_range[0]< jumping_time_step_range[1]:
        jumping_time_step = np.random.uniform(jumping_time_step_range[0], jumping_time_step_range[1])
    else:
        jumping_time_step = jumping_time_step_range[0]
    if jumping_step_height_range[0]< jumping_step_height_range[1]:
        jumping_step_height = np.random.uniform(jumping_step_height_range[0], jumping_step_height_range[1])
    else:
        jumping_step_height = jumping_step_height_range[0]
    if jumping_step_length_range[0]< jumping_step_length_range[1]:
        jumping_step_length = np.random.uniform(jumping_step_length_range[0], jumping_step_length_range[1])
    else:
        jumping_step_length = jumping_step_length_range[0]
    if jumping_ground_knots_range[0]< jumping_ground_knots_range[1]:
        jumping_ground_knots = np.random.randint(jumping_ground_knots_range[0], jumping_ground_knots_range[1])
    else:
        jumping_ground_knots = jumping_ground_knots_range[0]    
    if jumping_flying_knots_range[0]< jumping_flying_knots_range[1]:
        jumping_flying_knots = np.random.randint(jumping_flying_knots_range[0], jumping_flying_knots_range[1])
    else:
        jumping_flying_knots = jumping_flying_knots_range[0]
    if trotting_time_step_range[0]< trotting_time_step_range[1]:
        trotting_time_step = np.random.uniform(trotting_time_step_range[0], trotting_time_step_range[1])
    else:
        trotting_time_step = trotting_time_step_range[0]
    if trotting_step_height_range[0]< trotting_step_height_range[1]:
        trotting_step_height = np.random.uniform(trotting_step_height_range[0], trotting_step_height_range[1])
    else:
        trotting_step_height = trotting_step_height_range[0]
    if trotting_step_length_range[0]< trotting_step_length_range[1]:
        trotting_step_length = np.random.uniform(trotting_step_length_range[0], trotting_step_length_range[1])
    else:
        trotting_step_length = trotting_step_length_range[0]
    if trotting_step_knots_range[0]< trotting_step_knots_range[1]:
        trotting_step_knots = np.random.randint(trotting_step_knots_range[0], trotting_step_knots_range[1])
    else:
        trotting_step_knots = trotting_step_knots_range[0]
    if trotting_support_knots_range[0]< trotting_support_knots_range[1]:
        trotting_support_knots = np.random.randint(trotting_support_knots_range[0], trotting_support_knots_range[1])
    else:
        trotting_support_knots = trotting_support_knots_range[0]
    
    gait_type = np.random.choice(['bounding', 'jumping', 'trotting'], 1, p=gait_distribution)
    if gait_type == 'bounding':
        GAITPHASE = {
                    'bounding': {
                        'stepLength': bounding_step_length,
                        'stepHeight': bounding_step_height,
                        'timeStep': bounding_time_step,
                        'stepKnots': bounding_step_knots,
                        'supportKnots': bounding_support_knots
                    }
                }
    elif gait_type == 'jumping': 
        GAITPHASE = {
                    'jumping': {
                                'jumpHeight': jumping_step_height,
                                'jumpLength': [jumping_step_length, 0.0, 0.],
                                'timeStep': jumping_time_step,
                                'groundKnots': jumping_ground_knots,
                                'flyingKnots': jumping_flying_knots
                    }
                    }
    elif gait_type == 'trotting':
        GAITPHASE = {
                    'trotting': {
                                'stepLength': trotting_step_length,
                                'stepHeight': trotting_step_height,
                                'timeStep': trotting_time_step,
                                'stepKnots': trotting_step_knots,
                                'supportKnots': trotting_support_knots
                    }
                    }
    else:
        raise ValueError('Gait type not supported')
    yield GAITPHASE  
# generate random gait trajectories
def generate_random_gait_trajectories(kwarg_dict):
    """
    Generate random gait trajectories
    """
    fs_list = []
    dts_list = []
    ps_dict = {str(f_id): [] for f_id in feet_ids}
    xs_list = []
    us_list = []
    solve_status_list = []
    bound_solve_status_list = []
    jump_solve_status_list = []
    trot_solve_status_list = []

    N_trajectories = kwarg_dict['N_trajectories']
    ref_trajectory_file = kwarg_dict['ref_trajectory_file']
    x0 = kwarg_dict['x0']
    t_count = 0
    i = 0
    solver = [None] * N_trajectories
    while t_count < N_trajectories:
        phase = next(generate_random_gait_phases(kwarg_dict))
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
            elif key == 'trotting':
                # Creating a trotting problem
                solver[t_count] = crocoddyl.SolverFDDP(
                    gait.createTrottingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                                value['stepKnots'], value['supportKnots']))
            else:
                raise ValueError("Unknown gait phase")
                    # Added the callback functions
            display = crocoddyl.GepettoDisplay(robot, 50, 10, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
            solver[t_count].setCallbacks([crocoddyl.CallbackDisplay(display)])


            # Solving the problem with the DDP solver
            xs = [x0] * (solver[t_count].problem.T + 1)
            us = solver[t_count].problem.quasiStatic([x0] * solver[t_count].problem.T)
            status = solver[t_count].solve(xs, us, 100, False)
            if key == 'bounding':
                bound_solve_status_list.append(status)
            elif key == 'jumping':
                jump_solve_status_list.append(status)
            elif key == 'trotting':
                trot_solve_status_list.append(status)
            else:
                raise ValueError("Unknown gait phase")
            if status:
                x0 = solver[t_count].xs[-1]
 
            solve_status_list.append(status)

        display = crocoddyl.GepettoDisplay(robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        if not solve_status_list[t_count]:
                continue
        else:
            fs = display.getForceTrajectoryFromSolver(solver[t_count])
            ps = display.getFrameTrajectoryFromSolver(solver[t_count])

            models = solver[t_count].problem.runningModels.tolist() + [solver[t_count].problem.terminalModel]
            dts = [m.dt if hasattr(m, "differential") else 0. for m in models]
            xs_list = xs_list +  [x for x in solver[t_count].xs]
            fs_list = fs_list + fs
            dts_list = dts_list + dts
            for key, value in ps.items():
                ps_dict[key] +=  value
            t_count += 1
        i+=1
        print(f"iter {i}  bounding: {sum(bound_solve_status_list)}/{len(bound_solve_status_list)} \
                        jumping: {sum(jump_solve_status_list)}/{len(jump_solve_status_list)} \
                        trotting: {sum(trot_solve_status_list)}/{len(trot_solve_status_list)} \
                        total: {sum(solve_status_list)}/{len(solve_status_list)}")


    print(f"number of success solve: {sum(solve_status_list)}/{len(solve_status_list)}")
    print(f"number of success bound solve: {sum(bound_solve_status_list)}/{len(bound_solve_status_list)}")
    print(f"number of success jump solve: {sum(jump_solve_status_list)}/{len(jump_solve_status_list)}")
    print(f"number of success trot solve: {sum(trot_solve_status_list)}/{len(trot_solve_status_list)}")

    # print(f"ps_dict: {ps_dict.keys()}")
    # print(f"lfFootId: {lfFootId}, rfFootId: {rfFootId}, lhFootId: {lhFootId}, rhFootId: {rhFootId}")
    # #IsaacGym order [FL, FR, RL, RR].
    N_jumps = sum(jump_solve_status_list)
    N_bounds = sum(bound_solve_status_list)
    N_trot = sum(solve_status_list) - N_jumps - N_bounds
    # display = crocoddyl.GepettoDisplay(robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    # display.display(xs_list, fs=fs_list, ps=ps_dict, dts=dts_list, factor=1.)
    ref_trajectory_file = ref_trajectory_file + f"_Ntraj_{N_trajectories}_{N_jumps}_jumps_{N_bounds}_bounds_{N_trot}_trots.txt"
    ref_trajectory_file = os.path.join(kwargs['issacgym_dataset_path'], ref_trajectory_file)
    prompt = f"Save the trajectory to {ref_trajectory_file}? [y/n]"
    ans = input(prompt)
    if ans == 'y':
        save_ref_trajectory(xs_list, ps_dict, feet_ids, ref_trajectory_file)
        print(f"Saved the trajectory to {ref_trajectory_file}")
        #pickle solver obeject and save it to file
    data = {}
    data['xs_list'] = xs_list
    data['fs_list'] = fs_list
    data['dts_list'] = dts_list
    data['ps_dict'] = ps_dict
    data['feet_ids'] = feet_ids
    data['us_list'] = us_list
    return data

if __name__ == '__main__':

    WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
    WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

    # Loading the robot model
    # robot = example_robot_data.load('robot')
    robot = example_robot_data.load('a1')

    # Defining the initial state of the robot
    q0 = robot.model.referenceConfigurations['standing'].copy()
    # print(f"q0: {q0}")
    v0 = pinocchio.utils.zero(robot.model.nv)
    x0 = np.concatenate([q0, v0])

    # Setting up the 3d walking problem
    # lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
    lfFoot, rfFoot, lhFoot, rhFoot = 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
    gait = SimpleQuadrupedalGaitProblem(robot.model, lfFoot, rfFoot, lhFoot, rhFoot)
    cameraTF = [15., 10.68, 3.5, 0.2, 0.62, 0.72, 0.22]
    lfFootId = robot.model.getFrameId(lfFoot)
    rfFootId = robot.model.getFrameId(rfFoot)
    lhFootId = robot.model.getFrameId(lhFoot)
    rhFootId = robot.model.getFrameId(rhFoot)
    feet_ids = [lfFootId, rfFootId, lhFootId, rhFootId]
    kwargs = {}
    kwargs['bounding_step_height_range'] = [0.05, 0.05]
    kwargs['bounding_step_length_range'] = [0.12, 0.12] 
    kwargs['bounding_time_step_range'] = [1e-2, 1e-2] 
    kwargs['bounding_step_knots_range'] = [15, 15]
    kwargs['bounding_support_knots_range'] = [5, 5]
    kwargs['jumping_step_height_range'] = [0.4, 1.2] 
    kwargs['jumping_step_length_range'] = [0.4, 0.6] 
    kwargs['jumping_time_step_range'] = [1e-2, 1e-2] 
    kwargs['jumping_ground_knots_range'] = [5, 15] 
    kwargs['jumping_flying_knots_range'] = [5, 25] 
    kwargs['trotting_step_height_range'] = [0.05, 0.05] 
    kwargs['trotting_step_length_range'] = [0.05, 0.05] 
    kwargs['trotting_time_step_range'] = [1e-2, 1e-2]
    kwargs['trotting_step_knots_range'] = [15, 16] 
    kwargs['trotting_support_knots_range'] = [4, 6] 

    kwargs['feet_ids'] = feet_ids
    kwargs['cameraTF'] = cameraTF
    kwargs['x0'] = x0
    kwargs['N_trajectories'] = 10
    kwargs['ref_trajectory_file'] = 'ref_trajectory'

    kwargs['gait_distribution'] = [1., 0.0, 0.0] # [bounding, jumping, trotting]
    kwargs['issacgym_dataset_path'] = '/home/dan/repos/AMP_for_hardware/datasets/to_bounding_large'

    # check if folder exists and create it if not
    if not os.path.exists(kwargs['issacgym_dataset_path']):
        os.makedirs(kwargs['issacgym_dataset_path'])
    # check if user wants just playback of reference trajectory
    ref_trajectory_file = os.path.join(kwargs['issacgym_dataset_path'], kwargs['ref_trajectory_file'] + ".pkl")
    if 'playback' in sys.argv:
        try:
            traj_data = pickle.load(open(ref_trajectory_file, 'rb'))
        except:
            print(f"Could not load file {ref_trajectory_file} please generate it first")
            exit()
        try:
            playback_trajectory(traj_data, repeat=True)
        except:
            print("Exiting playback")
    else:
        print("Generating new reference trajectory")
        traj_data = generate_random_gait_trajectories(kwargs)
        playback_trajectory(traj_data, playback_speed=1.0, repeat=False)    
        # check if user wants to save the reference trajectory
        prompt = f"Save the trajectory to {ref_trajectory_file}? [y/n]"
        ans = input(prompt)
        if ans == 'y':
            # save dictionary to file
            pickle.dump(traj_data, open(ref_trajectory_file, "wb"))
