from __future__ import print_function

import os
import sys

import crocoddyl
from crocoddyl.utils.biped import plotSolution
import numpy as np
import example_robot_data
import pinocchio
import pickle
import yaml

def playback_trajectory(data, display, playback_speed=1.0, repeat=False):
    xs = data["xs"]
    ps  = data["ps"]
    dts = data["dts"]
    fs = data["fs"]
    # repeat display till user control + c
    print("Press ctrl+c to exit")
    while True:
        # display = crocoddyl.GepettoDisplay(robot,frameNames=[rightFoot, leftFoot])
        display.display(xs, fs=fs, ps=ps, dts=dts, factor=playback_speed)
        if not repeat:
            break
        # continously check control+c
        try:
            pass # do nothing
        except KeyboardInterrupt:
            break
def generate_scoccer_kick_trajectory(config_file):
    # Load robot


    # if config_file is not None:
    with open('soccer_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        robot_info = config["robot_info"]
        traj_params = config["traj_params"]
        debug_info = config["debug_info"]

        DT = traj_params["DT"]
        T = traj_params["T"]
        T_standup = traj_params["T_standup"]
        T_windup = traj_params["T_windup"]
        T_kick = traj_params["T_kick"]
        T_recovery = traj_params["T_recovery"]
        T_balance = traj_params["T_balance"]
        ball_radius = traj_params["ball_radius"]
        target = np.array(traj_params["target"])
        kickFootWindupPos = np.array(traj_params["kickFootWindupPos"])
        kickFootLandingPos = np.array(traj_params["kickFootLandingPos"])
        # display = debug_info["display"]
        display_step = debug_info["display_step"]
        plot = debug_info["plot"]
        ref_config = robot_info["ref_config"]
        rightFoot = robot_info["rightFoot"]
        leftFoot = robot_info["leftFoot"]
        kickFoot = robot_info["kickFoot"]

        robot = example_robot_data.load(robot_info["robot"])
        
        rmodel = robot.model
        lims = rmodel.effortLimit
        rmodel.effortLimit = lims

        # Create data structures
        rdata = rmodel.createData()
        state = crocoddyl.StateMultibody(rmodel)
        actuation = crocoddyl.ActuationModelFloatingBase(state)


        display = crocoddyl.GepettoDisplay(robot, frameNames=[rightFoot, leftFoot])
        display.robot.viewer.gui.addSphere('world/point', ball_radius, [0., 0., 0., 0.5])  # radius = .1, RGBA=1001
        display.robot.viewer.gui.applyConfiguration('world/point', target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion


    # Initialize reference state, target and reference CoM position
    kickFootId = rmodel.getFrameId(kickFoot)
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    q0 = rmodel.referenceConfigurations[ref_config]
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
    pinocchio.forwardKinematics(rmodel, rdata, q0)
    pinocchio.updateFramePlacements(rmodel, rdata)
    rfPos0 = rdata.oMf[rightFootId].translation
    lfPos0 = rdata.oMf[leftFootId].translation

    comRef = (rfPos0 + lfPos0) / 2
    comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

    # Create two contact models used along the motion
    contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
    supportContactModelLeft = crocoddyl.ContactModel6D(state, leftFootId, pinocchio.SE3.Identity(), actuation.nu,
                                                    np.array([0, 40]))
    supportContactModelRight = crocoddyl.ContactModel6D(state, rightFootId, pinocchio.SE3.Identity(), actuation.nu,
                                                        np.array([0, 40]))
    contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelLeft)
    contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
    contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

    # Cost for self-collision
    maxfloat = sys.float_info.max
    xlb = np.concatenate([
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv)
    ])
    xub = np.concatenate([
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv)
    ])
    bounds = crocoddyl.ActivationBounds(xlb, xub, 1.)
    xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
    limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

    # Cost for state and control
    xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)**2)
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv)**2)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # Cost for target reaching: hand and foot
    kickFootTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, kickFootId, pinocchio.SE3(np.eye(3), target),
                                                                actuation.nu)
    kickFootTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1] * 3 + [10.] * 3)**2)
    kickFootTrackingCost = crocoddyl.CostModelResidual(state, kickFootTrackingActivation, kickFootTrackingResidual)

    kickFootWindupTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, kickFootId,
                                                                pinocchio.SE3(np.eye(3), kickFootWindupPos),
                                                                actuation.nu)
    kickFootWindupTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0.1] + [10.] * 3)**2)
    kickFootWindupTrackingCost = crocoddyl.CostModelResidual(state, kickFootWindupTrackingActivation, kickFootWindupTrackingResidual)

    kickFootLandingTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, kickFootId,
                                                                pinocchio.SE3(np.eye(3), kickFootLandingPos),
                                                                actuation.nu)
    kickFootLandingTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0.1] + [10.] * 2 + [100])**2)
    kickFootLandingTrackingCost = crocoddyl.CostModelResidual(state, kickFootLandingTrackingActivation, kickFootLandingTrackingResidual)

    # Cost for CoM reference
    comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
    comTrack = crocoddyl.CostModelResidual(state, comResidual)

    # Create cost model per each action model. We divide the motion in 3 phases plus its terminal model
    runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu) # standing phase
    runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu) # windup phase
    runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu) # kick phase
    runningCostModel4 = crocoddyl.CostModelSum(state, actuation.nu) # recovery phase
    runningCostModel5 = crocoddyl.CostModelSum(state, actuation.nu) # balance phase
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu) # terminal phase


    # Then let's added the running and terminal cost functions
    runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel1.addCost("limitCost", limitCost, 1e3)

    runningCostModel2.addCost("kickFootWindupPose", kickFootWindupTrackingCost, 1e3)
    runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel2.addCost("limitCost", limitCost, 1e3)

    runningCostModel3.addCost("kickFootKickPose", kickFootTrackingCost, 1e3)
    runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel3.addCost("limitCost", limitCost, 1e3)

    runningCostModel4.addCost("kickFootLandingPose", kickFootLandingTrackingCost, 5e5)
    runningCostModel4.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel4.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel4.addCost("limitCost", limitCost, 1e3)

    runningCostModel5.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel5.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel5.addCost("limitCost", limitCost, 1e3)

    terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
    terminalCostModel.addCost("limitCost", limitCost, 1e3)

    # Create the action model
    dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet,
                                                                        runningCostModel1)
    dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                        runningCostModel2)
    dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                        runningCostModel3)
    dmodelRunning4 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                        runningCostModel4)
    dmodelRunning5 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet,
                                                                        runningCostModel5)
    dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet,
                                                                        terminalCostModel)

    runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
    runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
    runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
    runningModel4 = crocoddyl.IntegratedActionModelEuler(dmodelRunning4, DT)
    runningModel5 = crocoddyl.IntegratedActionModelEuler(dmodelRunning5, DT)
    terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

    # Problem definition
    x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
    problem = crocoddyl.ShootingProblem(x0, [runningModel1] * T_standup + 
                                            [runningModel2] * T_windup + 
                                            [runningModel3] * T_kick + 
                                            [runningModel4] * T_recovery + 
                                            [runningModel5] * T_balance, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverBoxFDDP(problem)
    if display_step and plot:
        solver.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackDisplay(display)])
    elif display_step:
        solver.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackDisplay(display)])
    elif plot:
        solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving it with the DDP algorithm
    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.th_stop = 1e-7
    solver.solve(xs, us, 500, False, 1e-9)
    models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
    dts = [m.dt if hasattr(m, "differential") else 0. for m in models]
    fs = display.getForceTrajectoryFromSolver(solver)
    ps = display.getFrameTrajectoryFromSolver(solver)
    xs = [x for x in solver.xs]
    us = [u for u in solver.us]
    ts = np.cumsum(dts)
    result = {}
    result['xs'] = xs
    result['us'] = us
    result['ts'] = ts
    result['fs'] = fs
    result['ps'] = ps
    result['dts'] = dts
    result['target'] = target

    playback_trajectory(result, display, playback_speed=1.0, repeat=False)  

    soccer_res_dir = os.path.join(os.path.dirname(__file__), debug_info['save_dir_path'])
    if not os.path.exists(soccer_res_dir):
        os.makedirs(soccer_res_dir)
    ref_trajectory_file = os.path.join(soccer_res_dir, debug_info['experiment_name'] + '.pkl')
    # check if user wants to save the reference trajectory
    prompt = f"Save the trajectory to {ref_trajectory_file}? [y/n]"
    ans = input(prompt)
    if ans == 'y':
    # save dictionary to file
        pickle.dump(result, open(ref_trajectory_file, "wb"))

if __name__ == '__main__':

    if 'playback' in sys.argv:
        try:
            with open('soccer_config.yaml') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
                debug_info = config['debug_info']
                soccer_res_dir = os.path.join(os.path.dirname(__file__), debug_info['save_dir_path'])
                ref_trajectory_file = os.path.join(soccer_res_dir, debug_info['experiment_name'] + '.pkl')
                traj_data = pickle.load(open(ref_trajectory_file, 'rb'))
                robot = example_robot_data.load(config['robot_info']['robot'])
                rightFoot = config['robot_info']['rightFoot']
                leftFoot = config['robot_info']['leftFoot']
                display = crocoddyl.GepettoDisplay(robot,frameNames=[rightFoot, leftFoot])
        except:
            print(f"Could not load file {ref_trajectory_file} please generate it first")
            exit()
        try:
            print(f"Playing back reference trajectory {ref_trajectory_file}")
            playback_trajectory(traj_data, display, repeat=True)
        except:
            print("Exiting playback")
    else:
        print("Generating new reference trajectory")
        generate_scoccer_kick_trajectory(config_file='soccer_config.yaml')
