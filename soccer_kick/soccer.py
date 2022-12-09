import crocoddyl
import pinocchio
import numpy as np
from simple_com_foot_ref import simpleRefGenerator

class SoccerProblem:
    """ Build simple bipedal soccer kicking problems.
    """

    def __init__(self, rmodel, rightFoot, leftFoot, integrator='euler', control='zero'):
        """ Construct biped-soccer problem.
        :param rmodel: robot model
        :param rightFoot: name of the right foot
        :param leftFoot: name of the left foot
        :param integrator: type of the integrator (options are: 'euler', and 'rk4')
        :param control: type of control parametrization (options are: 'zero', 'one', and 'rk4')
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        self.integrator = integrator
        if control == 'one':
            self.control = crocoddyl.ControlParametrizationModelPolyOne(self.actuation.nu)
        elif control == 'rk4':
            self.control = crocoddyl.ControlParametrizationModelPolyTwoRK(self.actuation.nu, crocoddyl.RKType.four)
        elif control == 'rk3':
            self.control = crocoddyl.ControlParametrizationModelPolyTwoRK(self.actuation.nu, crocoddyl.RKType.three)
        else:
            self.control = crocoddyl.ControlParametrizationModelPolyZero(self.actuation.nu)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)
    def createKickProblem(self, x0, targetLocation, timeStep, kickKnots, recoveryKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        rfFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])

        self.rWeight = 1e1
        loco3dModel = []
        kickPhase = []
        moveCOMForwardPhase = []
        for k in range(kickKnots):
            # Defining the CoM task
            comTask = comRef + (targetLocation - comRef) * float(k + 1) / kickKnots
            comTask[2] = comRef[2]
            moveCOMForwardPhase+=[self.createSwingFootModel(timeStep, [self.lfId], comTask, [])]

        for k in range(kickKnots):
            # Computing the reference location for the kicking foot
            tref = targetLocation * float(k + 1) / kickKnots
            tref[2] = 0.05
            # Computing the reference location for the CoM
            # comTask = comRef + (targetLocation - comRef) * float(k + 1) / kickKnots
            comTask = targetLocation + 0.1
            
            comTask[2] = comRef[2]
            # Creating the locomotion phase
            rfSwingTask = [[self.rfId, pinocchio.SE3(np.eye(3), tref + rfFootPos0)]]
            kickPhase += [self.createSwingFootModel(timeStep, [self.lfId], comTask, rfSwingTask)]        
    
        # Creating the recovery phase
        recoveryPhase = [ self.createSwingFootModel(timeStep, 
                                                      [self.lfId, self.rfId], 
                                                      [], 
                                                      []) 
                                                      for k in range(recoveryKnots)]

        loco3dModel += moveCOMForwardPhase
        loco3dModel += kickPhase
        loco3dModel += recoveryPhase

        costs = loco3dModel[-1].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep
        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem
    def createSoccerKickProblem(self, x0, jumpingHeight, landingLocation, targetLocation, timeStep, groundKnots, jumpKnots, kickKnots, recoveryKnots):
        self.t_take_off  = timeStep*groundKnots
        self.t_landing      = self.t_take_off + timeStep*jumpKnots
        self.t_kick = self.t_landing + timeStep*kickKnots
        self.t_recovery = self.t_kick + timeStep*recoveryKnots


        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        rfFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])
        ref_gen_func = simpleRefGenerator(self.t_take_off, self.t_landing, self.t_kick, jumpingHeight, landingLocation, targetLocation, comRef, rfFootPos0, lfFootPos0)
        t = self.t_landing    

        self.rWeight = 1e1
        loco3dModel = []
        kickPhase = []
        recoveryPhase = []
        for k in range(kickKnots):
            # Defining the CoM task
            com_ref, lf_ref, rf_ref, phase, cs = ref_gen_func(t)
            rfSwingTask = [[self.rfId, pinocchio.SE3(np.eye(3), rf_ref)]]
            kickPhase += [self.createSwingFootModel(timeStep, [self.lfId], com_ref, rfSwingTask)]        
            t+=timeStep
        for k in range(recoveryKnots):
            # Defining the CoM task
            com_ref, lf_ref, rf_ref, phase, cs = ref_gen_func(t)
            kickPhase += [self.createSwingFootModel(timeStep, [self.lfId, self.rfId], com_ref, [])]        
            t+=timeStep
        loco3dModel += kickPhase
        loco3dModel += recoveryPhase

        costs = loco3dModel[-1].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep
        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem
    def createSoccerTakeOffJumpProblem(self, x0, jumpingHeight, landingLocation, targetLocation, timeStep, groundKnots, jumpKnots, kickKnots, recoveryKnots):

        self.t_take_off  = timeStep*groundKnots
        self.t_landing      = self.t_take_off + timeStep*jumpKnots
        self.t_kick = self.t_landing + timeStep*kickKnots
        self.t_recovery = self.t_kick + timeStep*recoveryKnots


        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        rfFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])
        ref_gen_func = simpleRefGenerator(self.t_take_off, self.t_landing, self.t_kick, jumpingHeight, landingLocation, targetLocation, comRef, rfFootPos0, lfFootPos0)
        t = 0.0    
        loco3dModel = []
        takeOff = [self.createSwingFootModel(
            timeStep,
            [self.lfId, self.rfId],
        ) for k in range(groundKnots)]

        jumpPhase = []
        t += timeStep*groundKnots
        for k in range(jumpKnots):
            
            stepup_foot_task = []
            com_ref, lf_ref, rf_ref, phase, cs = ref_gen_func(t)
            stepup_foot_task += [[self.lfId, pinocchio.SE3(np.eye(3), lf_ref)]]
            stepup_foot_task += [[self.rfId, pinocchio.SE3(np.eye(3), rf_ref)]]
            if k<jumpKnots//2:
                jumpPhase += [
                    self.createSwingFootModel(
                        timeStep, [],
                        com_ref,
                        stepup_foot_task)
                ]
            else:
                jumpPhase += [
                    self.createSwingFootModel(
                        timeStep, [],
                        [],
                        stepup_foot_task)
                ]
            t += timeStep
        com_ref, lf_ref, rf_ref, phase, cs = ref_gen_func(t)
        footTask = [[self.lfId, pinocchio.SE3(np.eye(3), lf_ref)],
                    [self.rfId, pinocchio.SE3(np.eye(3), rf_ref)]]
                    
        landingPhase = [self.createFootSwitchModel([self.lfId, self.rfId], footTask, False)]

        loco3dModel += takeOff 
        loco3dModel += jumpPhase 
        loco3dModel += landingPhase

        # Rescaling the terminal weights
        # print("costs: ", loco3dModel[0].differential.costs.costs.todict())
        costs = loco3dModel[0].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem

    def createTakeOffProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots, final=False):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])

        self.rWeight = 1e1
        loco3dModel = []
        takeOff = [self.createSwingFootModel(
            timeStep,
            [self.lfId, self.rfId],
        ) for k in range(groundKnots)]

        flyingUpPhase = []
        for k in range(flyingKnots):
            stepup_foot_task = []
            dpl = np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]) * (k + 1) / flyingKnots
            dpr = np.array([0, 0, jumpHeight]) * (k + 1) / flyingKnots
            
            stepup_foot_task += [[self.lfId, pinocchio.SE3(np.eye(3), dpl + lfFootPos0)]]
            stepup_foot_task += [[self.rfId, pinocchio.SE3(np.eye(3), dpr + rfFootPos0)]]
        
    
            flyingUpPhase += [
                self.createSwingFootModel(
                    timeStep, [],
                    np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]) * (k + 1) / flyingKnots + comRef,
                    stepup_foot_task) #com task
                # for k in range(flyingKnots)
            ]
        flyingDownPhase = []
        for k in range(flyingKnots):
            stepdown_foot_task = []
            dpl = np.zeros(3)
            dpl[0] = 2*jumpLength[0] * (k + 1) / flyingKnots
            dpl[1] = 2*jumpLength[1] * (k + 1) / flyingKnots
            dpl[2] = jumpLength[2] * (1-(k + 1) / flyingKnots)
            dpr = np.array([0, 0, jumpHeight]) * (1-(k + 1) / flyingKnots)
            
            stepdown_foot_task += [[self.lfId, pinocchio.SE3(np.eye(3), dpl + lfFootPos0)]]
            stepdown_foot_task += [[self.rfId, pinocchio.SE3(np.eye(3), dpr + rfFootPos0)]]
            flyingDownPhase += [self.createSwingFootModel(timeStep, [], swingFootTask=stepdown_foot_task)]
        # timeStep, supportFootIds, comTask=None, swingFootTask=None
        f0 = jumpLength
        footTask = [[self.lfId, pinocchio.SE3(np.eye(3), 2*(np.array(jumpLength)+lfFootPos0))],
                    [self.rfId, pinocchio.SE3(np.eye(3), rfFootPos0)]]
                    
        landingPhase = [self.createFootSwitchModel([self.lfId, self.rfId], footTask, False)]
        f0[2] = df
        landed = [
            self.createSwingFootModel(timeStep, [self.lfId, self.rfId], comTask=comRef + f0)
            for k in range(groundKnots)
        ]
        loco3dModel += takeOff 
        loco3dModel +=flyingUpPhase 
        loco3dModel +=flyingDownPhase
        if final:
            loco3dModel += landingPhase
            loco3dModel += landed
            print("full model including impact!!")

        # Rescaling the terminal weights
        costs = loco3dModel[-1].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem
    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots, final=False):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])

        self.rWeight = 1e1
        loco3dModel = []
        takeOff = [self.createSwingFootModel(
            timeStep,
            [self.lfId, self.rfId],
        ) for k in range(groundKnots)]
        flyingUpPhase = [
            self.createSwingFootModel(
                timeStep, [],
                np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]) * (k + 1) / flyingKnots + comRef)
            for k in range(flyingKnots)
        ]
        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])]

        f0 = jumpLength
        footTask = [[self.lfId, pinocchio.SE3(np.eye(3), self.lfId + f0)],
                    [self.rfId, pinocchio.SE3(np.eye(3), self.rfId + f0)]]
        landingPhase = [self.createFootSwitchModel([self.lfId, self.rfId], footTask, False)]
        f0[2] = df
        if final is True:
            self.rWeight = 1e4
        landed = [
            self.createSwingFootModel(timeStep, [self.lfId], comTask=comRef + f0)
            for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        # Rescaling the terminal weights
        costs = loco3dModel[-1].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem

    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight, timeStep, numKnots, supportFootIds,
                             swingFootIds):
        """ Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs

        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length. The swing task
                # is decomposed on two phases: swing-up and swing-down. We decide
                # deliveratively to allocated the same number of nodes (i.e. phKnots)
                # in each phase. With this, we define a proper z-component for the
                # swing-leg motion.
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots])
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight])
                else:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp

                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]

            comTask = np.array([stepLength * (k + 1) / numKnots, 0., 0.]) * comPercentage + comPos0
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]

        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask, False)

        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0., 0.]
        for p in feetPos0:
            p += [stepLength, 0., 0.]
        return footSwingModel + [footSwitchModel]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = \
                crocoddyl.ContactModel6D(self.state, i,
                                         pinocchio.SE3.Identity(),
                                         nu, np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 5e6)
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(self.state, i, cone, nu)
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            wrenchCone = crocoddyl.CostModelResidual(self.state, wrenchActivation, wrenchResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1)

        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                footTrack = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6) #1e6

        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        if self.integrator == 'euler':
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, timeStep)
        elif self.integrator == 'rk4':
            model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.four, timeStep)
        elif self.integrator == 'rk3':
            model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.three, timeStep)
        elif self.integrator == 'rk2':
            model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.two, timeStep)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, timeStep)
        return model

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=True):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """

        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(self.state, i, pinocchio.SE3.Identity(), nu,
                                                           np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(self.state, i, cone, nu)
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            wrenchCone = crocoddyl.CostModelResidual(self.state, wrenchActivation, wrenchResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1)

        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state, i[0], pinocchio.Motion.Zero(),
                                                                             pinocchio.LOCAL, nu)
                footTrack = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
                impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_impulseVel", impulseFootVelCost, 1e6)

        stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        if self.integrator == 'euler':
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        elif self.integrator == 'rk4':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.four, 0.)
        elif self.integrator == 'rk3':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.three, 0.)
        elif self.integrator == 'rk2':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        return model

    def createImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(self.state, i)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   0)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8)

        stateWeights = np.array([1.] * 6 + [0.1] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, 0)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        return model


def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    xs, us = [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    if isinstance(solver, list):
        rmodel = solver[0].problem.runningModels[0].state.pinocchio
        for s in solver:
            xs.extend(s.xs[:-1])
            us.extend(s.us)
            if bounds:
                models = s.problem.runningModels.tolist() + [s.problem.terminalModel]
                for m in models:
                    us_lb += [m.u_lb]
                    us_ub += [m.u_ub]
                    xs_lb += [m.state.lb]
                    xs_ub += [m.state.ub]
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        xs, us = solver.xs, solver.us
        if bounds:
            models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
            for m in models:
                us_lb += [m.u_lb]
                us_ub += [m.u_ub]
                xs_lb += [m.state.lb]
                xs_ub += [m.state.ub]

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ['1', '2', '3', '4', '5', '6']
    # left foot
    plt.subplot(2, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 13))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 13))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 12))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 12))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 6))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(0, 6))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(0, 6))]
    plt.ylabel('LF')
    plt.legend()

    # right foot
    plt.subplot(2, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 19))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(13, 19))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(13, 19))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(2, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 18))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 18))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(6, 12))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(6, 12))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()

    plt.figure(figIndex + 1)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[:rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if show:
        plt.show()
