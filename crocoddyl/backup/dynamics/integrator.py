import abc
import pinocchio as se3


class Integrator(object):
  """ This abstract class allows us to define different integration rules.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __call__(dynamicModel, dynamicData, x, u, xNext):
    """ Integrate the system dynamics given an user-defined integration scheme.

    :param dynamicModel: dynamics model
    :param dynamicData: dynamics data
    :param x: current state
    :param u: current control
    :param xNext: next state after the integration
    """
    raise NotImplementedError("Not implemented yet.")

class EulerIntegrator(Integrator):
  """ Define a forward Euler integrator.
  """
  @staticmethod
  def __call__(dynamicModel, dynamicData, x, u, xNext):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param dynamicModel: dynamics model
    :param dynamicData: dynamics data
    :param x: current state
    :param u: current control
    :param xNext: next state after the integration
    """
    # Updating the dynamics
    dynamicModel.updateDynamics(dynamicData, x, u)

    xNext[dynamicModel.nq():] = \
      x[dynamicModel.nq():] +\
      dynamicData.dt * dynamicData.a
    xNext[:dynamicModel.nq()] = \
      dynamicModel.integrateConfiguration(
        x[:dynamicModel.nq()],
        dynamicData.dt * xNext[dynamicModel.nq():])