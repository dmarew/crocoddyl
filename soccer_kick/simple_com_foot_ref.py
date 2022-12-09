import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
# define enum for phase of motion
class SoccerPhase:
    TAKE_OFF = 0
    JUMPING = 1
    KICK = 2
    RECOVERY = 3

def simpleRefGenerator(t_take_off, t_landing, t_kick, jump_height, landing_loc, target_loc, com_init, rf_init, lf_init):
    """
    Generate a reference trajectory for the robot to kick a ball at a target location.
    The robot starts from a standing position, jumps to a landing location, and kicks the ball to a target location.
    The robot lands on its left foot, and kicks the ball with its right foot.
    param t_take_off: time when the robot starts jumping
    param t_landing: time when the robot lands
    param t_kick: time when the robot kicks the ball
    param jump_height: height of the jump
    param landing_loc: location of the landing
    param target_loc: location of the target
    param rf_init: initial location of the right foot
    """
    assert t_take_off < t_landing < t_kick, "t_take_off < t_landing < t_kick is required"
    assert jump_height > 0, "jump_height must be positive"
    assert landing_loc[0] > 0, "landing_loc[0] must be positive"
    assert target_loc[0] > landing_loc[0], "target_loc[0] must be larger than landing_loc[0]"
    def ref_traj(t):
        assert t >= 0, "t must be non-negative"
        com_ref = np.zeros(3)
        lf_ref = np.zeros(3)
        rf_ref = np.zeros(3)
        jump_duration = t_landing - t_take_off
        phase = SoccerPhase.TAKE_OFF
        contact_state = [1, 1] # 1: contact, 0: free #left foot, right foot
        if t <= t_take_off: # stand before jumping
            com_ref[0] = com_init[0]
            com_ref[1] = com_init[1]
            com_ref[2] = com_init[2]
            lf_ref[0] = lf_init[0]
            lf_ref[1] = lf_init[1]
            lf_ref[2] = lf_init[2]
            rf_ref[0] = rf_init[0]
            rf_ref[1] = rf_init[1]
            rf_ref[2] = rf_init[2]
        elif t <= t_landing: # jumping phase
            com_ref[0] = com_init[0] + (landing_loc[0]-com_init[0])*(t-t_take_off)/(jump_duration)
            com_ref[1] = com_init[1]
            com_ref[2] = com_init[2] + jump_height*np.sin(np.pi*(t-t_take_off)/(jump_duration))
            lf_ref[0] = lf_init[0] + (landing_loc[0]-lf_init[0])*(t-t_take_off)/(jump_duration)
            lf_ref[1] = lf_init[1] + (landing_loc[1]-lf_init[1])*(t-t_take_off)/(jump_duration)
            lf_ref[2] = lf_init[2] + jump_height*np.sin(np.pi*(t-t_take_off)/(jump_duration))
            rf_ref[0] = rf_init[0] + 0.5*(landing_loc[0]-rf_init[0])*(t-t_take_off)/(jump_duration)
            rf_ref[1] = rf_init[1]
            rf_ref[2] = rf_init[2] + jump_height*np.sin(0.5*np.pi*(t-t_take_off)/(jump_duration))
            phase = SoccerPhase.JUMPING
            contact_state = [0, 0]
        elif t <= t_kick: # land and kick
            t_phase = t-t_landing
            phase_duration = t_kick-t_landing

            com_ref[0] = landing_loc[0] + 0.5*(target_loc[0]-landing_loc[0])*(t_phase)/(phase_duration)
            com_ref[1] = com_init[1] 
            com_ref[2] = com_init[2]
            lf_ref[0] = landing_loc[0]
            lf_ref[1] = landing_loc[1]
            lf_ref[2] = 0.0
            rf_ref[0] = rf_init[0] + 0.5*landing_loc[0] + (target_loc[0]-0.5*landing_loc[0])*(t_phase)/(phase_duration)
            rf_ref[1] = rf_init[1] + (target_loc[1]-rf_init[1])*(t_phase)/(phase_duration)
            rf_ref[2] = (rf_init[2] + jump_height - target_loc[2])*(1-(t_phase)/(phase_duration)) + target_loc[2]
            phase = SoccerPhase.KICK
            contact_state = [1, 0]
        else: # recovery
            lf_ref[0] = landing_loc[0]
            lf_ref[1] = landing_loc[1]
            lf_ref[2] = 0.0
            rf_ref[0] = target_loc[0]
            rf_ref[1] = target_loc[1]
            rf_ref[2] = target_loc[2]
            com_ref[0] = 0.5*(lf_ref[0] + rf_ref[0])
            com_ref[1] = com_init[1]    
            com_ref[2] = com_init[2]
            phase = SoccerPhase.RECOVERY
            contact_state = [1, 1]

        return com_ref, lf_ref, rf_ref, phase, contact_state 
    return ref_traj

def animate_reference(com_ref, lf_ref, rf_ref):
    """Animate the reference trajectory"""
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0.0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1)
    ax.plot(com_ref[0, :], com_ref[1, :], com_ref[2, :], label='com_ref')
    ax.plot(lf_ref[0, :], lf_ref[1, :], lf_ref[2, :], label='lf_ref')
    ax.plot(rf_ref[0, :], rf_ref[1, :], rf_ref[2, :], label='rf_ref')
    line, = ax.plot([], [], [], 'o', lw=5)
    line1, = ax.plot([], [], [], 'o', lw=5)
    line2, = ax.plot([], [], [], 'o', lw=5)
    baseLf, = ax.plot([], [], [], lw=5)
    baseRf, = ax.plot([], [], [], lw=5)

    def animate(i):
        line.set_data(com_ref[0, i], com_ref[1, i])
        line.set_3d_properties(com_ref[2, i])
        line1.set_data(lf_ref[0, i], lf_ref[1, i])
        line1.set_3d_properties(lf_ref[2, i])
        line2.set_data(rf_ref[0, i], rf_ref[1, i])
        line2.set_3d_properties(rf_ref[2, i])
        baseLf.set_data(np.array([lf_ref[0, i], com_ref[0, i]]), np.array([lf_ref[1, i], com_ref[1, i]]))
        baseLf.set_3d_properties(np.array([lf_ref[2, i], com_ref[2, i]]))
        baseRf.set_data(np.array([rf_ref[0, i], com_ref[0, i]]), np.array([rf_ref[1, i], com_ref[1, i]]))
        baseRf.set_3d_properties(np.array([rf_ref[2, i], com_ref[2, i]]))
        return line, line1, line2, baseLf, baseRf

    # create animation
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=20, blit=True)
    return anim


if __name__ == '__main__':
    # define reference trajectory for the foot and COM of the robot
    com_init = np.array([0.0, 0.0, 0.5])
    lf_init = np.array([0.0,  0.2, 0.0])
    rf_init = np.array([0.0, -0.2, 0.0])

    take_off_time = 0.2
    landing_time = 0.55
    kick_time = 0.6
    jump_height = 0.1
    landing_loc = np.array([0.5, 0.15, 0.0])
    target_loc = np.array([0.7, -0.1, 0.11])
    ts = np.linspace(0.1, 0.8, 100)
    com_ref = np.zeros((3, len(ts)))
    lf_ref = np.zeros((3, len(ts)))
    rf_ref = np.zeros((3, len(ts)))
    ref_gen = simpleRefGenerator(take_off_time, 
                                 landing_time, 
                                 kick_time, 
                                 jump_height, 
                                 landing_loc, 
                                 target_loc,
                                 com_init, 
                                 rf_init,
                                 lf_init)
    for i in range(len(ts)):
        com_ref[:, i], lf_ref[:, i], rf_ref[:, i], _, _ = ref_gen(ts[i])
        # com_ref[:, i] += com_init
        # lf_ref[:, i] += lf_init
        # rf_ref[:, i] += rf_init

    anim = animate_reference(com_ref, lf_ref, rf_ref)
    plt.show()




