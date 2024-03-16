from enum import Enum
import numpy as np

EE_HEIGHT = 0.193 #0.141
BOARD_HEIGHT = 0

def ros_print(node, msg: str):
    """
    Easy print for ROS nodes
    """
    node.get_logger().info(str(msg))

class Grab:

    class Mode(Enum):
        START = 0
        TO_OBJ = 1
        GRAB = 2
        STOP = 3

    def __init__(self, pos, angle=None):
        self.mode = Grab.Mode.START
        self.pos = np.array(pos)
        self.angle = angle
        self.done = False

    def step(self, t, ready, armed):
        goal = None
        grip = None
        strike = None
        if self.mode == Grab.Mode.START:
            pass
        elif self.mode == Grab.Mode.TO_OBJ:
            goal = [None, None, None]
            goal[0] = self.pos
            if self.angle is None:
                self.angle = np.arctan2(self.pos[1], self.pos[0])
            goal[1] = self.angle
            goal[2] = 0
            grip = False
        elif self.mode == Grab.Mode.GRAB:
            grip = True
        elif self.mode == Grab.Mode.STOP:
            pass
        else:
            raise Exception('Grab: ILLEGAL MODE')

        if self.mode == Grab.Mode.START and ready:
            self.mode = Grab.Mode.TO_OBJ
            self.t0 = t
        elif self.mode == Grab.Mode.TO_OBJ and armed and t - self.t0 > 1.5:
            self.mode = Grab.Mode.GRAB
            self.t0 = t
        elif self.mode == Grab.Mode.GRAB and armed and t - self.t0 > 1.5:
            self.done = True

        return goal, grip, strike
        


class Drop:

    class Mode(Enum):
        START = 0
        TO_POS = 1
        DROP = 2
        STOP = 3

    def __init__(self, pos, angle=None):
        self.mode = Drop.Mode.START
        self.pos = np.array(pos)
        self.done = False
        self.t0 = None
        self.angle = angle

    def step(self, t, ready, armed):
        goal = None
        grip = None
        strike = None
        if self.mode == Drop.Mode.START:
            pass
        elif self.mode == Drop.Mode.TO_POS or self.mode == Drop.Mode.DROP:
            goal = [None, None, None]
            goal[0] = self.pos
            if self.angle is None:
                self.angle = np.arctan2(self.pos[1], self.pos[0])
            goal[1] = self.angle
            goal[2] = 0
            if self.mode == Drop.Mode.TO_POS:
                grip = True
            elif self.mode == Drop.Mode.DROP:
                grip = False
        elif self.mode == Drop.Mode.STOP:
            pass
        else:
            raise Exception('Grab: ILLEGAL MODE')

        if self.mode == Drop.Mode.START and ready:
            self.mode = Drop.Mode.TO_POS
            self.t0 = t
        elif self.mode == Drop.Mode.TO_POS and armed and t - self.t0 > 1.5:
            self.mode = Drop.Mode.DROP
            self.t0 = t
        elif self.mode == Drop.Mode.DROP and armed and t - self.t0 > 1.5:
            self.done = True

        return goal, grip, strike
        

class Strike:

    class Mode(Enum):
        START = 0
        TO_POS = 1
        STRIKE = 2
        STOP = 3

    def __init__(self, pos, angle):
        self.mode = Strike.Mode.START
        self.pos = pos
        self.angle = angle
        self.done = False
        self.t0 = None
        self.numSends = 0

    def step(self, t, ready, armed):
        goal = None
        grip = None
        strike = None
        if self.mode == Strike.Mode.START:
            pass
        elif self.mode == Strike.Mode.TO_POS:
            goal = [None, None, None]
            goal[0] = self.pos
            goal[1] = self.angle
            goal[2] = 0
        elif self.mode == Strike.Mode.STRIKE and self.numSends == 0:
            strike = 255
        elif self.mode == Strike.Mode.STRIKE:
            strike = None
        elif self.mode == Strike.Mode.STOP:
            pass
        else:
            raise Exception('Grab: ILLEGAL MODE')

        if self.mode == Strike.Mode.START and ready:
            self.mode = Strike.Mode.TO_POS
            self.t0 = t
        elif self.mode == Strike.Mode.TO_POS and armed and t - self.t0 > 1.5:
            self.mode = Strike.Mode.STRIKE
            self.t0 = t
        elif self.mode == Strike.Mode.STRIKE and armed and t - self.t0 <= 1.5:
            self.numSends += 1
        elif self.mode == Strike.Mode.STRIKE and armed and t - self.t0 > 1.5:
            self.done = True

        return goal, grip, strike
    

class Move:

    class Mode(Enum):
        START = 0
        TO_POS = 1
        STOP = 2

    def __init__(self, pos, angle, pitch = 0):
        self.mode = Move.Mode.START
        self.pos = pos
        self.angle = angle
        self.done = False
        self.pitch = pitch

    def step(self, t, ready, armed):
        goal = None
        grip = None
        strike = None
        if self.mode == Move.Mode.START:
            pass
        elif self.mode == Move.Mode.TO_POS:
            goal = [None, None, None]
            goal[0] = self.pos
            goal[1] = self.angle
            goal[2] = self.pitch
        elif self.mode == Move.Mode.STOP:
            pass
        else:
            raise Exception('Grab: ILLEGAL MODE')

        if self.mode == Move.Mode.START and ready:
            self.mode = Move.Mode.TO_POS
            self.t0 = t
        elif self.mode == Move.Mode.TO_POS and armed and t - self.t0 > 1.5:
            self.mode = Move.Mode.STOP
            self.t0 = t
        elif self.mode == Move.Mode.STOP and t - self.t0 > 1.5:
            self.done = True


        return goal, grip, strike
    

class Wait:

    class Mode(Enum):
        START = 0
        WAIT = 1
        STOP = 2

    def __init__(self, T):
        self.mode = Wait.Mode.START
        self.T = T
        self.t0 = None
        self.done = False

    def step(self, t, ready, armed):
        goal = None
        grip = None
        strike = None
        if self.mode == Wait.Mode.START:
            pass
        elif self.mode == Wait.Mode.WAIT:
            pass
        elif self.mode == Wait.Mode.STOP:
            pass
        else:
            raise Exception('Grab: ILLEGAL MODE')

        if self.mode == Wait.Mode.START and ready:
            self.mode = Wait.Mode.WAIT
            self.t0 = t
        elif self.mode == Wait.Mode.WAIT and t - self.t0 > self.T:
            self.mode = Wait.Mode.STOP
            self.t0 = t
        elif self.mode == Wait.Mode.STOP and t - self.t0 > 1.5:
            self.done = True

        return goal, grip, strike