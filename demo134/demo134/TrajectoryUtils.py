#
#   Constant Helpers
#
#   This is really only included for completeness and to zero the
#   velocity (with the same dimension).
#
def hold(p0):
    # Compute the current (p,v).
    p = p0
    v = 0*p0
    return (p,v)


#
#   Linear Helpers
#
#   Linearly interpolate between an initial/final position of the time T.
#
def interpolate(t, T, p0, pf):
    # Compute the current (p,v).
    p = p0 + (pf-p0)/T * t
    v =    + (pf-p0)/T
    return (p,v)


#
#   Cubic Spline Helpers
#
#   Compute a cubic spline position/velocity as it moves from (p0, v0)
#   to (pf, vf) over time T.
#
def goto(t, T, p0, pf):
    # Compute the current (p,v).
    p = p0 + (pf-p0)   * (3*(t/T)**2 - 2*(t/T)**3)
    v =    + (pf-p0)/T * (6*(t/T)    - 6*(t/T)**2)
    return (p,v)

def spline(t, T, p0, pf, v0, vf):
    # Compute the parameters.
    a = p0
    b = v0
    c =   3*(pf-p0)/T**2 - vf/T    - 2*v0/T
    d = - 2*(pf-p0)/T**3 + vf/T**2 +   v0/T**2
    # Compute the current (p,v).
    p = a + b * t +   c * t**2 +   d * t**3
    v =     b     + 2*c * t    + 3*d * t**2
    return (p,v)


#
#   Quintic Spline Helpers
#
#   Compute a quintic spline position/velocity as it moves from
#   (p0,v0,a0) to (pf,vf,af) over time T.
#

def goto5(t, T, p0, pf):
    # Compute the current (p,v).
    p = p0 + (pf-p0)   * (10*(t/T)**3 - 15*(t/T)**4 +  6*(t/T)**5)
    v =    + (pf-p0)/T * (30*(t/T)**2 - 60*(t/T)**3 + 30*(t/T)**4)
    return (p,v)

def spline5(t, T, p0, pf, v0, vf, a0, af):
    # Compute the parameters.
    a = p0
    b = v0
    c = a0
    d = + 10*(pf-p0)/T**3 - 6*v0/T**2 - 3*a0/T    - 4*vf/T**2 + 0.5*af/T
    e = - 15*(pf-p0)/T**4 + 8*v0/T**3 + 3*a0/T**2 + 7*vf/T**3 -     af/T**2
    f = +  6*(pf-p0)/T**5 - 3*v0/T**4 -   a0/T**3 - 3*vf/T**4 + 0.5*af/T**3
    # Compute the current (p,v).
    p = a + b * t +   c * t**2 +   d * t**3 +   e * t**4 +   f * t**5
    v =     b     + 2*c * t    + 3*d * t**2 + 4*e * t**3 + 5*f * t**4
    return (p,v)