from common import *
import scipy.optimize as opt
import autograd.numpy as np
from autograd import grad,jacobian
from matplotlib import pyplot as plt


def split_state(x):
    n = (x.shape[0]-1)//3
    xk = x[1:n+1]
    yk = x[n+1:n*2+1]
    thetak = x[n*2+1:]
    t_total = x[0]
    return t_total, xk, yk, thetak

def sign(v):
    tol = 1e-5
    k = 1e4
    res = np.where(np.fabs(v) <= tol,0, np.tanh(k*v))
    return res


# nonlinear (positive) constraints
def get_c(x):
    n = (x.shape[0]-1)//3
    t_total, xk, yk, thetak = split_state(x)
    dt = t_total/(n+1)

    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    dx = xk[1:] - xk[0:-1]
    dy = yk[1:] - yk[0:-1]
    dtheta =  thetak[1:] - thetak[0:-1]
    dk = np.array([dx,dy,np.zeros(n+1)]).T

    # turning radius

    mask = np.fabs(dtheta) > 1e-5
    dtheta_stable = np.where(mask,dtheta,1)
    turn_rad = np.where(mask,np.linalg.norm(dk,axis=1)/np.fabs(2*np.sin(dtheta_stable/2)),min_turning_radius)
    c_rad = turn_rad - min_turning_radius

    # linear velocity
    qk = np.array([np.cos(thetak[:-1]),np.sin(thetak[:-1]),np.zeros(n+1)]).T
    proj_q_d = np.sum(qk * dk, axis=1)
    sign_v = sign(proj_q_d)
    vk = np.linalg.norm(dk,axis=1)*sign_v/dt
    c_vel = max_vel - np.fabs(vk)

    # angular velocity
    wk = dtheta/dt
    c_w = max_ang_vel - np.fabs(wk)

    vk = np.concatenate(([start_vel],vk))
    # acceleration (finite differences)
    ak = (vk[1:] - vk[0:-1])/dt

    c_acc = np.where(ak<0, ak + max_dec,max_acc - ak)
    c = np.concatenate((c_rad,c_vel,c_w,c_acc))
    return c


# equality (=0) constraints
def get_ceq(x):
    n = (x.shape[0]-1)//3

    t_total, xk, yk, thetak = split_state(x)
    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    dx = xk[1:] - xk[0:-1]
    dy = yk[1:] - yk[0:-1]
    dk = np.array([dx,dy,np.zeros(n+1)]).T

    # constant curvature (trapezoidal collocation, see teb paper)
    ceq = np.array([np.cos(thetak[0:-1]) + np.cos(thetak[1:]),np.sin(thetak[0:-1]) + np.sin(thetak[1:]),np.zeros(n+1)]).T
    ceq = np.fabs(np.cross(ceq,dk,axisa=1,axisb=1)[:,2])


    return ceq


def obj_func(x):
    n = (x.shape[0]-1)//3
    t_total, xk, yk, thetak = split_state(x)
    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    time_cost =  t_total**2

    # constant curvature path constraint
    path_cost = np.fabs(get_ceq(x))
    # vel, acc, radius limits
    ineq_cost = np.fabs(np.minimum(0,get_c(x)))

    # discourage sharp turns (a bit hacky but wanted paths to be less jagged)
    dist = (thetak[1:] - thetak[:-1])**2

    # teb paper said that path cost should be much larger than the rest
    # but these weights/entire cost function could probably use some tuning
    cost = 100*time_cost + np.sum( 5000*path_cost) + 10*np.sum(ineq_cost)  + 100*np.sum(dist)
    return cost

def get_bounds(nc):
    bounds = np.zeros(((nc-1)*3 + 1,2))
    bounds[0,:] = [1e-5*nc,2]
    bounds[1:nc,:] = x_lim
    bounds[nc:2*nc,:] = y_lim
    bounds[2*nc:3*nc,:] = theta_lim
    return bounds


def compute_control_inputs(x):
    n = (x.shape[0]-1)//3
    t_total, xk, yk, thetak = split_state(x)
    xk = np.concatenate(([start_x],xk,[goal_x]))
    yk = np.concatenate(([start_y],yk,[goal_y]))
    thetak = np.concatenate(([start_theta],thetak,[goal_theta]))

    dt = (n+1)/t_total
    dx = xk[1:] - xk[0:-1]
    dy = yk[1:] - yk[0:-1]
    dk = np.array([dx,dy,np.zeros(n+1)]).T
    dtheta =  thetak[1:] - thetak[0:-1]

    # vel
    qk = np.array([np.cos(thetak[:-1]),np.sin(thetak[:-1]),np.zeros(n+1)]).T
    proj_q_d =  np.sum(qk * dk, axis=1)
    sign_v = sign(proj_q_d)
    vk = np.linalg.norm(dk,axis=1)*sign_v/dt

    # steering angle
    phi_k = np.arctan(wheelbase*dtheta/vk)

    u = np.array((vk,phi_k))
    return u


def plan(s0,sf,v0,nc):
    global start_x,start_y, start_theta, start_vel, goal_theta,goal_x, goal_y, goal_theta
    start_vel = v0
    start_x = s0[0]
    start_y = s0[1]
    start_theta = s0[2]
    goal_x = sf[0]
    goal_y = sf[1]
    goal_theta = sf[2]
    sf_rs = np.copy(sf)
    sf_rs[2] *= 180.0/np.pi

    # Optimization is very sensitive to initialization
    # TODO: Fix when sign change from start to goal, causes weirdness with linspace
    init_x = np.linspace(s0[0],goal_x,nc+1)
    init_y = np.linspace(s0[1],goal_y,nc+1)
    init_theta = np.linspace(s0[2],goal_theta,nc+1)
    x0 = np.concatenate(([1],init_x[1:-1],init_y[1:-1],init_theta[1:-1]))


    ineq_constr = {'type': 'ineq',
                   'fun': get_c,
                   'jac':  jacobian(get_c)}
    eq_constr = {'type':'eq',
                 'fun': get_ceq,
                 'jac': jacobian(get_ceq)}
    bounds = get_bounds(nc)
    options = {'ftol':1e-9,'disp':True,'iprint':2}
    jac = grad(obj_func)

    def jac_reg(x):
        j = jac(x)
        if np.isfinite(j).all():
            return j
        else:
            return opt.approx_fprime(x0,obj_func,1e-6)

    # not sure which method words best. both kind of suck
    res = opt.minimize(obj_func,x0,method='SLSQP',jac=jac_reg,constraints=[eq_constr,ineq_constr],options=options,bounds=bounds)
    #res = opt.least_squares(obj_func,x0,method="trf",ftol=1e-12,xtol=1e-15,jac=jac_reg,bounds = (bounds[:,0],bounds[:,1]),verbose=2)

    # compute control frequency
    t_total, xk, yk, thetak = split_state(res.x)


    if debug:

        print("goal:" + str(sf[0])+ " " + str(sf[1])+ " " + str(sf[2]))
        print("res:" + str(xk[-1])+ " " + str(yk[-1])+ " " + str(thetak[-1]))
        xk = np.concatenate(([start_x], xk, [goal_x]))
        yk = np.concatenate(([start_y], yk, [goal_y]))
        thetak = np.concatenate(([start_theta], thetak, [goal_theta]))
        plt.plot(xk,yk,'b',xk[-1],yk[-1],'g*')
        plt.show()

    # control frequency
    fc = nc/t_total
    # compute control inputs
    uc = compute_control_inputs(res.x)
    return fc,uc, np.array([xk,yk,thetak])


# Test Script
if __name__ == '__main__':
    global debug
    debug = True
    # Generate random waypoints
    waypoints = np.random.rand(num_waypoints, 3)
    waypoints[:, 0] = (x_lim[1] - x_lim[0]) * waypoints[:, 0] + x_lim[0]
    waypoints[:, 1] = (y_lim[1] - y_lim[0]) * waypoints[:, 1] + y_lim[0]
    waypoints[:, 2] = (theta_lim[1] - theta_lim[0]) * waypoints[:, 2] + theta_lim[0]
    
    nc = 50
    s0 = np.array([0,0,0])
    v0 = 0

    for i in range(0,num_waypoints):
        sf = waypoints[i, :]
        fc, uc, wp_out = plan(s0, sf,v0,nc)
        v0 = uc[0,-1]
        s0 = wp_out[:, -1]
