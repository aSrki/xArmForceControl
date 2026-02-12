import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
import time

urdf_path = "./mujoco/xarm7/xarm7.urdf" 
xml_path = "./mujoco/xarm7/scene.xml"

pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

JOINT_ID = pin_model.getFrameId("link7") 

def solve_ik(target_pose):
    q = pin.neutral(pin_model)
    eps = 1e-4
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-6
    
    for i in range(IT_MAX):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        
        dMi = pin_data.oMf[JOINT_ID].inverse() * target_pose
        err = pin.log(dMi).vector
        
        if np.linalg.norm(err) < eps:
            print(f"Convergence reached at iteration {i}")
            break
            
        J = pin.computeFrameJacobian(pin_model, pin_data, q, JOINT_ID, pin.ReferenceFrame.LOCAL)
        
        v = np.linalg.solve(J.T @ J + damp * np.eye(pin_model.nv), J.T @ err)
        
        q = pin.integrate(pin_model, q, v * DT)
        
    return q

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

target_pose = pin.SE3.Identity()
target_pose.translation = np.array([0.4, 0.1, 0.0])
target_pose.rotation = pin.utils.rotate('x', np.pi)

pin_model.gravity.linear = np.array([0, 0, -9.81])

target_pos = np.array([0.3, 0.3, 0.5])

target_rot = pin.utils.rpyToMatrix(3.14, 0, 0) 

target_pose = pin.SE3(target_rot, target_pos)

q_des = solve_ik(target_pose)

Kp = np.diag([55.0] * pin_model.nv)
Kd = np.diag([1.0] * pin_model.nv)

print(f"QDES = {q_des}")

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    time.sleep(1)
    while viewer.is_running():
        sim_timestep = mj_model.opt.timestep
        step_start_time = time.time()

        q = mj_data.qpos[:pin_model.nv]
        dq = mj_data.qpos[:pin_model.nv]
        q_error = q_des - q

        gravity_torques = pin.computeGeneralizedGravity(pin_model, pin_data, q)
        tau = Kp@q_error - Kd@dq + gravity_torques
        mj_data.qfrc_applied[:7] = tau
        
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()

        dt = time.time() - step_start_time
        sleep = max(0, sim_timestep - dt)
        time.sleep(sleep)
