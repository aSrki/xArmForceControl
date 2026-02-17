import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
import time
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
urdf_path = "./mujoco/xarm7/xarm7.urdf" 
xml_path = "./mujoco/xarm7/scene.xml"

pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

JOINT_ID = pin_model.getFrameId("link7") 

def solve_ik(target_pose):
    q = pin.neutral(pin_model)
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-6
    
    for i in range(IT_MAX):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        
        dMi = pin_data.oMf[JOINT_ID].inverse() * target_pose
        err = pin.log(dMi).vector
            
        J = pin.computeFrameJacobian(pin_model, pin_data, q, JOINT_ID, pin.ReferenceFrame.LOCAL)
        
        v = np.linalg.solve(J.T @ J + damp * np.eye(pin_model.nv), J.T @ err)
        
        q = pin.integrate(pin_model, q, v * DT)
        
    return q

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

target_pose = pin.SE3.Identity()
target_pose.translation = np.array([0.4, 0.1, 0.0])
target_pose.rotation = pin.utils.rotate('x', np.pi)

pin_model.gravity.linear = np.array([0, 0, -9.81])

target_pos = np.array([0.3, 0.3, 0.5])

target_rot = pin.utils.rpyToMatrix(np.pi, 0, 0) 

target_pose = pin.SE3(target_rot, target_pos)

q_des = solve_ik(target_pose)

K_pos = np.diag([400, 400, 400])
K_ori = np.diag([20, 20, 20])
D_pos = np.diag([50, 50, 50])  
D_ori = np.diag([10, 10, 10])

print(f"QDES = {q_des}")

x = []
y = []
z = []

force = []

sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "external_force_sensor")
adr = mj_model.sensor_adr[sensor_id]

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    time.sleep(1)
    sim_start = time.time()
    try:
        while viewer.is_running():
            sim_timestep = mj_model.opt.timestep
            step_start_time = time.time()

            q = mj_data.qpos[:7]
            v = mj_data.qvel[:7]

            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)

            p_curr = pin_data.oMf[JOINT_ID].translation
            x.append(p_curr[0])
            y.append(p_curr[1])
            z.append(p_curr[2])
            R_curr = pin_data.oMf[JOINT_ID].rotation

            J = pin.computeFrameJacobian(pin_model, pin_data, q, JOINT_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            err_p = target_pos - p_curr

            R_err = target_rot @ R_curr.T
            err_o = pin.log3(R_err)

            x_dot = J @ v
            v_curr = x_dot[:3]
            w_curr = x_dot[3:]

            f_task = K_pos @ err_p - D_pos @ v_curr
            m_task = K_ori @ err_o - D_ori @ w_curr
            F_ext = np.concatenate([f_task, m_task])

            tau_dyn = pin.nonLinearEffects(pin_model, pin_data, q, v)
            tau = J.T @ F_ext + tau_dyn

            mj_data.qfrc_applied[:7] = tau
            mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            dt = time.time() - step_start_time
            sleep = max(0, sim_timestep - dt)
            time.sleep(sleep)

            if(4 > time.time() - sim_start > 3):
                force_vector = np.array([0, 0, 50, 0, 0, 0])
                mj_data.xfrc_applied[body_id] = force_vector
                measured_force = mj_data.sensordata[adr : adr+3]
                force.append(measured_force[2])

    except KeyboardInterrupt:
        ax1.plot3D(x,y,z)
        ax2.plot(force)
        plt.show()