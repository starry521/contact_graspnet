import numpy as np
from Utils import Submodule

contact_graspnet = Submodule()
input_data = {
    "contact_grapsnet_cfg": "default",
    "input_path": "test_data/0.npy"
}
output = contact_graspnet.call(input_data, "contact_graspnet_env", "Perception/Grasp_Pose_Estimation/Contact_GraspNet/Contact_GraspNet.py")

print(f"\nresults:")
pred_grasps_cam, scores, contact_pts = output["pred_grasps_cam"], output["scores"], output["contact_pts"] 

for object_id in pred_grasps_cam.keys():
    object_pred_grasps_cam, object_scores, object_contact_pts = pred_grasps_cam[object_id], scores[object_id], contact_pts[object_id]
    object_grasp_pose_num = object_pred_grasps_cam.shape[0]
    if object_grasp_pose_num==0:
        print(f"Object {object_id} can not predict grasp pose!")
        continue
    max_idx = np.argmax(object_scores)
    max_score = object_scores[max_idx]
    print(f"Object {object_id} predict {object_grasp_pose_num} grasp pose, best grasp pose: {object_pred_grasps_cam[max_idx]}, score: {object_scores[max_idx]}, contact points: {object_contact_pts[max_idx]}\n")