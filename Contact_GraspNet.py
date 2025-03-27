import os
import sys
# import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from Utils import serialize, deserialize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
from contact_graspnet import *
from yacs.config import CfgNode as CN


class ContactGraspNet:  
    def __init__(self, config):
        """
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param gpu_id: GPU ID
        """
        
        self.config = config

        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_id)
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[self.config.gpu_id], True)
        
        # Build the model
        global_config = load_config(self.config.ckpt_dir, batch_size=self.config.forward_passes, arg_configs=self.config.arg_configs)
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Create a session
        session_cfg = tf.ConfigProto()
        session_cfg.gpu_options.allow_growth = True
        session_cfg.allow_soft_placement = True
        self.sess = tf.Session(config=session_cfg)

        # Load weights
        saver = tf.train.Saver(save_relative_paths=True)
        self.grasp_estimator.load_weights(self.sess, saver, self.config.ckpt_dir, mode='test')
        
        os.makedirs('results', exist_ok=True)

    def predict(self, input_path):
        """ Predict 6-DoF grasp distribution for given model and input data
        
        :param input_path: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
        """
        # Load available input data
        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(input_path, K=self.config.K)
        
        # Visualize input data
        if self.config.visualize:
            show_image(rgb, segmap)
        
        if segmap is None and (self.config.local_regions or self.config.filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=self.config.skip_border_objects, z_range=self.config.z_range)

        print('Generating Grasps...')
        
        # Predict grasps
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=self.config.local_regions, filter_grasps=self.config.filter_grasps, forward_passes=self.config.forward_passes)  

        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(input_path.replace('png','npz').replace('npy','npz'))), 
                  pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Visualize results 
        if self.config.visualize:      
            visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
                
        return pred_grasps_cam, scores, contact_pts


if __name__=="__main__":
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    input = deserialize("./data.pkl")
    
    # Load contact_graspnet module config
    with open(f"./Config/{input['contact_grapsnet_cfg']}.yaml", "r") as file:
        config = CN.load_cfg(file)
    
    input_path = input["input_path"]
    grasp_net = ContactGraspNet(config)
    pred_grasps_cam, scores, contact_pts = grasp_net.predict(input_path)
    
    output = {
        "pred_grasps_cam": pred_grasps_cam,   
        "scores": scores,
        "contact_pts": contact_pts
    }
    serialize(output, "./data.pkl")
    

# if __name__ == "__main__":
    
#     # Init
#     with open("./Config/default.yaml", "r") as file:
#         config = CN.load_cfg(file)
    
#     grasp_net = ContactGraspNet(config)
    
#     input_path = 'test_data/0.npy'
    
#     # Predict
#     pred_grasps_cam, scores, contact_pts = grasp_net.predict(input_path)