import os
import sys
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
from contact_graspnet import *


class ContactGraspNet:  
    def __init__(self, global_config, checkpoint_dir, gpu_id=0):
        """
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param gpu_id: GPU ID
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
        
        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        saver = tf.train.Saver(save_relative_paths=True)
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')
        
        os.makedirs('results', exist_ok=True)

    def predict(self, input_path, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
        """
        Predict 6-DoF grasp distribution for given model and input data
        
        :param input_path: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """
        # Load available input data
        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(input_path, K=K)
        
        # Visualize input data
        show_image(rgb, segmap)
        
        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)

        print('Generating Grasps...')
        
        # Predict grasps
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(input_path.replace('png','npz').replace('npy','npz'))), 
                  pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Visualize results          
        visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
                
        return pred_grasps_cam, scores, contact_pts


if __name__ == "__main__":

    # Params
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/0.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=True, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()
    
    global_config = load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    # Init
    grasp_net = ContactGraspNet(global_config, FLAGS.ckpt_dir)
    
    # Predict
    pred_grasps_cam, scores, contact_pts = grasp_net.predict(
        FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, 
        z_range=eval(str(FLAGS.z_range)),
        K=FLAGS.K, 
        local_regions=FLAGS.local_regions, 
        filter_grasps=FLAGS.filter_grasps, 
        segmap_id=FLAGS.segmap_id, 
        forward_passes=FLAGS.forward_passes, 
        skip_border_objects=FLAGS.skip_border_objects
    )
    
    print(pred_grasps_cam, scores, contact_pts)
    print(type(pred_grasps_cam))
    print(type(scores))
    print(type(contact_pts))
    print(pred_grasps_cam.keys())
    print(scores.keys())
    print(contact_pts.keys())
    # print(pred_grasps_cam["-1"])