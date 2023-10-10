import os
import subprocess
import sys
import argparse

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Colmap Directory')
    parser.add_argument("--dir", required=True, type=str)
    args = parser.parse_args()
    dir = args.dir
    
    pIntrisics = subprocess.Popen(["colmap", "matches_importer",
                                "--database_path", os.path.join(dir, "database.db"),
                                "--match_list_path", os.path.join(dir, "image_pairs_to_match.txt"),
                                "--SiftMatching.num_threads", "32",
                                "--SiftMatching.min_num_inliers", "18"]
                                )
    pIntrisics.wait()

    #if not os.path.exists(os.path.join(dir, "sparse")):
    pIntrisics = subprocess.Popen(["rm", "-rf", os.path.join(dir, "sparse")])
    pIntrisics.wait()
    os.mkdir(os.path.join(dir, "sparse"))
    pIntrisics = subprocess.Popen(["colmap", "mapper",
                                "--database_path", os.path.join(dir, "database.db"),
                                "--image_path", os.path.join(dir, "images"),
                                "--output_path", os.path.join(dir, "sparse"),
                                #"--Mapper.multiple_models", "0", # 1
                                "--Mapper.min_num_matches", "20", # 15
                                "--Mapper.ba_local_num_images", "6", # 6
                                "--Mapper.max_model_overlap", "60", # 20
                                #"--Mapper.init_min_num_inliers", "20", # 100
                                #"--Mapper.init_max_error", "6", # 4
                                #"--Mapper.abs_pose_min_num_inliers", "15", # 30
                                #"--Mapper.abs_pose_min_inlier_ratio", "0.15", # 0.25
                                #"--Mapper.ba_global_use_pba", "1",
                                #"--Mapper.ba_global_pba_gpu_index", "2",
                                #"--Mapper.ba_local_max_num_iterations", "40", # 25
                                "--Mapper.ba_global_max_num_iterations", "100"]
                                )
    pIntrisics.wait()
    pIntrisics = subprocess.Popen(["colmap", "model_converter",                               
                                "--input_path", os.path.join(dir, "sparse", "0"),
                                "--output_path", os.path.join(dir, "sparse", "0", "sparse.ply"),
                                "--output_type", "PLY"]
                                )
    pIntrisics.wait()
    pIntrisics = subprocess.Popen(["colmap", "model_converter",                               
                                "--input_path", os.path.join(dir, "sparse", "0"),
                                "--output_path", os.path.join(dir, "sparse", "0"),
                                "--output_type", "TXT"]
                                )
    pIntrisics.wait()
    '''

    if not os.path.exists(os.path.join(dir, "dense")):
        os.mkdir(os.path.join(dir, "dense"))
    pIntrisics = subprocess.Popen(["colmap", "image_undistorter",
                                "--image_path", os.path.join(dir, "images"),
                                "--input_path", os.path.join(dir, "sparse","0"),
                                "--output_path", os.path.join(dir, "dense"),
                                "--output_type", "COLMAP",
                                "--max_image_size", "2000"]
                                )
    pIntrisics.wait()

    pIntrisics = subprocess.Popen(["colmap", "patch_match_stereo",
                                "--workspace_path", os.path.join(dir, "dense"),
                                "--workspace_format", "COLMAP",
                                "--PatchMatchStereo.geom_consistency", "true",
                                "--PatchMatchStereo.gpu_index", "0"]
                                )
    pIntrisics.wait()

    pIntrisics = subprocess.Popen(["colmap", "stereo_fusion",
                                "--workspace_path", os.path.join(dir, "dense"),
                                "--workspace_format", "COLMAP",
                                "--input_type", "geometric",
                                "--output_path", os.path.join(dir, "dense", "fused.ply"),
                                "--StereoFusion.num_threads", "16"]
                                )
    pIntrisics.wait()

    pIntrisics = subprocess.Popen(["colmap", "poisson_mesher",
                                "--input_path", os.path.join(dir, "dense", "fused.ply"),
                                "--output_path", os.path.join(dir, "dense", "meshed-poisson.ply"),
                                "--PoissonMeshing.num_threads", "16"]
                                )
    pIntrisics.wait()
    '''
    
    # 需要CGAL
    '''
    pIntrisics = subprocess.Popen(["colmap", "delaunay_mesher",
                                "--input_path", os.path.join(dir, "dense"),
                                "--output_path", os.path.join(dir, "dense", "meshed-delaunay.ply")]
                                )
    pIntrisics.wait()
    '''

   