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
    
    # 相关的路径
    pIntrisics = subprocess.Popen(["colmap", "model_converter",                               
                               "--input_path", os.path.join(dir, "sparse", "0"),
                               "--output_path", os.path.join(dir, "sparse", "0", "sparse.ply"),
                               "--output_type", "PLY"]
                              )
    pIntrisics.wait()
    '''
    '''

    #if not os.path.exists(os.path.join(dir, "dense")):
    pIntrisics = subprocess.Popen(["rm", "-rf", os.path.join(dir, "dense")])
    pIntrisics.wait()
    os.mkdir(os.path.join(dir, "dense"))
    pIntrisics = subprocess.Popen(["colmap", "image_undistorter",
                                "--image_path", os.path.join(dir, "images"),
                                "--input_path", os.path.join(dir, "sparse","0"),
                                "--output_path", os.path.join(dir, "dense"),
                                "--output_type", "COLMAP",
                                "--max_image_size", "2400"]
                                )
    pIntrisics.wait()

    pIntrisics = subprocess.Popen(["colmap", "patch_match_stereo",
                                "--workspace_path", os.path.join(dir, "dense"),
                                "--workspace_format", "COLMAP",
                                "--PatchMatchStereo.geom_consistency", "true",
                                "--PatchMatchStereo.window_radius", "15", # 5
                                "--PatchMatchStereo.num_samples", "30", # 15
                                "--PatchMatchStereo.write_consistency_graph", "1", # 0
                                "--PatchMatchStereo.gpu_index", "2"]
                                )
    pIntrisics.wait()

    pIntrisics = subprocess.Popen(["colmap", "stereo_fusion",
                                "--workspace_path", os.path.join(dir, "dense"),
                                "--workspace_format", "COLMAP",
                                "--input_type", "geometric", # photometric, geometric
                                "--output_path", os.path.join(dir, "dense", "fused.ply"),
                                "--StereoFusion.min_num_pixels", "2", # 5
                                "--StereoFusion.max_reproj_error", "2", # 2
                                "--StereoFusion.max_depth_error", "0.005", # 0.01
                                "--StereoFusion.max_normal_error", "2", # 10
                                #"--StereoFusion.use_cache", "1",
                                "--StereoFusion.num_threads", "60"]
                                )
    pIntrisics.wait()
    
    pIntrisics = subprocess.Popen(["colmap", "poisson_mesher",
                                "--input_path", os.path.join(dir, "dense", "fused.ply"),
                                "--output_path", os.path.join(dir, "dense", "meshed-poisson.ply"),
                                "--PoissonMeshing.depth", "16", # 13
                                "--PoissonMeshing.color", "32", # 32
                                "--PoissonMeshing.trim", "5", # 10
                                "--PoissonMeshing.num_threads", "64"]
                                )
    pIntrisics.wait()

    '''
    pIntrisics = subprocess.Popen(["colmap", "delaunay_mesher",
                                "--input_path", os.path.join(dir, "dense"),
                                "--output_path", os.path.join(dir, "dense", "meshed-delaunay.ply")]
                                )
    pIntrisics.wait()
    '''

   