# Object-based Rendering

```bash
python -m pip install submodules/diff-gaussian-rasterization-lapisgs
python -m pip install submodules/pytorch-msssim
```

If you want to use the pre-trained model with specific Gaussian resolution. The gs_res_list should only have one integer or it will cause error.
```bash
python render-lapisgs.py -m <path to pre-trained model> -s <path to COLMAP dataset> --gs_res_list <Rendering resolutions>
```

If you want to render your own Gaussians in multiple Gaussian resolutions. The length of the gs_res_list should be the same as gs_res_list or it will cause error.
```bash
python render-lapisgs.py -m <path to pre-trained model> -s <path to COLMAP dataset> --gs_path_list <paths to pre-trained ply> --gs_res_list <Rendering resolutions>
```

```bash
python render-lapisgs.py \
-s /home/syjintw/Desktop/NUS/dataset/my_testing_dataset/longdress/1051 \
-m /home/syjintw/Desktop/NUS/dataset/my_testing_gs/longdress/res1/1051 \
--gs_res_list 1
```

```bash
python render-lapisgs.py \
-s /home/syjintw/Desktop/NUS/dataset/my_testing_dataset/longdress/1051 \
-m /home/syjintw/Desktop/NUS/dataset/my_testing_gs/longdress_shift/res1/1051 \
--gs_path_list \
/home/syjintw/Desktop/NUS/dataset/my_testing_gs/longdress_shift/res1/1051/point_cloud/iteration_30000/point_cloud.ply \
/home/syjintw/Desktop/NUS/dataset/my_testing_gs/longdress_shift/res2/1051/point_cloud/iteration_30000/point_cloud.ply \
/home/syjintw/Desktop/NUS/dataset/my_testing_gs/longdress_shift/res4/1051/point_cloud/iteration_30000/point_cloud.ply \
/home/syjintw/Desktop/NUS/dataset/my_testing_gs/longdress_shift/res8/1051/point_cloud/iteration_30000/point_cloud.ply \
--gs_res_list 1 1 1 1

```

Example command
```bash
python render-lapisgs.py \
-m /home/syjintw/Desktop/NUS/dlapisgs-output/longdress/opacity/longdress_res1/dynamic_1051 \
-s /home/syjintw/Desktop/NUS/dataset/longdress/longdress_res1/1051 \
--gs_res_list 1
```

For LapisGS results, we need to modify the cfg_args file by adding "depths='', train_test_exp=False"


```bash
python render-lapisgs_exp.py \
-m /home/syjintw/Desktop/NUS/lapisgs-output/materials/opacity/materials_res8 \
-s /home/syjintw/Desktop/NUS/dataset/materials \
--gs_res_list 1 \
--skip_train --skip_test \
--our_cam_name ours_1x_res8 \
--our_cam_path /home/syjintw/Desktop/NUS/tmp/results_test_8x/transforms.json
```

```bash
python ./metrics.py \
-m /home/syjintw/Desktop/NUS/lapisgs-output/materials/opacity/materials_res8 \
--cam_name ours_1x_res8 \
--mask_dir /home/syjintw/Desktop/NUS/tmp/results_test_1x
```
