# Installation of Contact-GraspNet

Create the conda env with python 3.7, tensorflow 2.5, CUDA 11.1, cudnn 8.1.0
```
conda env create -f contact_graspnet.yml
```

## Troubleshooting

- Recompile pointnet2 tf_ops:
```shell
sh compile_pointnet_tfops.sh
```


## You may encounter the following problems about open3d show:

1.``` libGL error: MESA-LOADER: failed to open swrast ```

**Solution**: 

```
cd  /usr/lib/
sudo mkdir dri
sudo ln -s /lib/x86_64-linux-gnu/dri/swrast_dri.so swrast_dri.so
```

2.``` .../libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by ...) ```

**Solution**: 

```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 {change to your anaconda3 install path}/envs/contact_graspnet/bin/../lib/libstdc++.so.6
```