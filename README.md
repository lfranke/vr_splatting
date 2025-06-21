# TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering

<div style="text-align: center;">Linus Franke, Laura Fink, Marc Stamminger</div>

Recent advances in novel view synthesis have demonstrated impressive results in fast photorealistic scene rendering through differentiable point rendering, either via Gaussian Splatting (3DGS) (Kerbl et al., 2023) or neural point rendering (Aliev et al., 2020). Unfortunately, these directions require either a large number of small Gaussians or expensive per-pixel post-processing for reconstructing fine details, which negatively impacts rendering performance. To meet the high performance demands of virtual reality (VR) systems, primitive or pixel counts therefore must be kept low, affecting visual quality.

In this paper, we propose a novel hybrid approach based on foveated rendering as a promising solution that combines the strengths of both point rendering directions regarding performance sweet spots. Analyzing the compatibility with the human visual system, we find that using a low-detailed, few primitive smooth Gaussian representation for the periphery is cheap to compute and meets the perceptual demands of peripheral vision. For the fovea only, we use neural points with a convolutional neural network for the small pixel footprint, which provides sharp, detailed output within the rendering budget. This combination also allows for synergistic method accelerations with point occlusion culling and reducing the demands on the neural network.

Our evaluation confirms that our approach increases sharpness and details compared to a standard VR-ready 3DGS configuration, and participants of a user study overwhelmingly preferred our method. Our system meets the necessary performance requirements for real-time VR interactions, ultimately enhancing the user’s immersive experience. The project page can be found at: https://lfranke.github.io/vr_splatting


## Citation

```
@article{franke2025vrsplatting,
    title={VR-Splatting: Foveated Radiance Field Rendering via 3D Gaussian Splatting and Neural Points},
    author={Linus Franke and Laura Fink and Marc Stamminger},
    journal = {Proc. ACM Comput. Graph. Interact. Tech.},
    year = {2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {8},
    number = {1},
    month = may,
    articleno = {18},
    url = {https://doi.org/10.1145/3728302},
    doi = {10.1145/3728302}
}

```


## Install Requirements

Supported Operating Systems: Windows
Supported VR Headsets: HTC Vive Pro Eye

Other OS or VR Headsets are unlikely to work, as there are driver specific implementations included in this repo.

### Compile 

Please follow TRIPS for compiling and running: [[Github]](https://github.com/lfranke/TRIPS)

Afterwards add ``openvr`` to the ``External/`` folder

### License

Please note the license of [[3D Gaussian Splatting]](https://github.com/graphdeco-inria/gaussian-splatting), [[STP]](https://github.com/r4dl/StopThePop) and [[gaussian-splatting-cuda]](https://github.com/MrNeRF/gaussian-splatting-cuda), which this project builds upon

