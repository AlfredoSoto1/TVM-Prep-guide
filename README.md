# Model Compilation with TVM
---

This project aims to build a guide on how-to use TVM to cross-compile different pre-trained ML models 
to specific target architectures. 

Our approach was to compile multiple pre-trained models with TVM, document each step of the process
to serve as future reference on how-to cross-compile and validate the results by executing the compiled
models through a C++ wrapper on the target platforms.

## Semester progress-review

- Deep research about TVM and its compilation workflow
- Breif research on how LLVM and MLIR are related to TVM and how TVM works internally with these tools.
- Research and selection of different kind of pre-trained models to test the cross-compilation process.
- Runned and tested a selected pre-trained model (resnet18) as first choice to develop the guide.
- Research about generated artifacts after compilation (dynamic libraries & parameters)
- C++ wrapper design to propperly call the cross-compiled code from the dynamic library generated.
- Research and selection on other pre-trained models to also test compilation process.
- Updated C++ wrappper to test the selected pre-trained models after compilaton.
- Validated output results from the compiled models.

## Tools envolved for this research progress
- Docker to run local devcontainer and mount the repository with the necessary tools and resources to
start compiling with TVM.
- Compiled official TVM repository to make use of their C++ libraries and tool set for the C++ wrapper.
- For models that make use of images, we used an open-source header-only C/C++ file to propperly read images of type (JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC).
  - `stb_image.h` https://github.com/nothings/stb
- To propperly compile the C++ wrapper, we used CMake to link dynamic libraries compiled by TVM and official TVM runtime & other TVM toolset from the official repository.
- TVM with python to compile the pre-trained models to dynamic library.
- For the models used to compile:
  - resnet18
  - squeeze
  - squeezeNet
  - shuffleNet
  - mobileNet
- Official TVM documentation

## Structure

```
TVM-PREP-GUIDE/
├── .devcontainer/
├── .vscode/
├── notebooks/
│   ├── compile_onnx.ipynb              
│   ├── MobileNetV2_preparation.ipynb   
│   ├── resnet.ipynb                    
│   ├── ShuffleNetV2_preparation.ipynb  
│   ├── SqueezeNet.ipynb                
│   ├── tvm_random.ipynb                
│   └── tvm_to_lib_deprecated.ipynb     
├── tvm/
│   └──...
├── tvm_cpp/
│   ├── artifacts/
│   ├── build/
│   ├── images/
│   ├── .clang-format
│   ├── CMakeLists.txt
│   ├── compile_resnet.sh
│   ├── main_old.cpp
│   ├── main.cpp
│   ├── README.md
│   ├── run_model.cpp
│   ├── run_resnet.cpp
│   ├── run_tvm.cpp
│   ├── stb_image.h
│   └── TARGET_INSTRUCTIONS.md
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── TVM-Prep-Guide.md
```

### Folder contents
- `/notebooks` folder contains all python notebooks with examples on how to compile each model.
- `/tvm` folder contains the official TVM repository at version `v0.19.0`
- `/tvm_cpp` folder contains the C++ wrapper and external libraries to run the compiled models.
  - `/artifacts` contain the genrated labels, dynamic library and parameters generated after running your selected model on each notebook with TVM
  - `/build` used to build and run the C++ wrapper
  - `/images` contains selective images related to models that make use of images (ex: resnet18)  

--- 
## How to run?
After cloning the repository, you have to reopen the local folder (where this readme is located) as a devcontainer throgh VSCode. This is going to take time while it builds the docker image and runs the container with its propper dependencies. 

## What you should see after?
After the docker container starts, python will get installed and the `requirements.txt` get included. To run a basic model, you have to compile `tvm` using the `TVM-Prep-Guide.md` file. This includes all the instructions on how to compile the propper version of TVM. 
* Note: This takes a few minutes while complies.

## Compiling a pre-trained model
You can choose one of the following models from `/notebooks` to complile with tvm. To start, you can use `resnet18` notebook. After running and following each step on the notebook, you can copy/paste the generated files: 
- `resnet18_tvm.json`
- `resnet18_tvm.params`
- `resnet18_tvm.so`, 
- `labels.txt` 

into `/tvm_cpp/artifacts/`.

## Compiling C++ wrapper
If you don't have a `build` folder in `tvm_cpp`:
```sh
mkdir build; cd build
```

If you have one already compile the wrapper by doing:
```sh
cmake ..

make
```

After this, you can run the wrapper model within the build folder:
```sh
./run_model resnet18 ../artifacts ../images/cat.png ../artifacts/labels.txt input0 --normalize
```
Make sure that the images and artifacts are propperly located in the folders displayed above.

You should see an output like:
```sh
Original image: 1200x1200x3
Resized to: 256x256
Final image size: 224x224x3
Color order: RGB
Pixel scaling: 0.0 to 1.0 (PyTorch-style)
Mean (R,G,B): [0.485, 0.456, 0.406]
Std (R,G,B):  [0.229, 0.224, 0.225]
Input tensor range: [-2.03571, 2.64], mean: 0.0921071
Output shape: 1 1000 
Logits range: [-6.20995, 14.4448], mean: 2.63146e-05

===== Running resnet18 =====
Top-5 Predictions:
  1) id=282  p=0.908828  tiger cat
  2) id=287  p=0.0444733  lynx
  3) id=281  p=0.029566  tabby
  4) id=285  p=0.00674846  Egyptian cat
  5) id=292  p=0.00368929  tiger
Predicted class: 282 (tiger cat)
```

---
### Conclusion & Future work
After research and implementation, we successfully build an example-guide to cross-compile models onto different target architectures. With these pre-trained models (in each notebook) we get a good view on how TVM operates and compiles. This can be used as reference for future work compiling with TVM. Although, the official TVM documentation is extensive and robust, with this guide you can get a practical glimpse on how to copmpile and run the model for practical use. 

As for upgrades and future work, compilation for CUDA and other API's that integrates the use for GPU to run and compile models is something to be done and tested. As for this project, it will also be useful to expand this guide with examples on how to work with the GPU with no manual setup or issues.

Expanding out of the scope of this research-project experience, depending on the use case and model used extensive benchmarking would be necessary to test the speed of the model running with the python runtime and after compiled with TVM. 

### Extra Notes
Each folder contains a `README.md` with detailed information about the things that happen there and extra parameters to tweak the results to something you might consider as expected.