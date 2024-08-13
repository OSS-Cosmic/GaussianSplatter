load("@tf//build_defs:package_app.bzl", "package_app")
load("@tf//:defs.bzl", "tf_default_shared_deps")
load("@prelude//paths.bzl", "paths")
load("@tf//build_defs:fsl_library.bzl", "fsl_library")


cxx_binary(
    name = "app",
    srcs = ["GaussianSplatter.cpp"],
    link_style = "static",
    deps = [
        "@tf//:TF",
        "@tf//:TF~Ply"
    ],
    extra_shared_deps = tf_default_shared_deps(),
    #_cxx_toolchain = "tf//toolchain:cxx",
    visibility = ['PUBLIC']
)

fsl_library(
    name = "fsl",
    srcs = ["Shaders/FSL/ShaderList.fsl", "@tf//:UI_ShaderList", "@tf//:Font_ShaderList"],
    visibility = ['PUBLIC']
)

export_file(
    name = "gpu.cfg",
    mode = "reference",
    src = "gpu.cfg",
    visibility = ['PUBLIC']
)

package_app(
    name = "GaussianSplatter",
    resources = [ 
      ("", "//:app"),
      ("",  "//:fsl")
    ] , 
    files = 
        {
              "GPUCfg/gpu.data": "@tf//:pc_gpu.data",
              "GPUCfg/gpu.cfg": "//:gpu.cfg",
        } | 
    {paths.relativize(file,"TFSamples-Media/UnitTestResources"): file for file in glob(["TFSamples-Media/UnitTestResources/Fonts/**/*"])} |
    {paths.join("Other",paths.relativize(file, "Models")): file for file in glob(["Models/**/*.ply"])}
) 

