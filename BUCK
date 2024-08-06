load("@tf//build_defs:package_app.bzl", "package_app")
load("@tf//:defs.bzl", "tf_default_shared_deps")
load("@prelude//paths.bzl", "paths")


cxx_binary(
    name = "app",
    srcs = ["GaussianSplatter.cpp"],
    link_style = "static",
    deps = [
        "@tf//:TF",
    ],
    extra_shared_deps = tf_default_shared_deps(),
    #_cxx_toolchain = "tf//toolchain:cxx",
    visibility = ['PUBLIC']
)

# fsl_library(
#     name = "fsl",
#     srcs = ["Shaders/FSL/ShaderList.fsl", "@tf//:UI_ShaderList", "@tf//:Font_ShaderList"],
#     visibility = ['PUBLIC']
# )

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
    #  ("",  "//01_Transformations:fsl")
    ] , 
    files = 
        {
              "GPUCfg/gpu.data": "@tf//:pc_gpu.data",
              "GPUCfg/gpu.cfg": "//:gpu.cfg",
        } | 
    {paths.join("drjohnson",paths.basename(file)): file for file in glob(["assets/db/drjohnson/sparse/0/*"])}
) 

